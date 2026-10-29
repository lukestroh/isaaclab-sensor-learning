#!/usr/bin/env python3
"""
pymoo problem for the sensor-rig search. One evaluation = decode the genome, check feasibility, then (if needed)
simulate the rig in its own Isaac Sim process and read its metrics.json.

Objectives (minimized, see moo/objectives.py): [-coverage, time to reference coverage, sensor cost].
Constraints (<= 0 is feasible):
    G_spacing: Lloyd sites too close for the sensors' footprints (checked before simulating; not simulated if > 0)
    G_failed:  the simulation failed (crash, CUDA OOM, timeout)
"""
import threading

import numpy as np
from pymoo.core.problem import ElementwiseProblem, LoopedElementwiseEvaluation

from moo import rig_codec as rc
from moo.objectives import N_OBJ, objectives_from_metrics, placeholder_objectives
from moo.run_dir import RunDir

N_IEQ_CONSTR = 2
_RUNTIME = ["elementwise_runner", "run_dir", "evaluator", "progress", "_flight_lock", "_in_flight"]  # see attach()


class SensorRigProblem(ElementwiseProblem):
    def __init__(
        self,
        space: rc.SearchSpace,
        catalog: rc.SensorCatalog,
        time_budget_s: float,
        run_dir: RunDir | None = None,
        evaluator=None,
        elementwise_runner=None,
        max_consecutive_failures: int = 5,
        progress=None,
    ):
        """
        :param evaluator: callable(rig_yaml_path, out_dir) -> metrics dict (e.g. evaluators.SubprocessEvaluator).
        :param time_budget_s: NBV time budget; used for placeholder objectives of rigs that were not simulated.
        :param progress: optional moo.progress.GenerationProgress (live per-generation progress bar).
        :param max_consecutive_failures: stop the run (RuntimeError) after this many simulations fail in a row, e.g. a
            setup bug that fails every rig; 0 disables. The run can be resumed once the cause is fixed.
        """
        super().__init__(
            n_var=space.n_var,
            n_obj=N_OBJ,
            n_ieq_constr=N_IEQ_CONSTR,
            xl=space.xl,
            xu=space.xu,
            vtype=int,
            exclude_from_serialization=_RUNTIME,
        )
        self.space = space
        self.catalog = catalog
        self.time_budget_s = float(time_budget_s)
        self.generation = 1  # set by CheckpointCallback; recorded in rig yamls and the index
        self.max_consecutive_failures = max_consecutive_failures
        self.consecutive_failures = 0
        self.attach(run_dir, evaluator, elementwise_runner, progress)

    def attach(self, run_dir: RunDir, evaluator, elementwise_runner=None, progress=None) -> None:
        """(Re)attach the parts that cannot be pickled, e.g. after loading a checkpoint."""
        self.run_dir = run_dir
        self.evaluator = evaluator
        self.elementwise_runner = elementwise_runner or LoopedElementwiseEvaluation()
        self.progress = progress
        self._flight_lock = threading.Lock()
        self._in_flight: dict[str, threading.Event] = {}

    # ------------------------------------------------------------------------------------------------------------------

    def evaluate(self, X, *args, **kwargs):
        """pymoo evaluates one generation's new rigs per call; wrap it in that generation's progress bar."""
        if self.progress is None or np.ndim(X) != 2:
            return super().evaluate(X, *args, **kwargs)
        self.progress.start(self.generation, len(X))
        try:
            return super().evaluate(X, *args, **kwargs)
        finally:
            self.progress.close()

    def _report(self, row: dict, cached: bool) -> None:
        if self.progress is not None:
            coverage = row.get("coverage_mean")
            coverage = float(coverage) if coverage not in (None, "") else None
            self.progress.rig_done(row["rig_id"], row["status"], coverage, cached=cached)

    def _evaluate(self, x, out, *args, **kwargs):
        genome = rc.Genome.from_x(x, self.space)
        design = rc.decode(genome, self.space, self.catalog)
        row = self._evaluate_rig(design)
        out["F"] = [float(row["F1"]), float(row["F2"]), float(row["F3"])]
        out["G"] = [float(row["G_spacing"]), float(row["G_failed"])]

    def _evaluate_rig(self, design: rc.RigDesign) -> dict:
        """Cached result for this rig, waiting if another thread is already simulating it."""
        while True:
            cached = self.run_dir.cached(design.rig_id)
            if cached is not None:
                self._report(cached, cached=True)
                return cached
            with self._flight_lock:
                event = self._in_flight.get(design.rig_id)
                if event is None:
                    self._in_flight[design.rig_id] = threading.Event()
                    break
            event.wait()
        try:
            row = self._run(design)
            self.run_dir.record(row)
            self._report(row, cached=False)
            self._count_failures(row)
            return self.run_dir.cached(design.rig_id)
        finally:
            with self._flight_lock:
                self._in_flight.pop(design.rig_id).set()

    def _count_failures(self, row: dict) -> None:
        if row["status"] == "infeasible":  # not simulated: says nothing about the simulation
            return
        with self._flight_lock:
            self.consecutive_failures = self.consecutive_failures + 1 if row["status"] == "failed" else 0
            n = self.consecutive_failures
        if self.max_consecutive_failures and n >= self.max_consecutive_failures:
            raise RuntimeError(
                f"{n} simulations failed in a row; stopping the run. Last error ({row['rig_id']}): {row.get('error')}. "
                f"See {self.run_dir.eval_dir(row['rig_id'])}/sim.log, fix the cause, then --resume {self.run_dir.run_id} "
                "(failed rigs are retried)."
            )

    def _run(self, design: rc.RigDesign) -> dict:
        g_spacing = design.spacing_violation if np.isfinite(design.spacing_violation) else -1.0
        row = {
            "rig_id": design.rig_id,
            "generation": self.generation,
            "G_spacing": g_spacing,
            "n_sensors": len(design.models),
            "radius": design.radius,
            "colatitude": design.colatitude,
            "models": "|".join(design.models),
            "cost": design.cost,
        }
        rig = rc.rig_yaml_dict(design, self.space, self.catalog, run_id=self.run_dir.run_id, generation=self.generation)
        rig_yaml = rc.write_rig_yaml(self.run_dir.rig_yaml(design.rig_id), rig)

        if not design.feasible:  # never simulated
            f = placeholder_objectives(design.cost, self.time_budget_s)
            return {**row, "status": "infeasible", "F1": f[0], "F2": f[1], "F3": f[2], "G_failed": -1.0}

        if self.progress is not None:
            self.progress.rig_running(design.rig_id, self.run_dir.eval_dir(design.rig_id))
        metrics = self.evaluator(rig_yaml, self.run_dir.eval_dir(design.rig_id))
        row["wall_s"] = metrics.get("wall_s")
        try:
            summary = objectives_from_metrics(metrics, design.cost)
        except (ValueError, KeyError) as e:
            f = placeholder_objectives(design.cost, self.time_budget_s)
            return {**row, "status": "failed", "F1": f[0], "F2": f[1], "F3": f[2], "G_failed": 1.0,
                    "error": str(metrics.get("error", e))[:500]}  # fmt: skip
        f = summary.pop("F")
        return {**row, **summary, "status": "ok", "F1": f[0], "F2": f[1], "F3": f[2], "G_failed": -1.0}
