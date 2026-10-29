#!/usr/bin/env python3
"""
Per-generation checkpointing and progress logging for the NSGA-II run.

After every generation: write the population (generations/gen_XXXX/population.csv), append progress.csv (hypervolume,
Pareto-front size, best coverage), and dill the whole algorithm to checkpoint.pkl. A run resumes from that checkpoint
with ``minimize(problem, algorithm, termination, copy_algorithm=False)`` (see run_nsga2.py). Rigs evaluated after the
last checkpoint are not lost either: their results are in the run's index.csv and are reused, not re-simulated.
"""
import csv
import os
import time

import dill
import numpy as np
from pymoo.core.callback import Callback
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from moo import rig_codec as rc
from moo.run_dir import RunDir

PROGRESS_FIELDS = [
    "generation", "n_evaluated", "n_feasible", "n_pareto", "hv_normalized", "hypervolume", "best_coverage", "best_rig",
    "wall_s",
]  # fmt: skip
IDEAL_POINT = (-1.0, 0.0, 0.0)  # best possible objectives: full coverage, zero time, zero cost


class CheckpointCallback(Callback):
    """
    :param hv_ref_point: hypervolume reference point (worst corner of objective space), e.g.
        [0 coverage, 2 x time budget, max rig cost]; only feasible points that dominate it count.

    Hypervolume (HV) = volume of objective space dominated by the feasible front, bounded by hv_ref_point. It grows
    whenever the front improves (more coverage, less time, or lower cost for some trade-off). ``hv_normalized`` divides
    by the box between IDEAL_POINT and hv_ref_point: the fraction of that box the front dominates, in [0, 1].
    """

    def __init__(self, run_dir: RunDir, space: rc.SearchSpace, hv_ref_point):
        super().__init__()
        self.space = space
        self.hv_ref_point = np.asarray(hv_ref_point, dtype=float)
        self._start = time.perf_counter()
        self.attach(run_dir)

    def attach(self, run_dir: RunDir) -> None:
        """(Re)attach the run directory (not pickled), e.g. after loading a checkpoint."""
        self.run_dir = run_dir

    def __getstate__(self):
        state = self.__dict__.copy()
        state["run_dir"] = None  # holds a lock; reattached on resume
        return state

    @property
    def hv_box_volume(self) -> float:
        return float(np.prod(self.hv_ref_point - np.asarray(IDEAL_POINT)))

    def hypervolume(self, F: np.ndarray, feasible: np.ndarray) -> float:
        F = F[feasible & np.all(F < self.hv_ref_point, axis=1)]
        return float(HV(ref_point=self.hv_ref_point)(F)) if len(F) else 0.0

    def notify(self, algorithm):
        generation = algorithm.n_gen
        pop = algorithm.pop
        X, F, CV = pop.get("X"), pop.get("F"), pop.get("CV").reshape(-1)
        feasible = CV <= 0
        rank = pop.get("rank")
        crowding = pop.get("crowding")
        front = np.zeros(len(pop), dtype=bool)
        if feasible.any():
            idx = np.flatnonzero(feasible)
            front[idx[NonDominatedSorting().do(F[idx], only_non_dominated_front=True)]] = True

        out = os.path.join(self.run_dir.generation_dir(generation), "population.csv")
        with open(out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["rig_id", "genome", "F1", "F2", "F3", "CV", "feasible", "pareto", "rank", "crowding"])
            for i in range(len(pop)):
                genome = rc.Genome.from_x(X[i], self.space)
                writer.writerow([
                    rc.rig_id(genome, self.space), " ".join(map(str, genome.to_x())), *F[i], CV[i], bool(feasible[i]),
                    bool(front[i]), rank[i], crowding[i],
                ])  # fmt: skip

        hv = self.hypervolume(F, feasible)
        best = int(np.flatnonzero(feasible)[np.argmin(F[feasible, 0])]) if feasible.any() else None
        best_rig = rc.rig_id(rc.Genome.from_x(X[best], self.space), self.space) if best is not None else ""
        progress = {
            "generation": generation,
            "n_evaluated": len(self.run_dir),
            "n_feasible": int(feasible.sum()),
            "n_pareto": int(front.sum()),
            "hv_normalized": hv / self.hv_box_volume,
            "hypervolume": hv,
            "best_coverage": float(-F[best, 0]) if best is not None else 0.0,
            "best_rig": best_rig,
            "wall_s": time.perf_counter() - self._start,
        }
        path = os.path.join(self.run_dir.root, "progress.csv")
        new_file = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=PROGRESS_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(progress)
        best_text = (
            f"best coverage {progress['best_coverage']:.3f}: {best_rig} ({self.run_dir.eval_dir(best_rig)})"
            if best_rig
            else "no feasible rig yet"
        )
        print(
            f"[NSGA]: gen {generation}: {progress['n_evaluated']} rigs evaluated, {progress['n_feasible']} feasible, "
            f"{progress['n_pareto']} on the front, HV {progress['hv_normalized']:.3f} (fraction of the objective box) | "
            f"{best_text}"
        )

        algorithm.problem.generation = generation + 1  # rigs created from now on belong to the next generation
        tmp = self.run_dir.checkpoint + ".tmp"
        with open(tmp, "wb") as f:
            dill.dump(algorithm, f)
        os.replace(tmp, self.run_dir.checkpoint)  # atomic: an interrupted write never corrupts the checkpoint


def load_checkpoint(run_dir: RunDir):
    with open(run_dir.checkpoint, "rb") as f:
        algorithm = dill.load(f)
    # The checkpoint is written inside the callback, which pymoo calls just *before* it increments n_iter; finish that
    # increment so the resumed run continues with the next generation instead of repeating the saved one.
    algorithm.n_iter += 1
    return algorithm
