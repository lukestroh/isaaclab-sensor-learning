#!/usr/bin/env python3
"""
Rig evaluators: given a rig yaml and an output directory, produce that rig's ``metrics.json`` contents.

SubprocessEvaluator launches scripts/evaluate_rig.py, one Isaac Sim process per rig, pinned to a free GPU slot.
Several slots (one per GPU, or several per GPU) let a thread-pool runner evaluate rigs in parallel.
"""
import json
import os
import queue
import subprocess
import sys
import time

SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVALUATE_RIG = os.path.join(SCRIPTS_DIR, "evaluate_rig.py")


def read_metrics(out_dir: str) -> dict:
    path = os.path.join(out_dir, "metrics.json")
    if not os.path.exists(path):
        return {"status": "failed", "error": f"no metrics.json in {out_dir}"}
    with open(path) as f:
        return json.load(f)


class SubprocessEvaluator:
    """
    :param gpu_slots: GPU ids, one entry per concurrent simulation (e.g. [0] or [0, 1] or [0, 0]).
    :param time_budget_s: NBV sim-time budget passed to evaluate_rig.py.
    :param reference_coverage: reference coverage passed to evaluate_rig.py.
    :param extra_args: further evaluate_rig.py flags (e.g. ["--num-envs", "20"]).
    :param timeout_s: wall-clock limit per rig; a timed-out rig counts as failed.
    """

    def __init__(
        self,
        gpu_slots: list[int],
        time_budget_s: float,
        reference_coverage: float,
        extra_args: list[str] | None = None,
        timeout_s: float | None = None,
        python: str = sys.executable,
        script: str = EVALUATE_RIG,
    ):
        self.time_budget_s = time_budget_s
        self.reference_coverage = reference_coverage
        self.extra_args = list(extra_args or [])
        self.timeout_s = timeout_s
        self.python = python
        self.script = script
        self._slots = queue.Queue()
        for gpu in gpu_slots:
            self._slots.put(gpu)
        self.n_slots = len(gpu_slots)

    def command(self, rig_yaml_path: str, out_dir: str) -> list[str]:
        return [
            self.python, self.script,
            "--rig-yaml-path", rig_yaml_path,
            "--out-dir", out_dir,
            "--coverage-target", "none",
            "--reference-coverage", str(self.reference_coverage),
            "--max-nbv-time-s", str(self.time_budget_s),
            "--quiet",
            *self.extra_args,
        ]  # fmt: skip

    def __call__(self, rig_yaml_path: str, out_dir: str) -> dict:
        os.makedirs(out_dir, exist_ok=True)
        gpu = self._slots.get()  # blocks until a slot is free
        start = time.perf_counter()
        try:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
            with open(os.path.join(out_dir, "sim.log"), "w") as log:
                proc = subprocess.run(
                    self.command(rig_yaml_path, out_dir),
                    cwd=SCRIPTS_DIR, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=self.timeout_s,
                )  # fmt: skip
            metrics = read_metrics(out_dir)
            if metrics.get("status") != "ok" and proc.returncode != 0:
                metrics.setdefault("error", f"evaluate_rig.py exited with {proc.returncode}")
        except subprocess.TimeoutExpired:
            metrics = {"status": "failed", "error": f"timed out after {self.timeout_s} s"}
        finally:
            self._slots.put(gpu)
        metrics["wall_s"] = time.perf_counter() - start
        return metrics


class SyntheticEvaluator:
    """
    Stand-in for SubprocessEvaluator (``run_nsga2.py --dry-run``): returns metrics.json-shaped results from a simple
    formula of the rig (more sensors and a larger, wider platform give more coverage, sooner), without simulating.
    It exercises the whole NSGA pipeline (decode, cache, objectives, checkpoints) in seconds.
    """

    def __init__(self, time_budget_s: float, reference_coverage: float, n_envs: int = 2, seed: int = 0):
        import numpy as np

        self.time_budget_s = time_budget_s
        self.reference_coverage = reference_coverage
        self.n_envs = n_envs
        self._rng = np.random.default_rng(seed)
        self.n_slots = 1

    def __call__(self, rig_yaml_path: str, out_dir: str) -> dict:
        import numpy as np
        import yaml

        with open(rig_yaml_path) as f:
            rig = yaml.safe_load(f)
        n, r, c = len(rig["sensors"]), rig["layout"]["radius"], rig["layout"]["colatitude"]
        envs = []
        for _ in range(self.n_envs):
            coverage = float(np.clip(0.04 * n + 0.6 * r + 0.002 * c + self._rng.normal(0, 0.02), 0.0, 0.95))
            reached = coverage >= self.reference_coverage
            t_ref = self.time_budget_s * self.reference_coverage / max(coverage, 1e-6) if reached else None
            envs.append({"reason_name": "TIME_BUDGET", "coverage_total": coverage, "reached_reference": reached,
                         "time_to_reference_s": t_ref, "path_to_reference_m": 0.3 * t_ref if reached else None,
                         "path_length_m": 0.3 * self.time_budget_s})  # fmt: skip
        metrics = {"status": "ok", "envs": envs, "wall_s": 0.0,
                   "config": {"reference_coverage": self.reference_coverage,
                              "termination": {"max_sim_time_s": self.time_budget_s}}}  # fmt: skip
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics
