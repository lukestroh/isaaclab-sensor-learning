#!/usr/bin/env python3
"""
On-disk layout of one NSGA-II run, and its evaluation cache (``index.csv``).

    data/runs/<run_id>/
        run.yaml                    run settings (seed, git hash, search space, catalog snapshot, configs)
        index.csv                   one row per evaluated rig (the cache: a finished rig is never simulated twice;
                                    failed rigs are retried in a later process, since the cause may be fixed)
        checkpoint.pkl              latest pymoo algorithm state (written by CheckpointCallback)
        generations/gen_XXXX/       population.csv per generation
        rigs/<rig_id>.yaml          decoded rig, read by evaluate_rig.py
        evals/<rig_id>/             capture.h5, nbv.h5, metrics.json, sim.log
"""
import csv
import os
import threading
from datetime import datetime

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNS_DIR = os.path.join(PROJECT_DIR, "data", "runs")

INDEX_FIELDS = [
    "rig_id", "status", "generation", "F1", "F2", "F3", "G_spacing", "G_failed",
    "n_sensors", "radius", "colatitude", "models", "cost",
    "coverage_mean", "coverage_std", "time_objective_mean", "n_envs_reached_reference", "time_to_reference_mean",
    "path_length_mean", "path_to_reference_mean", "n_envs", "n_envs_failed", "wall_s", "error",
]  # fmt: skip


def new_run_id(tag: str | None = None) -> str:
    run_id = f"nsga_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return f"{run_id}_{tag}" if tag else run_id


class RunDir:
    def __init__(self, run_id: str, runs_dir: str = RUNS_DIR):
        self.run_id = run_id
        self.root = os.path.join(runs_dir, run_id)
        for sub in ("rigs", "evals", "generations"):
            os.makedirs(os.path.join(self.root, sub), exist_ok=True)
        self.index_path = os.path.join(self.root, "index.csv")
        self._lock = threading.Lock()
        self._rows: dict[str, dict] = {}
        if os.path.exists(self.index_path):
            with open(self.index_path, newline="") as f:
                for row in csv.DictReader(f):
                    if row["status"] != "failed":  # retry failures after a restart (OOM, or a bug since fixed)
                        self._rows[row["rig_id"]] = row

    # paths
    @property
    def run_yaml(self) -> str:
        return os.path.join(self.root, "run.yaml")

    @property
    def checkpoint(self) -> str:
        return os.path.join(self.root, "checkpoint.pkl")

    def rig_yaml(self, rig_id: str) -> str:
        return os.path.join(self.root, "rigs", f"{rig_id}.yaml")

    def eval_dir(self, rig_id: str) -> str:
        return os.path.join(self.root, "evals", rig_id)

    def generation_dir(self, generation: int) -> str:
        path = os.path.join(self.root, "generations", f"gen_{generation:04d}")
        os.makedirs(path, exist_ok=True)
        return path

    # evaluation cache
    def cached(self, rig_id: str) -> dict | None:
        with self._lock:
            return self._rows.get(rig_id)

    def record(self, row: dict) -> None:
        """Append one evaluated rig to index.csv (thread-safe)."""
        row = {k: row.get(k) for k in INDEX_FIELDS}
        with self._lock:
            new_file = not os.path.exists(self.index_path)
            with open(self.index_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
            self._rows[row["rig_id"]] = {k: ("" if v is None else str(v)) for k, v in row.items()}

    def __len__(self) -> int:
        return len(self._rows)
