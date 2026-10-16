#!/usr/bin/env python3
import numpy as np
import os
import pickle
import subprocess
import sys
import tempfile
import queue
import yaml

from . import coverage_evaluation
from . import path_evaluation
from . import instance_parser as iparser

from pathlib import Path
from pymoo.core.crossover import Crossover
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling

from sim_utils.sim_worker import SimOutputs


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_gpu_env(gpu_id: int) -> dict:
    """Environment for a worker pinned to one GPU"""
    return{**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}


def simulate_on_gpu(sensor_cfg: dict, gpu_id: int) -> SimOutputs:
    """Run sim_worker.py in fresh subprocess pinned to 'gpu_id'."""
    # WORKER_SCRIPT = Path(__file__).with_name("sim_worker.py")
    WORKER_SCRIPT = Path(__file__).parents[1] / "sim_utils" / "sim_worker.py"
    with tempfile.TemporaryDirectory() as tmp:
        sensor_cfg_path, out_path = Path(tmp) / "sensor_cfg.yaml", Path(tmp) / "output.pkl"
        with open(sensor_cfg_path, 'w') as file:
            yaml.safe_dump(sensor_cfg, file)

        subprocess.run(
            [sys.executable, str(WORKER_SCRIPT), "--sensor-cfg", str(sensor_cfg_path), "--out", str(out_path)],
            env=get_gpu_env(gpu_id),
            check=True
        )

        with open(out_path, 'wb') as file:
            return SimOutputs(**pickle.load(file))


class ConfigSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        # Must return an object array of shape (n_samples, 1), each entry a config dict.
        raise NotImplementedError
 
 
class ConfigCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2)  # 2 parents -> 2 offspring
 
    def _do(self, problem, X, **kwargs):
        # X: (n_parents=2, n_matings, n_var=1). Return the same shape.
        raise NotImplementedError
 
 
class ConfigMutation(Mutation):
    def _do(self, problem, X, **kwargs):
        # X: (n_individuals, n_var=1). Return the mutated array, same shape.
        raise NotImplementedError
 
 
class ConfigDuplicateElimination(ElementwiseDuplicateElimination):
    def is_equal(self, a, b):
        return a.X[0] == b.X[0]  # dict equality


class SensorCfgProblem(ElementwiseProblem):
    """One decision variable: a whole sensor configuration (a dict), stored in X[i, 0]."""
    def __init__(
        self,
        sensor_dir: str,
        sensor_cfg_search_space: str,
        gpu_ids: list[int],
        semantic_labels = ["trunk", "branch", "spur"],
        **kwargs
    ) -> None:
        super().__init__(
            n_var=1, # variable storing entire cfg dict, stored in X[i, 0]
            n_obj=3, # objectives to optimize
            **kwargs
        )

        self.semantic_labels = semantic_labels

        self.free_gpus = queue.Queue()

        for gid in gpu_ids:
            self.free_gpus.put(gid)

        # ASSUMPTIONS: one yaml per sensor, file stem == sensor name. All sensors have a "cost" key
        self.sensor_costs = {p.stem: float(load_yaml(p)["cost"]) for p in Path(sensor_dir).glob("*.yaml")}

        # Sensor cfg search-space definition; used by the operators
        self.search_space = load_yaml(sensor_cfg_search_space)
        return

    def sensor_cfg_cost(self, cfg: dict) -> float:
        """Assumes cfg["sensors"] is a list of sensor names"""
        return sum(self.sensor_costs[sensor_name] for sensor_name in cfg["sensors"].keys())

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        sensor_cfg = x[0]
        gpu_id = self.free_gpus.get() # blocks if all gpus are busy

        try:
            sim = simulate_on_gpu(sensor_cfg=sensor_cfg, gpu_id=gpu_id)
        finally:
            self.free_gpus.put(gpu_id)

        coverage = coverage_evaluation.coverage_metrics(
            bin_counts=sim.bin_counts,
            instance_map=sim.instance_map,
            semantic_labels=self.semantic_labels,
            parse_instance=iparser.parse_instance,
        )
        mean_coverage = np.mean([v["pct_bins_seen"] for env, v in coverage.items() if env != "all"]) # coverage by tree (pooled ratio)
        mean_movement = path_evaluation.eef_motion_metrics(poses=sim.poses, t_final=sim.t_final)["path_length"].mean()

        out["F"] = np.array([-mean_coverage, mean_movement, self.sensor_cfg_cost(sensor_cfg)], dtype=float)

        return


