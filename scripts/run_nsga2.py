#!/usr/bin/env python3
import argparse
import numpy as np
import yaml

from multiprocessing.pool import ThreadPool
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.parallelization.starmap import StarmapParallelization

from moo.sensor_cfg_problem import (
    SensorCfgProblem,
    ConfigCrossover,
    ConfigDuplicateElimination,
    ConfigMutation,
    ConfigSampling
)

# parallel processing: https://pymoo.org/parallelization/starmap.html

# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def run_nsga2(sensor_dir, search_space_path, gpu_ids, pop_size=None, n_gen=30, seed=1):
    """One simulation per GPU at a time. pop_size defaults to len(gpu_ids): one wave per generation."""
    pop_size = pop_size or len(gpu_ids)
 
    pool = ThreadPool(len(gpu_ids))  # threads only wait on subprocesses, so no pickling needed
    try:
        problem = SensorCfgProblem(
            sensor_dir,
            search_space_path,
            gpu_ids,
            elementwise_runner=StarmapParallelization(pool.starmap),
        )
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=ConfigSampling(),
            crossover=ConfigCrossover(),
            mutation=ConfigMutation(),
            eliminate_duplicates=ConfigDuplicateElimination(),
        )
        return minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=True)
    finally:
        pool.close()


def save_results(res, path: str) -> None:
    F = np.atleast_2d(res.F)
    X = res.X.reshape(len(F), -1)
    rows = [
        {
            "config": X[i, 0],
            "mean_coverage_pct": float(-F[i, 0]),
            "mean_path_length_m": float(F[i, 1]),
            "cost": float(F[i, 2]),
        }
        for i in range(len(F))
    ]
    with open(path, "w") as f:
        yaml.safe_dump(rows, f)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-gpus", type=int, required=True)
    parser.add_argument("--n-gen", type=int, default=30)
    args = parser.parse_args()

    res = run_nsga2("sensors/", "sensor_cfg_search_space.yaml", gpu_ids=list(range(args.n_gpus)), n_gen=args.n_gen)
    save_results(res, "pareto_front.yaml")
