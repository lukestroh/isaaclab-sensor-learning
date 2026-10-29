#!/usr/bin/env python3
"""
NSGA-II search over sensor rigs (mount-platform radius / colatitude, sensor count, model per Lloyd site).

This process never imports Isaac Sim: each rig is evaluated by its own `evaluate_rig.py` subprocess, pinned to a GPU
slot. Results, checkpoints and rigs go to data/runs/<run_id>/ (see moo/run_dir.py). Interrupt at any time and resume
with --resume <run_id>; finished rigs are never re-simulated.

    ../../venv/bin/python scripts/run_nsga2.py --pop-size 20 --n-gen 30 --num-envs 20
    ../../venv/bin/python scripts/run_nsga2.py --dry-run --pop-size 8 --n-gen 3     # no simulation (pipeline check)
    ../../venv/bin/python scripts/run_nsga2.py --resume nsga_20261028_140000 --n-gen 40
"""
import argparse
import csv
import os
import platform
import subprocess
import sys
from datetime import datetime
from multiprocessing.pool import ThreadPool

import numpy as np
import pymoo
import yaml
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.parallelization import StarmapParallelization
from pymoo.termination import get_termination

from moo import rig_codec as rc
from moo.callbacks import CheckpointCallback, load_checkpoint
from moo.evaluators import SubprocessEvaluator, SyntheticEvaluator
from moo.progress import GenerationProgress
from moo.operators import RigCrossover, RigDuplicateElimination, RigMutation, RigRepair, RigSampling
from moo.run_dir import RUNS_DIR, RunDir, new_run_id
from moo.sensor_cfg_problem import SensorRigProblem

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)
# Settings that define a run; on --resume they come from run.yaml, not the command line.
RUN_SETTINGS = ("search_space", "pop_size", "seed", "seed_rigs", "time_budget_s", "reference_coverage", "num_envs",
                "max_capture_poses", "dry_run")  # fmt: skip


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--search-space", default=os.path.join(SCRIPTS_DIR, "moo", "config", "spherical_search_space.yaml")
    )
    parser.add_argument("--pop-size", type=int, default=20)
    parser.add_argument("--n-gen", type=int, default=30, help="Total generations (on --resume: run until this many).")
    parser.add_argument("--seed", type=int, default=0, help="GA seed (the simulation's tree seed is evaluate_rig's).")
    parser.add_argument(
        "--seed-rigs", nargs="*", default=[os.path.join(PROJECT_DIR, "data", "rigs", "test_rig0.yaml")],
        help="Rig yamls placed in the initial population (snapped to the search space). Pass nothing to disable.",
    )  # fmt: skip
    parser.add_argument("--time-budget-s", type=float, default=180.0, help="NBV sim-time budget per rig.")
    parser.add_argument("--reference-coverage", type=float, default=0.4, help="Coverage for objective 2 (time).")
    parser.add_argument("--num-envs", type=int, default=None, help="Trees per evaluation (default: task config).")
    parser.add_argument("--max-capture-poses", type=int, default=None, help="Debug: passed to evaluate_rig.py.")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids for simulations, e.g. '0,1'.")
    parser.add_argument("--slots-per-gpu", type=int, default=1, help="Concurrent simulations per GPU.")
    parser.add_argument("--timeout-s", type=float, default=None, help="Wall-clock limit per rig (failed if exceeded).")
    parser.add_argument(
        "--max-consecutive-failures", type=int, default=5,
        help="Stop the run after this many simulations fail in a row (0 = never); fix the cause, then --resume.",
    )  # fmt: skip
    parser.add_argument("--tag", default=None, help="Suffix for the run id.")
    parser.add_argument("--runs-dir", default=RUNS_DIR, help="Parent directory of run directories (default: data/runs).")
    parser.add_argument("--resume", default=None, metavar="RUN_ID", help="Continue a run from its checkpoint.")
    parser.add_argument("--dry-run", action="store_true", help="Synthetic results instead of simulating (pipeline test).")
    parser.add_argument("--no-progress", action="store_true", help="Disable the per-generation progress bar.")
    return parser.parse_args()


def git_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_run_yaml(run_dir: RunDir, settings: dict, space: rc.SearchSpace, catalog: rc.SensorCatalog) -> None:
    with open(settings["search_space"]) as f:
        space_cfg = yaml.safe_load(f)
    catalog_snapshot = {}
    for model in space.allowed_models:
        sensor_type = catalog.sensor_type(model)
        with open(os.path.join(catalog.sensor_cfg_dir, sensor_type, f"{model}.yaml")) as f:
            catalog_snapshot[model] = yaml.safe_load(f)
    run = {
        "run_id": run_dir.run_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_hash": git_hash(),
        "python": platform.python_version(),
        "pymoo": pymoo.__version__,
        "settings": settings,
        "search_space": space_cfg,
        "search_space_digest": space.digest,
        "catalog": catalog_snapshot,
        "objectives": ["-coverage (instances >= 5 hits, mean over envs)",
                       "time to reference coverage (s; penalized if not reached)", "sensor cost (USD)"],  # fmt: skip
        "resumed": [],
    }
    with open(run_dir.run_yaml, "w") as f:
        yaml.safe_dump(run, f, sort_keys=False)


def write_pareto(run_dir: RunDir, res, space: rc.SearchSpace) -> str:
    path = os.path.join(run_dir.root, "pareto.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rig_id", "coverage", "time_to_reference_obj_s", "cost_usd", "rig_yaml", "eval_dir"])
        if res.opt is not None:
            for ind in res.opt:
                if ind.CV[0] > 0:
                    continue
                rid = rc.rig_id(rc.Genome.from_x(ind.X, space), space)
                writer.writerow([rid, -ind.F[0], ind.F[1], ind.F[2], run_dir.rig_yaml(rid), run_dir.eval_dir(rid)])
    return path


def main():
    args = parse_args()
    settings = {k: getattr(args, k) for k in RUN_SETTINGS}
    if args.resume:
        run_dir = RunDir(args.resume, runs_dir=args.runs_dir)
        with open(run_dir.run_yaml) as f:
            run_yaml = yaml.safe_load(f)
        settings = run_yaml["settings"]
        run_yaml["resumed"].append({"at": datetime.now().isoformat(timespec="seconds"), "n_gen": args.n_gen})
        with open(run_dir.run_yaml, "w") as f:
            yaml.safe_dump(run_yaml, f, sort_keys=False)
    else:
        run_dir = RunDir(new_run_id(args.tag), runs_dir=args.runs_dir)

    space = rc.SearchSpace.from_yaml(settings["search_space"])
    catalog = rc.SensorCatalog()
    if not args.resume:
        write_run_yaml(run_dir, settings, space, catalog)
    print(f"[NSGA]: run {run_dir.run_id} -> {run_dir.root}")

    budget, c_ref = settings["time_budget_s"], settings["reference_coverage"]
    slots = [int(g) for g in args.gpus.split(",") for _ in range(args.slots_per_gpu)]
    if settings["dry_run"]:
        evaluator = SyntheticEvaluator(budget, c_ref, seed=settings["seed"])
    else:
        extra = []
        if settings["num_envs"] is not None:
            extra += ["--num-envs", str(settings["num_envs"])]
        if settings["max_capture_poses"] is not None:
            extra += ["--max-capture-poses", str(settings["max_capture_poses"])]
        evaluator = SubprocessEvaluator(slots, budget, c_ref, extra_args=extra, timeout_s=args.timeout_s)
    n_parallel = 1 if settings["dry_run"] else len(slots)
    pool = ThreadPool(n_parallel) if n_parallel > 1 else None
    runner = StarmapParallelization(pool.starmap) if pool else None
    progress = None if args.no_progress else GenerationProgress()

    try:
        if args.resume:
            algorithm = load_checkpoint(run_dir)
            algorithm.problem.attach(run_dir, evaluator, runner, progress)
            algorithm.problem.max_consecutive_failures = args.max_consecutive_failures
            algorithm.problem.consecutive_failures = 0
            algorithm.callback.attach(run_dir)
            algorithm.termination = get_termination("n_gen", args.n_gen)
            print(f"[NSGA]: resuming after generation {algorithm.n_gen}, running to {args.n_gen}")
            res = algorithm.run()
        else:
            problem = SensorRigProblem(
                space, catalog, budget, run_dir, evaluator, runner,
                max_consecutive_failures=args.max_consecutive_failures, progress=progress,
            )  # fmt: skip
            seeds = [rc.genome_from_rig_yaml(p, space).to_x() for p in settings["seed_rigs"]]
            max_cost = space.max_total * max(catalog.cost(m) for m in space.allowed_models)
            algorithm = NSGA2(
                pop_size=settings["pop_size"],
                sampling=RigSampling(space, seeds=seeds),
                crossover=RigCrossover(space),
                mutation=RigMutation(space),
                repair=RigRepair(space),
                eliminate_duplicates=RigDuplicateElimination(space),
            )
            algorithm.setup(
                problem,
                termination=get_termination("n_gen", args.n_gen),
                seed=settings["seed"],
                callback=CheckpointCallback(run_dir, space, hv_ref_point=[0.0, 2.0 * budget, max_cost]),
                verbose=True,
            )
            res = algorithm.run()
    finally:
        if pool:
            pool.close()

    print(f"[NSGA]: done. Pareto front: {write_pareto(run_dir, res, space)}")


if __name__ == "__main__":
    sys.exit(main())
