#!/usr/bin/env python3
"""
NSGA-II objectives from one rig evaluation's ``metrics.json`` (written by scripts/evaluate_rig.py). All minimized.

F1 = -mean over envs of total coverage (fraction of all instances with >= 5 hits; class-agnostic).
F2 =  mean over envs of the time to first reach the reference coverage. An env that never reaches it gets the time
      budget plus a shortfall penalty, T * (1 + (c_ref - c_final) / c_ref): always worse than any env that reached
      it (range [T, 2T]), and worse the further short it fell. The budget (not the env's stop time) is used, so an
      env stopped early by a plateau is not rewarded for stopping.
F3 =  total sensor cost (USD), from the rig design (no simulation needed).
"""
import numpy as np

N_OBJ = 3


def time_objective(env: dict, time_budget_s: float, reference_coverage: float) -> float:
    """F2 contribution of one env (see module docstring)."""
    if env["reached_reference"]:
        return float(env["time_to_reference_s"])
    shortfall = (reference_coverage - min(max(env["coverage_total"], 0.0), reference_coverage)) / reference_coverage
    return time_budget_s * (1.0 + shortfall)


def objectives_from_metrics(metrics: dict, cost: float) -> dict:
    """
    :return: ``{"F": [F1, F2, F3], **summary}`` for a successful evaluation; raises ValueError otherwise.
    """
    if metrics.get("status") != "ok":
        raise ValueError(f"Evaluation did not succeed: {metrics.get('error', metrics.get('status'))}")
    envs = metrics["envs"]
    budget = float(metrics["config"]["termination"]["max_sim_time_s"])
    c_ref = float(metrics["config"]["reference_coverage"])
    coverage = np.array([e["coverage_total"] for e in envs])
    times = np.array([time_objective(e, budget, c_ref) for e in envs])
    reached = np.array([e["reached_reference"] for e in envs])
    return {
        "F": [-float(coverage.mean()), float(times.mean()), float(cost)],
        "coverage_mean": float(coverage.mean()),
        "coverage_std": float(coverage.std()),
        "time_objective_mean": float(times.mean()),
        "n_envs_reached_reference": int(reached.sum()),
        "time_to_reference_mean": float(np.mean([e["time_to_reference_s"] for e in envs if e["reached_reference"]]))
        if reached.any()
        else None,
        "path_length_mean": float(np.mean([e["path_length_m"] for e in envs])),
        "path_to_reference_mean": float(np.mean([e["path_to_reference_m"] for e in envs if e["reached_reference"]]))
        if reached.any()
        else None,
        "n_envs": len(envs),
        "n_envs_failed": int(sum(e["reason_name"] == "UNREACHABLE_GOALS" for e in envs)),
    }


def placeholder_objectives(cost: float, time_budget_s: float) -> list[float]:
    """F for rigs that were not simulated (infeasible) or whose simulation failed: no coverage, worst time."""
    return [0.0, 2.0 * time_budget_s, float(cost)]
