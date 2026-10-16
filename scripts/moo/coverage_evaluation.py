#!/usr/bin/env python3
import numpy as np
import pprint as pp

from typing import Callable


# def coverage_metrics(bin_counts: dict[str, np.ndarray]) -> dict:
#     # Add all bins final state, not just sparse bin updates
#     coverages = {}
#     for env_key, counts in bin_counts.items():
#         coverages[env_key] = np.where(counts > 5)



#     return{}
#
def _tally(counts: np.ndarray, rows_by_label: dict, min_count: int) -> dict:
    """Raw tallies for one environment. counts: (n_instances, *bin_shape)."""
    bins_per_instance = int(np.prod(counts.shape[1:]))
    flat = counts.reshape(counts.shape[0], -1)
    seen_per_instance = (flat >= min_count).sum(axis=1)
    counts_per_instance = flat.sum(axis=1, dtype=np.int64)

    seen_by_type, total_by_type, best = {}, {}, None
    for label, items in rows_by_label.items():
        ids = np.asarray([i for i, _ in items], dtype=int)
        seen_by_type[label] = int(seen_per_instance[ids].sum())
        total_by_type[label] = len(ids) * bins_per_instance
        for i, path in items:
            if best is None or counts_per_instance[i] > best["counts"]:
                best = {
                    "name": path,
                    "counts": int(counts_per_instance[i]),
                    "seen": int(seen_per_instance[i]),
                    "total": bins_per_instance,
                }
    return {"seen_by_type": seen_by_type, "total_by_type": total_by_type, "best": best}


def _merge(tallies: list[dict]) -> dict:
    """Combine per-environment tallies into one."""
    labels = tallies[0]["seen_by_type"].keys()
    return {
        "seen_by_type": {lbl: sum(t["seen_by_type"][lbl] for t in tallies) for lbl in labels},
        "total_by_type": {lbl: sum(t["total_by_type"][lbl] for t in tallies) for lbl in labels},
        "best": max((t["best"] for t in tallies), key=lambda b: b["counts"]),
    }


def _pct(seen: int, total: int) -> float:
    return 100.0 * seen / total if total > 0 else float("nan")


def _summarize(tally: dict) -> dict:
    seen = sum(tally["seen_by_type"].values())
    total = sum(tally["total_by_type"].values())
    best = tally["best"]
    return {
        "total_bins_seen": seen,
        "pct_bins_seen": _pct(seen, total),
        "bins_seen_by_type": tally["seen_by_type"],
        "pct_bins_seen_by_type": {
            lbl: _pct(tally["seen_by_type"][lbl], tally["total_by_type"][lbl])
            for lbl in tally["seen_by_type"]
        },
        "top_instance": best["name"],
        "top_instance_total_counts": best["counts"],
        "top_instance_bins_seen": best["seen"],
        "top_instance_pct_bins_seen": _pct(best["seen"], best["total"]),
    }



def coverage_metrics(
    bin_counts: dict[str, np.ndarray],
    instance_map: dict[str, int],
    semantic_labels: list[str],
    parse_instance: Callable[[str, list[str]], tuple[str, str]],
    min_count: int = 5,
) -> dict:
    """Bin coverage per environment, plus an aggregate under the key "all".

    Args:
        bin_counts: env name -> (n_instances, *bin_shape) counts. Axis 0 indexes instances.
        instance_map: env name -> {instance label: row index into that environment's
            bin_counts}. Rows not referenced by the map are ignored.
        semantic_labels: all semantic types, e.g. ["trunk", "branch", "spur"].
        parse_instance: (instance_label, semantic_labels) -> semantic_label.
            Env names (the keys of instance_map) must be keys of bin_counts.
        min_count: a bin is "seen" if its count is >= min_count.

    Returns:
        {env_name or "all": {
            total_bins_seen, pct_bins_seen,                 # percentages are 0-100
            bins_seen_by_type, pct_bins_seen_by_type,       # dicts keyed by semantic label
            top_instance,                                    # path of instance with the largest summed counts
            top_instance_total_counts,
            top_instance_bins_seen, top_instance_pct_bins_seen,
        }}
    """
    rows = {}  # env -> label -> [(row_index, instance_label), ...]
    for env, instances in instance_map.items():
        rows[env] = {lbl: [] for lbl in semantic_labels}
        for name, idx in instances.items():
            label = parse_instance(name, semantic_labels)
            if label not in semantic_labels:
                raise ValueError(f"Parser returned '{label}', not in semantic_labels: {name}")
            rows[env][label].append((idx, name))

    tallies = {
        env: _tally(np.asarray(bin_counts[env]), rows_by_label, min_count)
        for env, rows_by_label in rows.items()
    }
    results = {env: _summarize(t) for env, t in tallies.items()}
    results["all"] = _summarize(_merge(list(tallies.values())))
    return results


