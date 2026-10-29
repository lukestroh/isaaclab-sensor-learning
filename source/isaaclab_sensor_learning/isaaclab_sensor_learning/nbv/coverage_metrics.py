#!/usr/bin/env python3
"""
Coverage metrics on bin counts of shape (n_instances, n_length_bins, n_theta_bins).

Depends only on torch and the standard library, so code that runs outside Isaac Sim (plotting, the optimizer) can
load this file directly without importing the ``isaaclab_sensor_learning`` package, whose ``__init__`` needs Kit:

    spec = importlib.util.spec_from_file_location("coverage_metrics", "<path>/nbv/coverage_metrics.py")

NBVEvaluator uses these same functions, so the simulation and offline analysis always agree.
"""
from dataclasses import dataclass, field

import torch

# Tree part classes, finest last. Keep instance_class() in sync with scripts/moo/instance_parser.parse_instance;
# tests/test_instance_coverage.py checks they agree.
INSTANCE_CLASSES = ("trunk", "branch", "spur")
EXTENT_METRICS = ("length_coverage", "theta_coverage", "continuous")


def instance_class(prim_path: str, classes: tuple[str, ...] = INSTANCE_CLASSES) -> str | None:
    """Class of a tree prim: the class its leaf name starts with (longest match), e.g. ".../spurId_0806" -> "spur"."""
    leaf = str(prim_path).rstrip("/").rsplit("/", 1)[-1].lower()
    matches = [c for c in classes if leaf.startswith(c)]
    return max(matches, key=len) if matches else None


@dataclass
class CoverageConfig:
    # Hit thresholds k, reported side by side. Detection: an instance counts as seen at k once it has >= k hits in
    # total. Extent: a length (theta) bin counts as hit at k once its hits summed around (along) the instance are >= k.
    instance_min_hits: tuple[int, ...] = (1, 5)
    # An instance is "continuous" when at least this fraction of its length bins is hit ...
    continuity_min_length_frac: float | dict[str, float] = 0.5  # "L"
    # ... and no run of consecutive unhit length bins, including runs at either end, is longer than this.
    continuity_max_gap_bins: int | dict[str, int] = field(
        # "g"; each length bin is 1 / n_length_bins (20) of the instance's length. Chosen as a physical break size:
        #   spur   ~10 cm -> 0.5 cm/bin, tolerate a 2 cm break -> 4 bins
        #   branch ~50 cm -> 2.5 cm/bin, tolerate a 5 cm break -> 2 bins
        #   trunk: 2 bins (10% of its length)
        default_factory=lambda: {"trunk": 2, "branch": 2, "spur": 4}
    )
    # L and g are relative to each instance's length. Either one value for every class, or a per-class dict keyed by
    # INSTANCE_CLASSES (missing classes, and instances of unknown class, use the "default" key or 0.5 / 2).


def class_ids_from_prim_map(prim_path_to_custom_id: dict, device: str | torch.device = "cpu") -> torch.Tensor:
    """(n_instances,) index into INSTANCE_CLASSES per custom id, from a {prim_path: custom_id} map; -1 if unknown."""
    n = max((int(v) for v in prim_path_to_custom_id.values()), default=-1) + 1
    class_ids = torch.full((n,), -1, dtype=torch.int64, device=device)
    for prim_path, custom_id in prim_path_to_custom_id.items():
        cls = instance_class(prim_path)
        if cls is not None:
            class_ids[int(custom_id)] = INSTANCE_CLASSES.index(cls)
    return class_ids


def per_class_values(value, class_ids: torch.Tensor, fallback: float) -> torch.Tensor:
    """Expand a scalar or {class: value} threshold into a per-instance tensor (class_ids index INSTANCE_CLASSES)."""
    if not isinstance(value, dict):
        return torch.full(class_ids.shape, float(value), device=class_ids.device)
    default = float(value.get("default", fallback))
    lookup = torch.tensor(
        [float(value.get(cls, default)) for cls in INSTANCE_CLASSES] + [default], device=class_ids.device
    )
    return lookup[torch.where(class_ids >= 0, class_ids, len(INSTANCE_CLASSES))]


def _class_groups(class_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    groups = {cls: class_ids == i for i, cls in enumerate(INSTANCE_CLASSES)}
    groups["total"] = torch.ones_like(class_ids, dtype=torch.bool)
    return groups


def instance_coverage(
    counts: torch.Tensor, class_ids: torch.Tensor, min_hits: tuple[int, ...]
) -> dict[str, dict]:
    """
    Instance detection: the fraction of instances with at least k hits (summed over all their bins).

    :return: ``{cls: {"n_instances": int, "seen": {k: fraction}}}`` for each class in INSTANCE_CLASSES plus "total"
        (all instances, including any of unknown class).
    """
    hits = counts.reshape(counts.shape[0], -1).sum(dim=1)
    coverage = {}
    for cls, mask in _class_groups(class_ids).items():
        n = int(mask.sum())
        coverage[cls] = {
            "n_instances": n,
            "seen": {k: (float((hits[mask] >= k).sum()) / n if n else float("nan")) for k in min_hits},
        }
    return coverage


def instance_extent_per_instance(
    counts: torch.Tensor, class_ids: torch.Tensor, min_hits: int, config: CoverageConfig
) -> dict[str, torch.Tensor]:
    """
    Per-instance extent at hit threshold ``min_hits``.

    :return: (n_instances,) tensors:
        - "length_coverage": fraction of length bins hit (a length bin's hits are summed over theta).
        - "theta_coverage": fraction of theta bins hit (a theta bin's hits are summed over length).
        - "longest_gap": longest run of consecutive unhit length bins, counting runs at either end.
          An unseen instance has longest_gap = n_length_bins.
        - "continuous": length_coverage >= L and longest_gap <= g, with L and g looked up per instance class
          (``CoverageConfig.continuity_min_length_frac`` / ``continuity_max_gap_bins``).
    """
    length_hit = counts.sum(dim=2) >= min_hits  # (n_instances, n_length_bins)
    theta_hit = counts.sum(dim=1) >= min_hits  # (n_instances, n_theta_bins)

    run = torch.zeros(counts.shape[0], dtype=torch.int64, device=counts.device)
    longest_gap = torch.zeros_like(run)
    for j in range(length_hit.shape[1]):
        run = torch.where(length_hit[:, j], torch.zeros_like(run), run + 1)
        longest_gap = torch.maximum(longest_gap, run)

    length_coverage = length_hit.float().mean(dim=1)
    min_length_frac = per_class_values(config.continuity_min_length_frac, class_ids, fallback=0.5)
    max_gap_bins = per_class_values(config.continuity_max_gap_bins, class_ids, fallback=2)
    return {
        "length_coverage": length_coverage,
        "theta_coverage": theta_hit.float().mean(dim=1),
        "longest_gap": longest_gap,
        "continuous": (length_coverage >= min_length_frac) & (longest_gap <= max_gap_bins),
    }


def instance_extent(
    counts: torch.Tensor, class_ids: torch.Tensor, min_hits: tuple[int, ...], config: CoverageConfig
) -> dict[str, dict]:
    """
    Per-class extent, averaged over *all* instances of the class (unseen instances contribute 0).

    :return: ``{cls: {k: {"length_coverage", "theta_coverage", "continuous"}}}`` for each class in INSTANCE_CLASSES
        plus "total". "continuous" is the fraction of instances that are continuous.
    """
    per_k = {k: instance_extent_per_instance(counts, class_ids, k, config) for k in min_hits}
    extent = {}
    for cls, mask in _class_groups(class_ids).items():
        extent[cls] = {
            k: {
                name: (float(values[name][mask].float().mean()) if bool(mask.any()) else float("nan"))
                for name in EXTENT_METRICS
            }
            for k, values in per_k.items()
        }
    return extent
