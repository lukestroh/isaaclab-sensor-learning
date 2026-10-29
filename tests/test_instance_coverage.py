"""Instance-level coverage: fraction of tree instances per class with at least k hits."""

import importlib.util
import os

import pytest
import torch
from conftest import CUSTOM_IDS, Pixel, make_evaluator, write_capture

from isaaclab_sensor_learning.nbv import nbv

_PARSER_PATH = os.path.join(os.path.dirname(__file__), "..", "scripts", "moo", "instance_parser.py")
_spec = importlib.util.spec_from_file_location("instance_parser", _PARSER_PATH)
instance_parser = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(instance_parser)

PATHS = [
    "/World/envs/env_0/tree/trunkId_0",
    "/World/envs/env_0/tree/trunkId_0/branchL_0_Id_007",
    "/World/envs/env_12/tree/trunkId_0/spurId_0806",
    "/World/envs/env_1/tree/trunkId_0/branchL_3_Id_120/spurId_0001",
]


@pytest.mark.parametrize("path", PATHS)
def test_classifier_matches_optimizer_parser(path):
    """The sim and the optimizer (scripts/moo) must assign every prim to the same class."""
    assert nbv.instance_class(path) == instance_parser.parse_instance(path, list(nbv.INSTANCE_CLASSES))


def test_unknown_leaf_has_no_class():
    assert nbv.instance_class("/World/envs/env_0/tree/Looks") is None


def test_instance_coverage_per_class_and_total(tmp_path, uv_rgb):
    path = write_capture(str(tmp_path / "c.h5"), uv_rgb, [[[Pixel("trunk")]]])
    with make_evaluator(path, n_envs=1) as ev:
        counts = ev.binned_tree.bin_counts["env_0"]
        counts[CUSTOM_IDS["trunk"], 0, 0] = 4  # hits are summed over an instance's bins: 4 + 2 = 6
        counts[CUSTOM_IDS["trunk"], 3, 7] = 2
        counts[CUSTOM_IDS["branch"], 1, 1] = 2  # spur: no hits
        cov = ev.instance_coverage(min_hits=(1, 5))["env_0"]
    assert cov["trunk"] == {"n_instances": 1, "seen": {1: 1.0, 5: 1.0}}
    assert cov["branch"] == {"n_instances": 1, "seen": {1: 1.0, 5: 0.0}}
    assert cov["spur"] == {"n_instances": 1, "seen": {1: 0.0, 5: 0.0}}
    assert cov["total"]["seen"] == pytest.approx({1: 2 / 3, 5: 1 / 3})


def test_default_thresholds_come_from_coverage_config(tmp_path, uv_rgb):
    path = write_capture(str(tmp_path / "c.h5"), uv_rgb, [[[Pixel("trunk")]]])
    ev = nbv.NBVEvaluator(
        trial_data_path=path,
        observation_config=nbv.ObservationConfig(),
        gain_config=nbv.GainConfig(),
        coverage_config=nbv.CoverageConfig(instance_min_hits=(3,)),
        device="cpu",
        n_envs=1,
    )
    with ev:
        assert set(ev.instance_coverage()["env_0"]["total"]["seen"]) == {3}
        assert torch.equal(ev.instance_class_ids["env_0"], torch.tensor([0, 1, 2]))


# --------------------------------------------------------------------------------------------------------------------
# Extent: length / theta coverage and continuity (L = 0.5, g = 2 by default)
# --------------------------------------------------------------------------------------------------------------------


def _extent_evaluator(tmp_path, uv_rgb):
    path = write_capture(str(tmp_path / "c.h5"), uv_rgb, [[[Pixel("trunk")]]])
    return make_evaluator(path, n_envs=1)


def test_extent_examples_from_the_definition(tmp_path, uv_rgb):
    with _extent_evaluator(tmp_path, uv_rgb) as ev:
        counts = ev.binned_tree.bin_counts["env_0"]
        counts[CUSTOM_IDS["trunk"], 0:20:2, 0] = 1  # every other length bin: coverage 0.5, gaps of 1 -> continuous
        counts[CUSTOM_IDS["branch"], 0:10, 0] = 1  # first half only: coverage 0.5, 10-bin gap at the tip -> not
        counts[CUSTOM_IDS["spur"], 5:20, 0] = 1  # misses its first 5 bins: end gaps count, 5 > spur g = 4 -> not
        ext = ev.instance_extent_per_instance("env_0", min_hits=1)
    t, b, s = CUSTOM_IDS["trunk"], CUSTOM_IDS["branch"], CUSTOM_IDS["spur"]
    assert ext["length_coverage"].tolist() == pytest.approx([0.5, 0.5, 15 / 20])
    assert ext["longest_gap"][[t, b, s]].tolist() == [1, 10, 5]
    assert ext["continuous"][[t, b, s]].tolist() == [True, False, False]


def test_extent_thresholds_and_theta(tmp_path, uv_rgb):
    with _extent_evaluator(tmp_path, uv_rgb) as ev:
        counts = ev.binned_tree.bin_counts["env_0"]
        trunk = CUSTOM_IDS["trunk"]
        counts[trunk, :, 0] = 3  # every length bin: 3 hits in theta bin 0 ...
        counts[trunk, :, 1] = 2  # ... plus 2 in theta bin 1 -> 5 per length bin
        one, five = (ev.instance_extent_per_instance("env_0", k) for k in (1, 5))
    assert one["length_coverage"][trunk] == 1.0 and five["length_coverage"][trunk] == 1.0
    # theta bins sum over all 20 length bins: 60 and 40 hits, both >= 5
    assert one["theta_coverage"][trunk] == pytest.approx(2 / 36)
    assert five["theta_coverage"][trunk] == pytest.approx(2 / 36)
    assert bool(five["continuous"][trunk])


def test_unseen_instances_count_as_zero_in_class_means(tmp_path, uv_rgb):
    with _extent_evaluator(tmp_path, uv_rgb) as ev:
        ev.binned_tree.bin_counts["env_0"][CUSTOM_IDS["trunk"], :, 0] = 5  # only the trunk is seen, fully
        ext = ev.instance_extent(min_hits=(1, 5))["env_0"]
        per = ev.instance_extent_per_instance("env_0", 1)
    assert ext["trunk"][5] == {"length_coverage": 1.0, "theta_coverage": pytest.approx(1 / 36), "continuous": 1.0}
    assert ext["spur"][1] == {"length_coverage": 0.0, "theta_coverage": 0.0, "continuous": 0.0}
    assert ext["total"][1]["length_coverage"] == pytest.approx(1 / 3)
    assert per["longest_gap"][CUSTOM_IDS["spur"]] == 20  # unseen: the whole length is one gap


def test_per_class_continuity_thresholds(tmp_path, uv_rgb):
    path = write_capture(str(tmp_path / "c.h5"), uv_rgb, [[[Pixel("trunk")]]])
    cfg = nbv.CoverageConfig(continuity_min_length_frac={"spur": 0.2}, continuity_max_gap_bins={"spur": 12})
    ev = nbv.NBVEvaluator(
        trial_data_path=path,
        observation_config=nbv.ObservationConfig(),
        gain_config=nbv.GainConfig(),
        coverage_config=cfg,
        device="cpu",
        n_envs=1,
    )
    with ev:
        counts = ev.binned_tree.bin_counts["env_0"]
        counts[CUSTOM_IDS["branch"], 0:10, 0] = 1  # coverage 0.5, 10-bin gap: fails the default g = 2
        counts[CUSTOM_IDS["spur"], 0:10, 0] = 1  # same pattern, passes the spur-specific g = 12, L = 0.2
        ext = ev.instance_extent_per_instance("env_0", min_hits=1)
    assert not bool(ext["continuous"][CUSTOM_IDS["branch"]])
    assert bool(ext["continuous"][CUSTOM_IDS["spur"]])


@pytest.mark.parametrize(
    "cls, gap, continuous",
    [
        ("spur", 4, True),  # ~10 cm spur: a 4-bin (~2 cm) break is tolerated
        ("spur", 5, False),
        ("branch", 2, True),  # ~50 cm branch: a 2-bin (~5 cm) break is tolerated
        ("branch", 3, False),
        ("trunk", 2, True),
        ("trunk", 3, False),
    ],
)
def test_default_gap_tolerance_per_class(tmp_path, uv_rgb, cls, gap, continuous):
    with _extent_evaluator(tmp_path, uv_rgb) as ev:
        ev.binned_tree.bin_counts["env_0"][CUSTOM_IDS[cls], gap:20, 0] = 1  # first `gap` length bins missed
        ext = ev.instance_extent_per_instance("env_0", min_hits=1)
    assert bool(ext["continuous"][CUSTOM_IDS[cls]]) is continuous
