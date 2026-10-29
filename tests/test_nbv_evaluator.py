"""NBVEvaluator: which pixels count, how candidate poses are scored, and which pose is picked."""

import numpy as np
import pytest
import torch
from conftest import CUSTOM_IDS, Z_FAR, Z_NEAR, Pixel, make_evaluator, prim_paths, write_capture

N_INSTANCES = len(CUSTOM_IDS)


def dense(pose_delta: torch.Tensor) -> np.ndarray:
    return pose_delta.to_dense().numpy()


def expected_counts(*hits: tuple[str, int, int, int]) -> np.ndarray:
    """Dense (n_instances, 20, 36) counts from (cls, length_bin, theta_bin, count) tuples."""
    counts = np.zeros((N_INSTANCES, 20, 36), np.int32)
    for cls, l, t, n in hits:
        counts[CUSTOM_IDS[cls], l, t] += n
    return counts


# --------------------------------------------------------------------------------------------------------------------
# Pixel validity
# --------------------------------------------------------------------------------------------------------------------


def test_only_valid_labelled_pixels_are_counted(tmp_path, uv_rgb):
    pixels = [
        Pixel("trunk", 2, 3),
        Pixel("trunk", 2, 3),  # same bin twice -> count 2
        Pixel("spur", 7, 30, incidence_deg=30.0),  # inside the 45 deg incidence limit
        Pixel("branch", 1, 1, incidence_deg=60.0),  # too oblique
        Pixel("branch", 1, 2, depth=Z_FAR + 0.5),  # beyond z_far
        Pixel("branch", 1, 3, depth=Z_NEAR / 2),  # closer than z_near
        Pixel(None, 4, 4),  # render id with no label
    ]
    path = write_capture(str(tmp_path / "c.h5"), uv_rgb, [[pixels]])
    with make_evaluator(path, n_envs=1) as ev:
        (delta,) = ev._get_pose_deltas(0, np.array([0]))
    np.testing.assert_array_equal(dense(delta), expected_counts(("trunk", 2, 3, 2), ("spur", 7, 30, 1)))


YAW_90 = (0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4))  # (x, y, z, w): rotate +90 deg about world Z
X, Y, Z = np.eye(3)


@pytest.mark.parametrize(
    "convention, optical_axis, tilt_axis",
    [
        ("ros", Z, X),  # yaw about Z leaves the ROS optical axis (+Z) unchanged
        ("opengl", -Z, X),
        ("world", Y, Z),  # world convention: optical axis is +X, which yaw turns into +Y
    ],
)
def test_incidence_is_measured_against_the_conventions_optical_axis(tmp_path, uv_rgb, convention, optical_axis, tilt_axis):
    """
    Regression: the incidence mask must use the optical axis of the convention pose_w was stored in. Using the
    wrong axis compares normals against the camera's up/side vector and discards most valid pixels.
    """
    pixels = [Pixel("trunk", 5, 5, incidence_deg=10.0), Pixel("spur", 6, 6, incidence_deg=80.0)]
    path = write_capture(
        str(tmp_path / "c.h5"),
        uv_rgb,
        [[pixels]],
        sensor_quat=YAW_90,
        forward_axis=optical_axis,
        tilt_axis=tilt_axis,
        convention=convention,
    )
    with make_evaluator(path, n_envs=1) as ev:
        (delta,) = ev._get_pose_deltas(0, np.array([0]))
    np.testing.assert_array_equal(dense(delta), expected_counts(("trunk", 5, 5, 1)))


# --------------------------------------------------------------------------------------------------------------------
# Loading from disk
# --------------------------------------------------------------------------------------------------------------------


def _varied_capture(tmp_path, uv_rgb, n_poses=7):
    rng = np.random.default_rng(0)
    pixels = [
        [
            [Pixel(str(rng.choice(list(CUSTOM_IDS))), int(rng.integers(20)), int(rng.integers(36))) for _ in range(16)]
            for _ in range(n_poses)
        ]
        for _ in range(2)
    ]
    return write_capture(str(tmp_path / "c.h5"), uv_rgb, pixels)


@pytest.mark.parametrize("batch_size", [1, 3, 100])
@pytest.mark.parametrize("cache", [True, False])
def test_deltas_do_not_depend_on_batching_or_cache(tmp_path, uv_rgb, batch_size, cache):
    path = _varied_capture(tmp_path, uv_rgb)
    pose_ids = np.arange(7)
    with make_evaluator(path, n_envs=2, candidate_batch_size=7) as ref:
        reference = [dense(d) for e in range(2) for d in ref._get_pose_deltas(e, pose_ids)]
    with make_evaluator(path, n_envs=2, candidate_batch_size=batch_size, cache_pose_deltas=cache) as ev:
        for _ in range(2):  # second pass exercises the cache
            got = [dense(d) for e in range(2) for d in ev._get_pose_deltas(e, pose_ids)]
            for a, b in zip(reference, got):
                np.testing.assert_array_equal(a, b)


def test_cache_reads_each_pose_once(tmp_path, uv_rgb, monkeypatch):
    path = _varied_capture(tmp_path, uv_rgb)
    with make_evaluator(path, n_envs=2) as ev:
        loaded = []
        original = ev._load_masked_sensor_batch
        monkeypatch.setattr(
            ev, "_load_masked_sensor_batch", lambda s, e, ids: loaded.extend(ids.tolist()) or original(s, e, ids)
        )
        ev._get_pose_deltas(0, np.array([1, 2, 3]))
        ev._get_pose_deltas(0, np.array([2, 3, 4]))
    assert sorted(loaded) == [1, 2, 3, 4]


# --------------------------------------------------------------------------------------------------------------------
# Scoring and selection
# --------------------------------------------------------------------------------------------------------------------


def test_spur_paths_get_spur_weight_even_though_they_contain_trunk(tmp_path, uv_rgb):
    path = write_capture(str(tmp_path / "c.h5"), uv_rgb, [[[Pixel("trunk")]]])
    with make_evaluator(path, n_envs=1) as ev:
        gains = ev.gain_config.branch_gains
        assert "trunk" in prim_paths(0)["spur"]
        weights = ev.instance_semantic_weight["env_0"]
        assert weights[CUSTOM_IDS["spur"]] == gains["spur"]
        assert weights[CUSTOM_IDS["branch"]] == gains["branch"]
        assert weights[CUSTOM_IDS["trunk"]] == gains["trunk"]


def test_score_matches_closed_form(tmp_path, uv_rgb):
    path = write_capture(str(tmp_path / "c.h5"), uv_rgb, [[[Pixel("trunk")]]])
    with make_evaluator(path, n_envs=1) as ev:
        cfg = ev.gain_config
        delta = torch.tensor(expected_counts(("trunk", 0, 0, 2), ("spur", 3, 4, 1)))
        live = torch.tensor(expected_counts(("trunk", 0, 0, 4), ("branch", 9, 9, 7)))
        weights = ev.instance_semantic_weight["env_0"]
        score = ev.score_pose_delta(delta.to_sparse().coalesce(), live, weights)

    def gain(n):
        return cfg.saturation_lambda / (1 + np.exp(cfg.steepness * (n - cfg.n_midpoint)))

    # only bins the candidate observes contribute, at their projected (live + delta) count
    expected = gain(4 + 2) * cfg.branch_gains["trunk"] + gain(0 + 1) * cfg.branch_gains["spur"]
    assert score == pytest.approx(expected, rel=1e-6)


def _selection_capture(tmp_path, uv_rgb):
    # env 0: pose 0 sees nothing, pose 1 sees 3 trunk bins, pose 2 sees 1 spur bin (weight 10 beats 3 x weight 1)
    # env 1: pose 0 sees 1 branch bin, poses 1-2 see nothing
    trunk3 = [Pixel("trunk", 0, t) for t in range(3)]
    pixels = [[[], trunk3, [Pixel("spur", 4, 4)]], [[Pixel("branch", 2, 2)], [], []]]
    return write_capture(str(tmp_path / "c.h5"), uv_rgb, pixels)


def _all_candidates(n_envs=2, n_poses=3):
    return torch.tensor([[e, p] for e in range(n_envs) for p in range(n_poses)])


def test_picks_highest_weighted_gain_per_env(tmp_path, uv_rgb):
    with make_evaluator(_selection_capture(tmp_path, uv_rgb), n_envs=2) as ev:
        best = ev.compute_env_info_gain_from_pose_idxs(torch.arange(2), _all_candidates())
    assert best.tolist() == [[0, 2], [1, 0]]


def test_saturated_bins_change_the_choice(tmp_path, uv_rgb):
    with make_evaluator(_selection_capture(tmp_path, uv_rgb), n_envs=2) as ev:
        ev.binned_tree.bin_counts["env_0"][CUSTOM_IDS["spur"], 4, 4] = 50  # spur bin already well observed
        best = ev.compute_env_info_gain_from_pose_idxs(torch.arange(2), _all_candidates())
    assert best[0].tolist() == [0, 1]


def test_ties_go_to_lowest_pose_id(tmp_path, uv_rgb):
    same = [Pixel("trunk", 1, 1)]
    path = write_capture(str(tmp_path / "c.h5"), uv_rgb, [[[], same, same]])
    with make_evaluator(path, n_envs=1) as ev:
        best = ev.compute_env_info_gain_from_pose_idxs(torch.arange(1), _all_candidates(n_envs=1))
    assert best.tolist() == [[0, 1]]


def test_only_requested_envs_and_their_candidates_are_returned(tmp_path, uv_rgb):
    with make_evaluator(_selection_capture(tmp_path, uv_rgb), n_envs=2) as ev:
        # env 1 is requested but has no candidates; env 0 is restricted to poses 0-1
        best = ev.compute_env_info_gain_from_pose_idxs(torch.arange(2), torch.tensor([[0, 0], [0, 1]]))
        assert best.tolist() == [[0, 1]]
        assert ev.compute_env_info_gain_from_pose_idxs(torch.tensor([1]), _all_candidates()).tolist() == [[1, 0]]
        assert ev.compute_env_info_gain_from_pose_idxs(torch.tensor([], dtype=torch.long), _all_candidates()).numel() == 0
        with pytest.raises(ValueError):
            ev.compute_env_info_gain_from_pose_idxs(torch.tensor([5]), _all_candidates())


def test_closest_poses_expand_radius_until_something_is_found(tmp_path, uv_rgb):
    positions = np.array([[[0.0, 0, 0], [0.3, 0, 0], [2.0, 0, 0]]], np.float32)
    path = write_capture(str(tmp_path / "c.h5"), uv_rgb, [[[], [], []]], eef_positions=positions)
    with make_evaluator(path, n_envs=1, pose_search_radius=0.5) as ev:
        near = ev.get_closest_pose_idxs(torch.tensor([[0.0, 0, 0]]))
        far = ev.get_closest_pose_idxs(torch.tensor([[5.0, 0, 0]]))  # nothing within 0.5 m
    assert near.tolist() == [[0, 0], [0, 1]]
    assert far.tolist() == [[0, 2]]
