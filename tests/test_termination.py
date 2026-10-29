"""Per-env termination: each condition stops only its env, records the reason, and other envs keep running."""

import numpy as np
import pytest
import torch
from conftest import Pixel, make_evaluator, write_capture

from isaaclab_sensor_learning.nbv.termination import TerminationConfig, TerminationReason, TerminationTracker

R = TerminationReason


def tracker(n_envs=2, **cfg):
    return TerminationTracker(n_envs, TerminationConfig(**cfg))


def test_coverage_target_stops_only_that_env():
    tr = tracker(coverage_target=0.5)
    assert tr.update(t=1.0, step=120, coverage={0: 0.51, 1: 0.2}) == [0]
    assert tr.states[0].reason == R.COVERAGE_TARGET and tr.states[0].t_end == 1.0 and tr.states[0].step_end == 120
    assert tr.active_env_ids == [1] and not tr.all_done


def test_time_budget():
    tr = tracker(max_sim_time_s=10.0)
    assert tr.update(t=9.9, step=1, coverage={}) == []
    assert tr.update(t=10.0, step=2, coverage={}) == [0, 1]
    assert tr.all_done and all(s.reason == R.TIME_BUDGET for s in tr.states)


def test_first_reason_is_kept():
    tr = tracker(max_sim_time_s=1.0)
    tr.update(t=2.0, step=1, coverage={})
    assert not tr.finish(0, R.PLATEAU, t=3.0, step=2)
    assert tr.states[0].reason == R.TIME_BUDGET and tr.states[0].t_end == 2.0


def test_goal_budget():
    tr = tracker(n_envs=1, max_goals=2)
    assert not tr.on_goal_assigned(0, t=0, step=0, best_score=10.0)
    assert not tr.on_goal_assigned(0, t=1, step=1, best_score=10.0)
    assert tr.on_goal_assigned(0, t=2, step=2, best_score=10.0)
    assert tr.states[0].reason == R.GOAL_BUDGET and tr.states[0].n_goals == 2


def test_predicted_gain_floor_is_relative_to_first_goal():
    tr = tracker(n_envs=1, min_rel_predicted_gain=0.01)
    assert not tr.on_goal_assigned(0, t=0, step=0, best_score=200.0)
    assert not tr.on_goal_assigned(0, t=1, step=1, best_score=2.0)  # exactly 1%: still above the floor
    assert tr.on_goal_assigned(0, t=2, step=2, best_score=1.9)
    assert tr.states[0].reason == R.PREDICTED_GAIN_FLOOR


def test_plateau_over_window_of_goals():
    tr = tracker(n_envs=1, plateau_window_goals=3, plateau_min_gain=0.01)
    for i, cov in enumerate([0.10, 0.20, 0.30, 0.305]):  # 4th goal: gain over last 3 goals is 0.205 -> fine
        tr.update(t=i, step=i, coverage={0: cov})
        assert not tr.on_goal_reached(0, t=i, step=i)
    for i, cov in enumerate([0.306, 0.307, 0.308], start=4):  # gain over last 3 goals drops below 0.01
        tr.update(t=i, step=i, coverage={0: cov})
        stopped = tr.on_goal_reached(0, t=i, step=i)
    assert stopped and tr.states[0].reason == R.PLATEAU


def test_timeouts_consecutive_and_total():
    tr = tracker(n_envs=2, goal_timeout_s=5.0, max_consecutive_timeouts=3, max_total_timeouts=4)
    tr.on_goal_assigned(0, t=0.0, step=0, best_score=1.0)
    assert not tr.goal_timed_out(0, t=5.0) and tr.goal_timed_out(0, t=5.1)
    # env 0: three in a row -> failed
    assert not tr.on_goal_timeout(0, t=6, step=6) and not tr.on_goal_timeout(0, t=7, step=7)
    assert tr.on_goal_timeout(0, t=8, step=8) and tr.states[0].reason == R.UNREACHABLE_GOALS
    # env 1: a reached goal resets the streak, but the total still counts
    for t in (1, 2):
        tr.on_goal_timeout(1, t=t, step=t)
    tr.on_goal_reached(1, t=3, step=3)
    assert not tr.on_goal_timeout(1, t=4, step=4)
    assert tr.on_goal_timeout(1, t=5, step=5) and tr.states[1].reason == R.UNREACHABLE_GOALS


def test_summary_rows():
    tr = tracker(coverage_target=0.5)
    tr.update(t=3.5, step=420, coverage={0: 0.6, 1: 0.1})
    rows = tr.summary()
    assert rows[0]["reason"] == int(R.COVERAGE_TARGET) and rows[0]["reason_name"] == "COVERAGE_TARGET"
    assert rows[1]["reason_name"] == "RUNNING" and np.isnan(rows[1]["t_end"])
    assert rows[0]["final_coverage"] == 0.6


# --------------------------------------------------------------------------------------------------------------------
# Candidate search (NBVEvaluator.get_candidate_pose_idxs)
# --------------------------------------------------------------------------------------------------------------------


def _line_capture(tmp_path, uv_rgb, xs=(0.0, 0.3, 0.9, 2.0)):
    positions = np.zeros((2, len(xs), 3), np.float32)
    positions[:, :, 0] = xs
    path = write_capture(str(tmp_path / "c.h5"), uv_rgb, [[[Pixel("trunk")]] * len(xs)] * 2, eef_positions=positions)
    return make_evaluator(path, n_envs=2, pose_search_radius=0.5)


def test_candidates_exclude_visited_and_grow_radius_per_env(tmp_path, uv_rgb):
    with _line_capture(tmp_path, uv_rgb) as ev:
        visited = torch.zeros((2, 4), dtype=torch.bool)
        visited[0, :2] = True  # env 0 already used the two poses near the origin
        here = torch.zeros((2, 3))
        cands, exhausted = ev.get_candidate_pose_idxs(here, [0, 1], visited=visited, max_radius=1.0)
    # env 0: nearest unvisited pose is 0.9 m away -> radius grows to 1.1 m and finds it; env 1 unaffected (0.5 m)
    assert cands.tolist() == [[0, 2], [1, 0], [1, 1]]
    assert exhausted == []


def test_candidates_exhausted_beyond_max_radius_or_all_visited(tmp_path, uv_rgb):
    with _line_capture(tmp_path, uv_rgb) as ev:
        visited = torch.zeros((2, 4), dtype=torch.bool)
        visited[0, :3] = True  # env 0: only the pose 2.0 m away is left
        visited[1, :] = True  # env 1: nothing left
        cands, exhausted = ev.get_candidate_pose_idxs(torch.zeros((2, 3)), [0, 1], visited=visited, max_radius=1.0)
    assert cands.numel() == 0 and exhausted == [0, 1]


def test_best_pose_scores_are_returned(tmp_path, uv_rgb):
    with _line_capture(tmp_path, uv_rgb) as ev:
        best, scores = ev.compute_env_info_gain_from_pose_idxs(
            torch.tensor([0]), torch.tensor([[0, 0], [0, 1]]), return_scores=True
        )
    assert best.tolist() == [[0, 0]] and scores.shape == (1,) and float(scores[0]) > 0


def test_coverage_target_can_be_disabled():
    tr = tracker(coverage_target=None, max_sim_time_s=10.0)
    assert tr.update(t=1.0, step=1, coverage={0: 0.99, 1: 1.0}) == []  # full coverage, but no target
    assert tr.update(t=10.0, step=2, coverage={0: 0.99, 1: 1.0}) == [0, 1]
    assert all(s.reason == R.TIME_BUDGET for s in tr.states)
