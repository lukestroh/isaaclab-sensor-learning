"""Post-run analysis: coverage_metrics loads without Isaac Sim, and the plot data replays runs correctly."""

import os
import subprocess
import sys

import h5py
import numpy as np
import pytest

PROJECT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(PROJECT, "scripts"))

from data_utils.plot import nbv_run_plots as plots  # noqa: E402


def test_coverage_metrics_loads_without_isaac_sim():
    """The plot script and the optimizer load coverage_metrics.py by path, without the Kit-only package __init__."""
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('cm', {plots._COVERAGE_METRICS_PATH!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "assert 'isaaclab_sensor_learning' not in sys.modules and 'omni' not in sys.modules\n"
        "print(m.INSTANCE_CLASSES)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "trunk" in out.stdout


T0 = 35.0  # absolute sim time at the NBV start (the capture pass came first)
PRIMS = {"/World/envs/env_0/tree/trunkId_0": 0, "/World/envs/env_0/tree/trunkId_0/spurId_0000": 1}


def _write_run(tmp_path):
    """One env, two instances (trunk, spur), 4 s of NBV; the spur gets 3 hits at t = 1 s and 2 more at t = 3 s."""
    cap = tmp_path / "capture" / "SL-PDC_x.h5"
    ev = tmp_path / "evaluation" / "SL-PDEVAL_x.h5"
    cap.parent.mkdir(); ev.parent.mkdir()
    with h5py.File(cap, "w") as f:
        g = f.require_group("trees/env_0")
        g.attrs["position"] = np.array([0.0, 2.0, 0.0])  # relative to the env origin
        for path, cid in PRIMS.items():
            g.attrs[path] = cid
    with h5py.File(ev, "w") as f:
        f.attrs["datetime_id"] = "x"
        t = T0 + np.arange(0, 4.01, 0.5)
        f.create_dataset("timestamp", data=t)
        pose = np.zeros((1, len(t), 7)); pose[..., 6] = 1.0
        pose[0, :, 0] = np.linspace(0, 2, len(t))  # 2 m along x
        f.create_dataset("pose", data=pose)
        goal_dtype = np.dtype([("timestamp", np.float32), ("pose", np.float32, (7,))])
        f.create_dataset("nbv_goal/0", data=np.array([(T0, np.zeros(7)), (T0 + 2, np.zeros(7))], dtype=goal_dtype))
        upd_dtype = np.dtype([("timestamp", np.float32), ("instance_idx", np.int32), ("length_bin", np.int32),
                              ("theta_bin", np.int32), ("count_delta", np.int32)])
        rows = [(T0 + 3, 1, 0, 0, 2), (T0 + 1, 1, 0, 0, 3), (T0 + 1, 0, 5, 5, 1)]  # out of time order on purpose
        f.create_dataset("sparse_bin_updates/0", data=np.array(rows, dtype=upd_dtype))
        f.create_dataset("bin_counts/env_0", data=np.zeros((2, 20, 36), np.int32))
        ext = f.require_group("instance_extent")
        ext.attrs["instance_min_hits"] = np.array([1, 5])
        ext.attrs["continuity_min_length_frac"] = 0.5
        for cls, g in (("trunk", 2), ("branch", 2), ("spur", 4)):
            ext.attrs[f"continuity_max_gap_bins/{cls}"] = g
        f.create_dataset("termination/t_end", data=np.array([4.0]))
        f.create_dataset("termination/reason_name", data=np.array([b"TIME_BUDGET"]))
        f.create_dataset("scene_frames/env_origins", data=np.array([[2.5, 0.0, 0.0]]))
        f.create_dataset("scene_frames/robot_base_pose_w", data=np.array([[2.5, -0.5, 0.1, 0, 0, 0, 1.0]]))
    return str(ev)


def test_coverage_replay_over_time(tmp_path):
    run = plots.load_run(_write_run(tmp_path))
    assert run.t[0] == 0.0 and run.coverage_config.instance_min_hits == (1, 5)
    assert run.coverage_config.continuity_max_gap_bins == {"trunk": 2, "branch": 2, "spur": 4}  # read back per class
    times, series = plots.coverage_over_time(run, dt=1.0)
    spur = series["instances_seen"]["spur"]
    assert times.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert spur[1][0].tolist() == [0, 1, 1, 1, 1]  # >=1 hit from t = 1 s
    assert spur[5][0].tolist() == [0, 0, 0, 1, 1]  # 3 + 2 = 5 hits at t = 3 s
    assert series["instances_seen"]["total"][1][0, -1] == 1.0
    assert series["length_coverage"]["spur"][1][0, -1] == pytest.approx(1 / 20)


def test_movement_over_time(tmp_path):
    run = plots.load_run(_write_run(tmp_path))
    movement = plots.movement_over_time(run)
    assert movement["path_length"][0, -1] == pytest.approx(2.0)
    assert movement["rotation"][0, -1] == pytest.approx(0.0)
    assert movement["goals"][0].tolist() == [1, 1, 1, 1, 2, 2, 2, 2, 2]


def test_eef_path_is_cut_at_stop_time_and_downsampled(tmp_path):
    run = plots.load_run(_write_run(tmp_path))  # 9 samples, 0.5 s apart, x from 0 to 2 m; goals at t = 0 and 2 s
    run.termination = {"t_end": np.array([1.6]), "reason_name": np.array([b"PLATEAU"])}
    fig = plots.eef_path_figure(run, dt=1.0)  # stride 2 samples
    path, goals, start_stop, tree = fig.data
    # samples at t = 0, 1.0 (stride 2), plus the last sample before the stop (t = 1.5)
    assert list(path.customdata) == [0.0, 1.0, 1.5]
    assert len(goals.x) == 1  # the goal assigned at t = 2 s is after the stop
    assert start_stop.text[1] == "PLATEAU @ 2 s"
    assert [b.label for b in fig.layout.updatemenus[0].buttons] == ["all envs", "env_0"]
    assert all(t.visible for t in fig.data)


def test_tree_marker_in_base_frame(tmp_path):
    run = plots.load_run(_write_run(tmp_path))
    np.testing.assert_allclose(run.tree_positions_b[0], [0.0, 2.5, -0.1])  # origin + tree - base
    tree = next(t for t in plots.eef_path_figure(run).data if t.name == "env_0 tree")
    assert (tree.x[0], tree.y[0], tree.z[0]) == pytest.approx((0.0, 2.5, -0.1))
    assert tree.legendgroup == "env_0"  # shown and hidden with its env in the dropdown
