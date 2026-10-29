#!/usr/bin/env python3
"""
Post-run plots of one NBV evaluation: coverage metrics and EEF movement over time, per env and averaged over envs.

Runs outside Isaac Sim (plain venv python) and writes a standalone HTML report. Either entry point works:

    python scripts/plot_nbv_run.py --show                       # newest data/evaluation/*.h5
    python scripts/data_utils/plot/nbv_run_plots.py --eval 20261028_114336 --show

Coverage over time is rebuilt by replaying the evaluation file's ``sparse_bin_updates`` and evaluating the same
metric functions the simulation uses (``nbv/coverage_metrics.py``, loaded by file path because the
``isaaclab_sensor_learning`` package can only be imported inside Isaac Sim).
"""
import argparse
import glob
import importlib.util
import os
import time
import webbrowser
from dataclasses import dataclass

import h5py
import numpy as np
import plotly.colors
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

_COVERAGE_METRICS_PATH = os.path.join(
    os.path.dirname(__file__), *[".."] * 3, "source", "isaaclab_sensor_learning", "isaaclab_sensor_learning", "nbv",
    "coverage_metrics.py",
)
_spec = importlib.util.spec_from_file_location("coverage_metrics", os.path.abspath(_COVERAGE_METRICS_PATH))
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)

GROUPS = (*cm.INSTANCE_CLASSES, "total")
ENV_COLORS = plotly.colors.qualitative.Dark24  # 24 distinct colors, so ~20 envs don't repeat
MEAN_COLOR = "black"
# The mean is drawn thick but semi-transparent so the per-env lines underneath stay readable.
ENV_LINE = dict(width=1.5, opacity=0.75)
MEAN_LINE = dict(width=3.0, opacity=0.65)
STOP_LINE = dict(width=1.0, opacity=1.0)  # dotted vertical line at each env's stop time
K_DASH = ["solid", "dash", "dot", "dashdot"]  # line style per hit threshold k
_LEGEND_BELOW = dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0)


@dataclass
class NBVRun:
    name: str
    n_envs: int
    t: np.ndarray  # (T,) NBV time (s), 0 at the start of the NBV loop
    pose_b: np.ndarray  # (n_envs, T, 7) EEF pose, position in base frame, quaternion (x, y, z, w)
    goal_times: list[np.ndarray]  # per env: NBV time of each assigned goal
    goal_poses: list[np.ndarray]  # per env: (n_goals, 7) goal pose, position in base frame, quaternion (x, y, z, w)
    sparse_updates: list[np.ndarray]  # per env: structured rows (timestamp, instance_idx, length_bin, theta_bin, count)
    counts_shape: list[tuple]  # per env: (n_instances, n_length_bins, n_theta_bins)
    class_ids: list[torch.Tensor]  # per env: index into INSTANCE_CLASSES per instance
    coverage_config: "cm.CoverageConfig"
    termination: dict  # termination table columns
    tree_positions_b: np.ndarray  # (n_envs, 3) tree base position in the EEF path's frame (world - robot base)
    t0: float  # absolute sim time of the NBV start


def _coverage_config_from_attrs(attrs) -> "cm.CoverageConfig":
    """Rebuild the CoverageConfig a run used from its ``instance_extent`` attrs."""
    kwargs = {"instance_min_hits": tuple(int(k) for k in np.atleast_1d(attrs["instance_min_hits"]))}
    for name in ("continuity_min_length_frac", "continuity_max_gap_bins"):
        if name in attrs:
            kwargs[name] = attrs[name].item()
        per_class = {key.split("/", 1)[1]: attrs[key].item() for key in attrs if key.startswith(f"{name}/")}
        if per_class:
            kwargs[name] = per_class
    return cm.CoverageConfig(**kwargs)


def load_run(eval_path: str, capture_path: str | None = None) -> NBVRun:
    """Load an evaluation h5 (and its capture h5, for the instance -> class map)."""
    with h5py.File(eval_path, "r") as e:
        sibling = os.path.join(os.path.dirname(os.path.abspath(eval_path)), "capture.h5")  # NSGA evals/<rig_id>/ layout
        if capture_path is None and os.path.exists(sibling):
            capture_path = sibling
        sibling = os.path.join(os.path.dirname(os.path.abspath(eval_path)), "capture.h5")  # NSGA evals/<rig_id>/
        if capture_path is None and os.path.exists(sibling):
            capture_path = sibling
        if capture_path is None:
            capture_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(eval_path))),
                "capture",
                f"SL-PDC_{e.attrs['datetime_id']}.h5",
            )
        timestamps = e["timestamp"][()]
        t0 = float(timestamps.min())
        pose_b = e["pose"][()]
        n_envs = pose_b.shape[0]
        goals = [e[f"nbv_goal/{k}"][()] for k in range(n_envs)]
        goal_times = [g["timestamp"] - t0 for g in goals]
        goal_poses = [g["pose"] for g in goals]
        sparse = [e[f"sparse_bin_updates/{k}"][()] for k in range(n_envs)]
        shapes = [tuple(e[f"bin_counts/env_{k}"].shape) for k in range(n_envs)]
        cfg = _coverage_config_from_attrs(dict(e["instance_extent"].attrs))
        termination = {k: e["termination"][k][()] for k in e["termination"]}
        frames = {k: e[f"scene_frames/{k}"][()] for k in e["scene_frames"]}

    with h5py.File(capture_path, "r") as c:
        class_ids = []
        for k in range(n_envs):
            prim_map = {p: int(i) for p, i in c[f"trees/env_{k}"].attrs.items() if str(p).startswith("/World")}
            ids = cm.class_ids_from_prim_map(prim_map)
            padded = torch.full((shapes[k][0],), -1, dtype=torch.int64)
            padded[: ids.numel()] = ids
            class_ids.append(padded)
        # tree pose in trees/env_<i> is relative to the env origin; EEF poses are relative to the robot base
        tree_local = np.stack([np.asarray(c[f"trees/env_{k}"].attrs["position"], dtype=float) for k in range(n_envs)])
        tree_positions_b = frames["env_origins"] + tree_local - frames["robot_base_pose_w"][:, :3]

    return NBVRun(
        name=os.path.splitext(os.path.basename(eval_path))[0],
        n_envs=n_envs,
        t=timestamps - t0,
        pose_b=pose_b,
        goal_times=goal_times,
        goal_poses=goal_poses,
        sparse_updates=sparse,
        counts_shape=shapes,
        class_ids=class_ids,
        coverage_config=cfg,
        termination=termination,
        tree_positions_b=tree_positions_b,
        t0=t0,
    )


def coverage_over_time(run: NBVRun, dt: float = 1.0) -> tuple[np.ndarray, dict]:
    """
    Replay sparse bin updates and evaluate coverage metrics every ``dt`` seconds of NBV time.

    :return: (sample_times, series) with ``series[metric][group][k]`` an (n_envs, n_samples) array. Metrics are
        "instances_seen" plus the extent metrics ("length_coverage", "theta_coverage", "continuous").
    """
    times = np.arange(0.0, run.t.max() + dt, dt)
    ks = run.coverage_config.instance_min_hits
    metrics = ("instances_seen", *cm.EXTENT_METRICS)
    series = {m: {g: {k: np.full((run.n_envs, len(times)), np.nan) for k in ks} for g in GROUPS} for m in metrics}

    for env in range(run.n_envs):
        updates = np.sort(run.sparse_updates[env], order="timestamp")
        update_t = updates["timestamp"] - run.t0
        counts = np.zeros(run.counts_shape[env], dtype=np.int32)
        idx = (updates["instance_idx"], updates["length_bin"], updates["theta_bin"])
        cursor = 0
        for i, t in enumerate(times):
            end = int(np.searchsorted(update_t, t, side="right"))
            if end > cursor:
                np.add.at(counts, tuple(a[cursor:end] for a in idx), updates["count_delta"][cursor:end])
                cursor = end
            counts_t = torch.from_numpy(counts)
            detection = cm.instance_coverage(counts_t, run.class_ids[env], ks)
            extent = cm.instance_extent(counts_t, run.class_ids[env], ks, run.coverage_config)
            for g in GROUPS:
                for k in ks:
                    series["instances_seen"][g][k][env, i] = detection[g]["seen"][k]
                    for m in cm.EXTENT_METRICS:
                        series[m][g][k][env, i] = extent[g][k][m]
    return times, series


def movement_over_time(run: NBVRun) -> dict[str, np.ndarray]:
    """Cumulative EEF path length (m), rotation (rad) and goals assigned, each (n_envs, T) over ``run.t``."""
    pos, quat = run.pose_b[..., :3], run.pose_b[..., 3:7]
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    step_len = np.linalg.norm(np.diff(pos, axis=1), axis=-1)
    # same angle as scripts/moo/path_evaluation.quat_angular_distance (q and -q are the same rotation)
    step_rot = 2.0 * np.arccos(np.clip(np.abs(np.sum(quat[:, :-1] * quat[:, 1:], axis=-1)), 0.0, 1.0))
    zeros = np.zeros((run.n_envs, 1))
    goals = np.stack([np.searchsorted(np.sort(g), run.t, side="right") for g in run.goal_times])
    return {
        "path_length": np.concatenate([zeros, np.cumsum(step_len, axis=1)], axis=1),
        "rotation": np.concatenate([zeros, np.cumsum(step_rot, axis=1)], axis=1),
        "goals": goals.astype(float),
    }


def _env_label(run: NBVRun, env: int) -> str:
    reason = run.termination["reason_name"][env]
    reason = reason.decode() if isinstance(reason, bytes) else str(reason)
    return f"env_{env} ({reason} @ {float(run.termination['t_end'][env]):.0f} s)"


def _add_series(fig, row, col, x, per_env, run, dash, show_legend, legend_suffix=""):
    """Thin per-env lines plus a thick mean line."""
    for env in range(run.n_envs):
        fig.add_trace(
            go.Scatter(
                x=x, y=per_env[env], mode="lines",
                line=dict(color=ENV_COLORS[env % len(ENV_COLORS)], width=ENV_LINE["width"], dash=dash),
                opacity=ENV_LINE["opacity"],
                name=_env_label(run, env) + legend_suffix, legendgroup=f"env_{env}{legend_suffix}",
                showlegend=show_legend,
            ),
            row=row, col=col,
        )
    fig.add_trace(
        go.Scatter(
            x=x, y=np.nanmean(per_env, axis=0), mode="lines",
            line=dict(color=MEAN_COLOR, width=MEAN_LINE["width"], dash=dash), opacity=MEAN_LINE["opacity"],
            name=f"mean over envs{legend_suffix}", legendgroup=f"mean{legend_suffix}", showlegend=show_legend,
        ),
        row=row, col=col,
    )


def _add_termination_lines(fig, run: NBVRun):
    for env in range(run.n_envs):
        fig.add_vline(
            x=float(run.termination["t_end"][env]),
            line=dict(color=ENV_COLORS[env % len(ENV_COLORS)], width=STOP_LINE["width"], dash="dot"),
            opacity=STOP_LINE["opacity"], row="all", col="all",
        )


def coverage_figure(run: NBVRun, times: np.ndarray, series: dict) -> go.Figure:
    """
    Grid: rows = metric, columns = instance class (+ total; n = instances in env_0). Line style = hit threshold k
    (solid >= 1, dashed >= 5); thin colored = per env, thick black = mean over envs; dotted verticals = env stop times.
    """
    row_titles = {
        "instances_seen": "instances seen",
        "length_coverage": "length coverage",
        "theta_coverage": "theta coverage",
        "continuous": "continuous",
    }
    ks = run.coverage_config.instance_min_hits
    n_instances = {g: int((run.class_ids[0] == i).sum()) for i, g in enumerate(cm.INSTANCE_CLASSES)}
    n_instances["total"] = int(run.class_ids[0].numel())
    fig = make_subplots(
        rows=len(row_titles), cols=len(GROUPS), shared_xaxes=True, shared_yaxes=True,
        column_titles=[f"{g} (n={n_instances[g]})" for g in GROUPS], row_titles=list(row_titles.values()),
        horizontal_spacing=0.03, vertical_spacing=0.04,
    )
    for r, metric in enumerate(row_titles, start=1):
        for c, group in enumerate(GROUPS, start=1):
            for ki, k in enumerate(ks):
                _add_series(
                    fig, r, c, times, series[metric][group][k], run, dash=K_DASH[ki % len(K_DASH)],
                    show_legend=(r == 1 and c == 1), legend_suffix=f", >={k} hits",
                )
    _add_termination_lines(fig, run)
    fig.update_yaxes(range=[0, 1.02])
    fig.update_xaxes(title_text="NBV time (s)", row=len(row_titles))
    cfg = run.coverage_config
    fig.update_layout(
        title=(
            f"{run.name}: coverage over time (fraction of all instances of the class; "
            f"continuous: L={cfg.continuity_min_length_frac}, g={cfg.continuity_max_gap_bins})"
        ),
        height=260 * len(row_titles), template="plotly_white", legend=_LEGEND_BELOW,
    )
    return fig


def movement_figure(run: NBVRun, movement: dict[str, np.ndarray]) -> go.Figure:
    titles = {"path_length": "EEF path length (m)", "rotation": "EEF rotation (rad)", "goals": "goals assigned"}
    fig = make_subplots(rows=1, cols=len(titles), subplot_titles=list(titles.values()), horizontal_spacing=0.06)
    for c, key in enumerate(titles, start=1):
        _add_series(fig, 1, c, run.t, movement[key], run, dash="solid", show_legend=(c == 1))
    _add_termination_lines(fig, run)
    fig.update_xaxes(title_text="NBV time (s)")
    fig.update_layout(
        title=f"{run.name}: EEF movement over time (cumulative)", height=480, template="plotly_white",
        legend={**_LEGEND_BELOW, "y": -0.22},  # single row of plots: needs more room below the x-axis titles
    )
    return fig


def _stop_time(run: NBVRun, env: int) -> float:
    """NBV time at which the env stopped (the termination table is written once every env has stopped)."""
    return float(run.termination["t_end"][env])


def eef_path_figure(run: NBVRun, dt: float = 0.05) -> go.Figure:
    """
    3D EEF path in the robot base frame, colored by NBV time, with goals, start, and stop markers. Each env's tree
    is marked at its base with a vertical line along the trunk axis.

    Each env's path is cut at its stop time (a finished env holds still) and downsampled to one point per ``dt`` s.
    A dropdown shows a single env or all of them overlaid (all envs share the base frame).
    """
    stride = max(1, int(round(dt / max(float(np.median(np.diff(run.t))), 1e-9))))
    t_max = max(_stop_time(run, env) for env in range(run.n_envs))
    color_axis = dict(colorscale="Viridis", cmin=0.0, cmax=t_max)
    fig = go.Figure()
    traces_per_env = []
    for env in range(run.n_envs):
        t_stop = _stop_time(run, env)
        keep = np.flatnonzero(run.t <= t_stop)[::stride]
        if keep[-1] != np.flatnonzero(run.t <= t_stop)[-1]:
            keep = np.append(keep, np.flatnonzero(run.t <= t_stop)[-1])  # always include the stop point
        xyz, t = run.pose_b[env, keep, :3], run.t[keep]
        goal_mask = run.goal_times[env] <= t_stop
        goal_xyz, goal_t = run.goal_poses[env][goal_mask, :3], run.goal_times[env][goal_mask]
        label = _env_label(run, env)
        first = len(fig.data)
        fig.add_trace(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines", name=f"{label} path", legendgroup=f"env_{env}",
            line=dict(color=t, width=5, **color_axis, colorbar=dict(title="NBV time (s)") if env == 0 else None,
                      showscale=env == 0),
            customdata=t, hovertemplate="t = %{customdata:.1f} s<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>",
        ))
        fig.add_trace(go.Scatter3d(
            x=goal_xyz[:, 0], y=goal_xyz[:, 1], z=goal_xyz[:, 2], mode="markers", name=f"env_{env} goals",
            legendgroup=f"env_{env}", marker=dict(size=3, color=goal_t, **color_axis, symbol="circle"),
            customdata=np.stack([goal_t, np.arange(len(goal_t))], axis=-1),
            hovertemplate="goal %{customdata[1]} at t = %{customdata[0]:.1f} s<extra></extra>",
        ))
        fig.add_trace(go.Scatter3d(
            x=xyz[[0, -1], 0], y=xyz[[0, -1], 1], z=xyz[[0, -1], 2], mode="markers+text", name=f"env_{env} start/stop",
            legendgroup=f"env_{env}", text=["start", label.split(" ", 1)[-1].strip("()") or "end"],
            textposition="top center", marker=dict(size=[6, 7], color=["green", "red"], symbol=["diamond", "x"]),
            hoverinfo="text",
        ))
        tree = run.tree_positions_b[env]
        top = max(float(run.pose_b[:, :, 2].max()), tree[2] + 0.5)
        fig.add_trace(go.Scatter3d(
            x=[tree[0], tree[0]], y=[tree[1], tree[1]], z=[tree[2], top], mode="lines+markers+text",
            name=f"env_{env} tree", legendgroup=f"env_{env}", text=[f"tree (env_{env})", ""],
            textposition="bottom center", line=dict(color="saddlebrown", width=6),
            marker=dict(size=[7, 0], color="saddlebrown", symbol="square"),
            hovertemplate=f"tree base (env_{env})<br>(%{{x:.2f}}, %{{y:.2f}}, %{{z:.2f}})<extra></extra>",
        ))
        traces_per_env.append(list(range(first, len(fig.data))))

    n_traces = len(fig.data)

    def visible(envs):
        v = [False] * n_traces
        for env in envs:
            for i in traces_per_env[env]:
                v[i] = True
        return v

    buttons = [dict(label="all envs", method="update", args=[{"visible": visible(range(run.n_envs))}])]
    buttons += [dict(label=f"env_{env}", method="update", args=[{"visible": visible([env])}]) for env in range(run.n_envs)]
    for i, v in enumerate(visible([0])):
        fig.data[i].visible = v
    fig.update_layout(
        title=f"{run.name}: EEF path (robot base frame), colored by NBV time",
        scene=dict(aspectmode="data", xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)"),
        # active=1: the menu opens on env_0 (button 0 is "all envs"), matching the initial visibility
        updatemenus=[dict(buttons=buttons, active=1, direction="down", x=0.0, xanchor="left", y=1.08, yanchor="top")],
        height=750, template="plotly_white", legend={**_LEGEND_BELOW, "y": 0.0, "itemsizing": "constant"},
    )
    return fig


def write_report(figures: list[go.Figure], path: str) -> str:
    """Write several figures into one standalone HTML file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    parts = [fig.to_html(full_html=False, include_plotlyjs=("cdn" if i == 0 else False)) for i, fig in enumerate(figures)]
    with open(path, "w") as f:
        f.write("<html><head><meta charset='utf-8'></head><body>" + "\n".join(parts) + "</body></html>")
    return path


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 3))
EVAL_DIR = os.path.join(PROJECT_DIR, "data", "evaluation")


def resolve_eval_path(arg: str | None) -> str:
    """Path, datetime id (e.g. 20261028_114336), or None for the newest evaluation file."""
    if arg is None:
        files = glob.glob(os.path.join(EVAL_DIR, "SL-PDEVAL_*.h5"))
        if not files:
            raise FileNotFoundError(f"No evaluation files in {EVAL_DIR}")
        return max(files, key=os.path.getmtime)
    if os.path.exists(arg):
        return arg
    path = os.path.join(EVAL_DIR, f"SL-PDEVAL_{arg}.h5")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{arg} is neither a file nor a datetime id in {EVAL_DIR}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", default=None, help="Evaluation h5 path or datetime id (default: newest).")
    parser.add_argument("--capture", default=None, help="Capture h5 path (default: matching SL-PDC_<datetime_id>.h5).")
    parser.add_argument("--dt", type=float, default=1.0, help="Coverage sampling interval in NBV seconds.")
    parser.add_argument("--path-dt", type=float, default=0.05, help="EEF path sampling interval in NBV seconds.")
    parser.add_argument("--out", default=None, help="Output HTML (default: data/plots/<eval name>.html).")
    parser.add_argument("--show", action="store_true", help="Open the report in a browser.")
    args = parser.parse_args()

    eval_path = resolve_eval_path(args.eval)
    print(f"[INFO]: Loading {eval_path}")
    start = time.perf_counter()
    run = load_run(eval_path, args.capture)
    times, series = coverage_over_time(run, dt=args.dt)
    figures = [
        coverage_figure(run, times, series),
        movement_figure(run, movement_over_time(run)),
        eef_path_figure(run, dt=args.path_dt),
    ]

    out = os.path.abspath(args.out or os.path.join(PROJECT_DIR, "data", "plots", f"{run.name}.html"))
    write_report(figures, out)
    print(f"[INFO]: Wrote {out} ({time.perf_counter() - start:.1f} s)")
    if args.show and not webbrowser.open(f"file://{out}"):
        print(f"[WARNING]: Could not open a browser; open {out} manually.")


if __name__ == "__main__":
    main()
