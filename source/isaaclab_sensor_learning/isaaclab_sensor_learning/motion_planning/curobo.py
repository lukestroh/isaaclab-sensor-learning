#!/usr/bin/env python3
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.trajectory_optimizer import TrajectoryOptimizer, TrajectoryOptimizerCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.scene import Scene, Cuboid, Sphere, Mesh
from curobo.types import JointState, GoalToolPose

import torch

from typing import List, Optional
from pathlib import Path


def get_motion_planner():
    motion_planner_cfg = MotionPlannerCfg(
        robot="/workspace/src/isaaclab-pose-data-capture/source/pose_data_capture/pose_data_capture/config/curobo/fr3.yml"
    )
    planner = MotionPlanner(motion_planner_cfg)
    planner.warmup()
    return planner


def _plot_trajectory(
    positions: torch.Tensor,
    joint_names: List[str],
    dt: float,
    save_path: str,
    title: str = "Joint Trajectory",
    phase_boundaries: Optional[List[int]] = None,
    phase_labels: Optional[List[str]] = None,
):
    """Plot joint positions over time and save to *save_path*.

    Args:
        positions: Joint positions tensor of shape ``(timesteps, n_joints)``.
        joint_names: Label for each joint.
        dt: Time step between waypoints (seconds).
        save_path: Output file path (e.g. ``"trajectory.pdf"``).
        title: Plot title.
        phase_boundaries: Timestep indices where a new phase starts.
        phase_labels: Label for each phase (length must match *phase_boundaries*).
    """
    import matplotlib.pyplot as plt

    pos_np = positions.cpu().numpy()
    n_steps = pos_np.shape[0]
    t = [i * dt for i in range(n_steps)]

    fig, ax = plt.subplots(figsize=(10, 5))
    for j, name in enumerate(joint_names):
        ax.plot(t, pos_np[:, j], label=name)

    if phase_boundaries and phase_labels:
        for idx, label in zip(phase_boundaries, phase_labels):
            ax.axvline(x=idx * dt, color="grey", linestyle="--", linewidth=0.8)
            ax.text(
                idx * dt,
                ax.get_ylim()[1],
                f" {label}",
                fontsize=8,
                va="top",
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint position (rad)")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Trajectory plot saved to: {save_path}")


def plan_path(planner, start_joint_state, goal_joint_state, output_dir: Optional[Path] = None):
    q_start = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0),
        joint_names=planner.joint_names,
    )

    goal_pose = GoalToolPose(
        tool_frames=planner.tool_frames,
        position=torch.tensor([[[[[0.4, 1.0, 0.3]]]]], device="cuda", dtype=torch.float32),
        quaternion=torch.tensor([[[[[1.0, 0.0, 0.0, 0.0]]]]], device="cuda", dtype=torch.float32),
    )

    result = planner.plan_pose(goal_pose, q_start)

    trajectory = planner.plan(start_joint_state, goal_joint_state)

    interp_dt = planner.trajopt_solver.config.interpolation_dt
    if result is not None and result.success.any():
        print("✓ Planning succeeded!")
        interpolated = result.get_interpolated_plan()
        n_waypoints = interpolated.position.shape[-2]
        print(f"Trajectory has {n_waypoints} waypoints")
        print(f"Duration: {n_waypoints * interp_dt:.2f}s")

        _plot_trajectory(
            interpolated.position.squeeze(0),
            planner.joint_names,
            dt=interp_dt,
            save_path=str(output_dir / "motion_plan.pdf"),
            title="Pose-to-Pose Trajectory",
        )
        return True
    else:
        print("✗ Planning failed - try adjusting the goal or obstacles")
        return False

    # return trajectory
