#!/usr/bin/env python3
import numpy as np


def quat_angular_distance(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Angle (rad) between unit quaternions along the last axis. Any leading shape."""
    # abs() handles the q / -q double cover; clip guards arccos against rounding
    return 2.0 * np.arccos(np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0.0, 1.0))


def eef_motion_metrics(poses: np.ndarray, t_final: float) -> dict:
    """Translation and rotation metrics for end-effector trajectories across environments.

    Samples are assumed uniformly spaced from t=0 to t=t_final.

    Args:
        poses: (n_envs, N, 7) array, last axis is (x, y, z, qx, qy, qz, qw). N >= 2.
        t_final: time of the last sample, in seconds.

    Returns:
        dict of (n_envs,) arrays:
          path_length       [m]     sum of distances between consecutive positions
          net_displacement  [m]     straight-line distance, first to last sample
          total_rotation    [rad]   sum of angular distances between consecutive orientations
          net_rotation      [rad]   angular distance, first to last orientation
          mean_lin_speed    [m/s]   path_length / t_final
          mean_ang_speed    [rad/s] total_rotation / t_final
    """
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 3 or poses.shape[-1] != 7 or poses.shape[1] < 2:
        raise ValueError("Expected poses of shape (n_envs, N, 7) with N >= 2.")

    pos = poses[..., :3]
    quat = poses[..., 3:]
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)  # guard against drift

    path_length = np.linalg.norm(np.diff(pos, axis=1), axis=-1).sum(axis=1)
    total_rotation = quat_angular_distance(quat[:, :-1], quat[:, 1:]).sum(axis=1)

    return {
        "path_length": path_length,
        "net_displacement": np.linalg.norm(pos[:, -1] - pos[:, 0], axis=-1),
        "total_rotation": total_rotation,
        "net_rotation": quat_angular_distance(quat[:, 0], quat[:, -1]),
        "mean_lin_speed": path_length / t_final,
        "mean_ang_speed": total_rotation / t_final,
    }
