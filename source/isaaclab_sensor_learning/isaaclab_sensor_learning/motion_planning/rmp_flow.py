#!/usr/bin/env python3
from isaaclab.controllers import RmpFlowControllerCfg
from isaaclab.utils.math import quat_slerp
import os
import torch

from isaaclab_sensor_learning import URDF_DIR

_RMP_CONFIG_DIR = (
    "/workspace/src/isaaclab-sensor-learning/source/isaaclab_sensor_learning/isaaclab_sensor_learning/motion_planning"
)


FR3_RMPFLOW_CFG = RmpFlowControllerCfg(
    config_file=os.path.join(_RMP_CONFIG_DIR, "FR3", "rmpflow", "fr3_rmpflow_config.yaml"),
    urdf_file=os.path.join(URDF_DIR, "fr3", "fr3.urdf"),
    collision_file=os.path.join(_RMP_CONFIG_DIR, "FR3", "rmpflow", "fr3_robot_description.yaml"),
    frame_name="fr3_link8",
    evaluations_per_frame=5,
)


def interpolate_cartesian_path(
    start_pos: torch.Tensor,  # (3,)
    start_quat: torch.Tensor,  # (4,) wxyz
    end_pos: torch.Tensor,  # (3,)
    end_quat: torch.Tensor,  # (4,) wxyz
    n_waypoints: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (positions, quaternions) each of shape (n_waypoints, 3/4)"""
    t = torch.linspace(0, 1, n_waypoints)
    positions = torch.stack([start_pos + ti * (end_pos - start_pos) for ti in t])
    # slerp for orientation
    quaternions = torch.stack([quat_slerp(start_quat, end_quat, ti) for ti in t])
    return positions, quaternions
