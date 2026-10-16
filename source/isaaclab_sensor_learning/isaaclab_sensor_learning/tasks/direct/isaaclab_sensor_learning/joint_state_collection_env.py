#!/usr/bin/env python3
from __future__ import annotations

import torch
from collections.abc import Sequence


from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
import isaaclab.utils.math as math_utils

from isaaclab_sensor_learning.controllers.limit_aware_ik import LimitAwareIK
import isaaclab_sensor_learning.sensor.yaml_to_cfg as yaml_to_cfg
from isaaclab_sensor_learning.sensor.generate_rig import RigGenerator

from .pose_data_capture_env_cfg import PoseDataCaptureEnvCfg

import pprint as pp


class PoseDataCaptureEnv(DirectRLEnv):
    cfg: PoseDataCaptureEnvCfg

    def __init__(self, cfg: PoseDataCaptureEnvCfg, render_mode: str | None = None, **kwargs):
        # self._sensor_cfgs = yaml_to_cfg.rig_yaml_to_sensor_cfgs(rig_yaml_path=cfg.rig_yaml_path) # TODO: change to dict later

        super().__init__(cfg, render_mode, **kwargs)

        self.robot_entity_cfg = SceneEntityCfg(name="robot", joint_names=[".*"], body_names=["ur10e__tool0"])
        self.robot_entity_cfg.resolve(self.scene)
        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = self.robot.data.default_joint_vel.clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.robot.reset()

        # controllers
        self.ik_controller = DifferentialIKController(
            cfg=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": 0.01}
            ),
            num_envs=self.cfg.num_envs,
            device=self.device,
        )
        self.ik_controller.reset()

        # actions
        self.actions = torch.zeros(
            size=(self.num_envs, self.ik_controller.action_dim), dtype=torch.float32, device=self.device
        )  # (N, 7) target pose in world frame

        return

    def _setup_scene(self):
        self.robot = self.scene.articulations["robot"]

        self.scene.clone_environments(copy_from_source=True)

        # markers
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.eef_markers = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
        self.goal_markers = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
        return

    def _apply_action(self) -> None:
        self.ik_controller.reset()
        self.ik_controller.set_command(self.actions)

        if self.robot.is_fixed_base:
            eef_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        else:
            eef_jacobi_idx = self.robot_entity_cfg.body_ids[0]
        jacobi_joint_ids = [j + self.robot.num_base_dofs for j in self.robot_entity_cfg.joint_ids]
        jacobians = self.robot.data.body_link_jacobian_w.torch[:, eef_jacobi_idx, :, jacobi_joint_ids]
        eef_pose_w = self.robot.data.body_pose_w.torch[:, self.robot_entity_cfg.body_ids[0]]
        root_pose_w = self.robot.data.root_pose_w.torch
        joint_pos = self.robot.data.joint_pos.torch[:, self.robot_entity_cfg.joint_ids]

        # Primay task: compute joint position target from desired end-effector pose
        eef_pos_b, eef_quat_b = math_utils.subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], eef_pose_w[:, 0:3], eef_pose_w[:, 3:7]
        )
        joint_pos_des = self.ik_controller.compute(eef_pos_b, eef_quat_b, jacobians, joint_pos)
        # # apply actions
        self.robot.set_joint_position_target_index(target=joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids)
        self.scene.write_data_to_sim()

        # update marker poses
        eef_pose_w = self.robot.data.body_state_w.torch[:, self.robot_entity_cfg.body_ids[0], 0:7]
        self.eef_markers.visualize(eef_pose_w[:, 0:3], eef_pose_w[:, 3:7])
        self.goal_markers.visualize(self.actions[:, 0:3] + self.scene.env_origins, self.actions[:, 3:7])
        return

    def _get_observations(self) -> dict:
        sensor_data = {}
        return sensor_data
    
    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        )

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        return