# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
from rich import inspect
import torch
from collections.abc import Sequence


from isaaclab.assets import Articulation, AssetBase, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.controllers import pink_ik
from isaaclab.envs import DirectRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import Camera, CameraCfg, MultiMeshRayCasterCamera, MultiMeshRayCasterCameraCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import isaaclab.utils.math as math_utils

from isaaclab_sensor_learning import URDF_DIR, USD_DIR
from isaaclab_sensor_learning.controllers.limit_aware_ik import LimitAwareIK
import isaaclab_sensor_learning.sensor.yaml_to_cfg as yaml_to_cfg
import isaaclab_sensor_learning.sensor.rig_utils as rig_utils
from isaaclab_sensor_learning.sensor.generate_rig import RigGenerator
import isaaclab_sensor_learning.utils.usd_utils as usd_utils
from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_gr1t2_env_cfg import PinkIKControllerCfg

from pxr import Usd, UsdGeom, Sdf, Gf

from .pose_data_capture_env_cfg import PoseDataCaptureEnvCfg

import pprint as pp


class PoseDataCaptureEnv(DirectRLEnv):
    cfg: PoseDataCaptureEnvCfg

    def __init__(self, cfg: PoseDataCaptureEnvCfg, render_mode: str | None = None, **kwargs):
        # self._sensor_cfgs = yaml_to_cfg.rig_yaml_to_sensor_cfgs(rig_yaml_path=cfg.rig_yaml_path) # TODO: change to dict later
        self._rig_cfg = yaml_to_cfg.load_rig_yaml(rig_yaml_path=cfg.rig_yaml_path)

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

        # print(self.robot_entity_cfg.joint_ids)
        self.limit_aware_ik_controller = LimitAwareIK(
            ik_controller=self.ik_controller,
            num_envs=self.cfg.num_envs,
            num_joints=len(self.robot_entity_cfg.joint_ids),
            device=self.device,
        )
        self.limit_aware_ik_controller.reset()

        # actions
        self.actions = torch.zeros(
            size=(self.num_envs, self.ik_controller.action_dim), dtype=torch.float32, device=self.device
        )  # (N, 7) target pose in world frame
        return

    def _setup_scene(self):

        # add sensors # TODO: Put in reset function and call that? No, that is only for epochs, i need full evolutionary runs
        self.rig_generator = RigGenerator(rig_sensor_cfgs=self._rig_cfg["sensors"])
        self.eef_layout_dict = {
            # "sensor_model": "filler_sensor_model",
            "layout_type": "sphere",
            "radius": 0.2,
            "colatitude": 25,  # degrees
            "max_iter": 200,
            "tol": 1e-6,
            "mesh_points": 8000,
        }
        self.eef_sensor_layout = self.rig_generator.generate_eef_layout(layout_dict=self.eef_layout_dict)
        self.eef_sensor_cfgs = self.rig_generator.generate_eef_sensor_cfgs(sensor_layout=self.eef_sensor_layout)

        self.robot = self.scene.articulations["robot"]

        # add sensors to the scene
        self.eef_sensors = self.rig_generator.generate_rig(sensor_cfgs=self.eef_sensor_cfgs)
        # TODO: Add lidar in generate_rig
        self.scene.sensors.update(self.eef_sensors)
        self.sensors = self.scene.sensors

        self.scene.clone_environments(copy_from_source=True)

        # markers
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.eef_markers = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
        self.goal_markers = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
        return

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions[:] = actions.clone().to(dtype=torch.float32)
        return

    def _apply_action(self) -> None:
        # self.robot.reset()
        # joint_pos = self.robot.data.default_joint_pos.clone()
        # joint_vel = self.robot.data.default_joint_vel.clone()
        # self.robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel)

        self.ik_controller.reset()
        self.ik_controller.set_command(self.actions)  # (N, 7discrete_poses.shape[0]) target pose in world frame

        joint_limits = self.robot.data.soft_joint_pos_limits.torch
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

        # joint_pos_des = self.limit_aware_ik_controller.solve(
        #     eef_pos_b, eef_quat_b, jacobians, joint_pos, joint_limits, max_iters=10, eps=1e-3
        # )

        # print("target joint pos:\n", joint_pos_des)
        # print("Joint pos:\n", joint_pos)
        # print("target pose:\n", self.actions)
        # print("eef pose:\n", eef_pose_w)
        # print("eef_pose_b:\n", torch.cat([eef_pos_b, eef_quat_b], dim=-1))

        # print("joint limits:\n", joint_limits)

        # apply actions
        self.robot.set_joint_position_target_index(target=joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids)
        self.scene.write_data_to_sim()
        # self.sim.step()
        # self.sim.forward()
        # self.sim.render()
        # self.sim.step()
        # self.scene.update(self.sim.get_physics_dt())

        # Set angles directly (doesn't work?)
        # self.robot.reset()
        # joint_pos = self.robot.data.default_joint_pos.clone()
        # joint_vel = self.robot.data.default_joint_vel.clone()
        # self.robot.set_joint_position_target(joint_pos_des)

        # self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros(size=joint_pos_des.shape, device=joint_pos_des.device))
        # print(eef_pose_w)
        # self.robot.write_joint_position_to_sim_index(position=joint_pos_des)
        # self.sim.forward()
        # self.sim.render()
        # self.sim.step()

        # manually update sensor positions to ensure they are in sync with the robot's end-effector
        for sensor in self.sensors.values():
            sensor_cfg = sensor.cfg
            parent_offset = sensor_cfg.offset
            new_sensor_pos_w, new_sensor_quat_w = math_utils.combine_frame_transforms(
                t01=eef_pose_w[:, 0:3],
                q01=eef_pose_w[:, 3:7],
                t12=torch.tensor(parent_offset.pos, device=self.device).unsqueeze(0).repeat(self.cfg.num_envs, 1),
                q12=torch.tensor(parent_offset.rot, device=self.device).unsqueeze(0).repeat(self.cfg.num_envs, 1),
            )
            sensor.set_world_poses(positions=new_sensor_pos_w, orientations=new_sensor_quat_w)
            # print(f"Sensor {sensor_cfg")
            # print(parent_offset)
            # sensor.set_world_poses()

        # joint_pos_des2 = torch.tensor(joint_pos_des2[torch.newaxis, :], dtype=torch.float32, device=self.device)

        # self.robot.reset()

        # update marker poses
        eef_pose_w = self.robot.data.body_state_w.torch[:, self.robot_entity_cfg.body_ids[0], 0:7]
        self.eef_markers.visualize(eef_pose_w[:, 0:3], eef_pose_w[:, 3:7])
        self.goal_markers.visualize(self.actions[:, 0:3] + self.scene.env_origins, self.actions[:, 3:7])

        return

    def _get_observations(self) -> dict:
        sensor_data = {}
        for sensor_name, sensor in self.sensors.items():
            sensor_data[sensor_name] = sensor.data
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
