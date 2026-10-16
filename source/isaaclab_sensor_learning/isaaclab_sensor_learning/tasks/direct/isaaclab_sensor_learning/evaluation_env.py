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

# from isaaclab_sensor_learning.controllers.limit_aware_ik import LimitAwareIK
from isaaclab_sensor_learning.sensor.generate_rig import RigGenerator
import isaaclab_sensor_learning.sensor.yaml_to_cfg as yaml_to_cfg
from .evaluation_env_cfg import PoseEvaluationEnvCfg


class PoseEvaluationEnv(DirectRLEnv):
    cfg: PoseEvaluationEnvCfg

    def __init__(self, cfg: PoseEvaluationEnvCfg, render_mode: str | None = None, **kwargs):
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

        # # print(self.robot_entity_cfg.joint_ids)
        # self.limit_aware_ik_controller = LimitAwareIK(
        #     ik_controller=self.ik_controller,
        #     num_envs=self.cfg.num_envs,
        #     num_joints=len(self.robot_entity_cfg.joint_ids),
        #     device=self.device,
        # )
        # self.limit_aware_ik_controller.reset()

        # actions
        self.actions = torch.zeros(
            size=(self.num_envs, self.ik_controller.action_dim), dtype=torch.float32, device=self.device
        )  # (N, 7) target pose in world frame

        return

    def _setup_scene(self) -> None:
        self._rig_cfg = yaml_to_cfg.load_yaml(yaml_path=self.cfg.rig_yaml_path)
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

        # controllers
        self.ik_controller = DifferentialIKController(
            cfg=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": 0.0001}
            ),
            num_envs=self.cfg.num_envs,
            device=self.device,
        )
        self.ik_controller.reset()
        self.actions = torch.zeros(
            (self.cfg.num_envs, self.ik_controller.action_dim), device=self.device
        )  # (N, 7) target pose in world frame

        # Markers
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.eef_markers = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
        self.goal_markers = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

        # add sensors to robot. read rig data from h5 file and spawn cameras based on rig config

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions()
        return

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions[:] = actions.clone().to(dtype=torch.float32)
        return

    def _apply_action(self) -> None:

        eef_pose_w = self.robot.data.body_pose_w.torch[:, self.robot_entity_cfg.body_ids[0]]

        self.ik_controller.reset()
        self.ik_controller.set_command(self.actions)  # (N, 7discrete_poses.shape[0]) target pose in world frame

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

        # apply actions
        self.robot.set_joint_position_target_index(target=joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids)
        self.scene.write_data_to_sim()
        # self.sim.forward()
        # self.sim.render()
        # self.sim.step()
        # self.scene.update(self.sim.get_physics_dt())
        # check if the robot's end-effector is close to the goal pose

        # robot_base_w = self.robot.data.root_pose_w.torch
        # robot_eef_w = self.robot.data.body_pose_w.torch[:, self.robot.find_bodies("ur10e__tool0")[0]].reshape(-1, 7)
        # # position
        # robot_positions_b = torch.subtract(robot_eef_w[:, 0:3], robot_base_w[:, 0:3])
        # positions_close = torch.isclose(robot_positions_b[:, 0:3], self.actions[:, 0:3], atol=0.002)
        # # orientation
        # quat_diff_err = math_utils.quat_error_magnitude(self.actions[:, 3:7], robot_eef_w[:, 3:7])
        # orientations_close = torch.isclose(quat_diff_err, torch.tensor(0.0), atol=0.006)

        # manually update Camera positions to ensure they are in sync with the robot's end-effector
        for sensor in self.sensors.values():
            if not isinstance(sensor, Camera):
                continue
            sensor_cfg = sensor.cfg
            parent_offset = sensor_cfg.offset
            new_sensor_pos_w, new_sensor_quat_w = math_utils.combine_frame_transforms(
                t01=eef_pose_w[:, 0:3],
                q01=eef_pose_w[:, 3:7],
                t12=torch.tensor(parent_offset.pos, device=self.device).unsqueeze(0).repeat(self.cfg.num_envs, 1),
                q12=torch.tensor(parent_offset.rot, device=self.device).unsqueeze(0).repeat(self.cfg.num_envs, 1),
            )
            sensor.set_world_poses(positions=new_sensor_pos_w, orientations=new_sensor_quat_w)

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
