#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera, CameraCfg, MultiMeshRayCasterCamera, MultiMeshRayCasterCameraCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import subtract_frame_transforms

import isaaclab_sensor_learning as isl
import isaaclab_sensor_learning.utils.usd_utils as usd_utils
import isaaclab_sensor_learning.utils.quaternion_utils as qutils
from .evaluation_env_cfg import PoseEvaluationEnvCfg

import os


class PoseEvaluationEnv(DirectRLEnv):
    cfg: PoseEvaluationEnvCfg

    def __init__(self, cfg: PoseEvaluationEnvCfg, render_mode: str | None = None, **kwargs):
        self.curr_tree_mesh_prim = None

        self.sensors = {}
        self._prims_removed = False
        super().__init__(cfg, render_mode, **kwargs)

        self.robot_entity_cfg = SceneEntityCfg(
            name="robot",
            joint_names=["fr3_joint.*"],
            body_names=["fr3_link8"],
        )
        self.robot_entity_cfg.resolve(self.scene)

        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = self.robot.data.default_joint_vel.clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.robot.reset()
        return

    def _setup_scene(self):
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # tree
        # self.cfg.tree_usd_path = os.path.join(pdc.PKG_DIR, "trees/models", "lpy_envy_00000_uv.usda")
        # self.curr_tree_mesh_prim = usd_utils.load_usd(usd_path=self.cfg.tree_usd_path, name="lpy_envy_00000", env=0)

        # robot
        # self.robot = Articulation(
        #     cfg=self.cfg.robot_cfg.replace(
        #         init_state=ArticulationCfg.InitialStateCfg(
        #             pos=(0.0, 1.0, 0.0), rot=tuple(qutils.xyzw_to_wxyz(np.asarray([0.0, 0.0, 0.0, 1.0])))
        #         )
        #     )
        # )
        self.cfg.robot_cfg.init_state.pos = (0.0, 0.0, 0.0)
        self.cfg.robot_cfg.init_state.rot = tuple(qutils.xyzw_to_wxyz(np.asarray([0.0, 0.0, 0.0, 1.0])))
        self.robot = Articulation(cfg=self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # controllers
        self.ik_controller = DifferentialIKController(
            cfg=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": 0.01}
            ),
            num_envs=self.cfg.n_envs,
            device=self.device,
        )
        self.ik_controller.reset()

        
        

        # self.rmp_flow_controller = RmpFlowController(
        #     cfg=self.cfg.rmp_flow_cfg,
        #     device=self.device,
        # )

        # add sensors to robot. read rig data from h5 file and spawn cameras based on rig config
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions()
        return

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().to(dtype=torch.float32).reshape((1,-1))   
        return

    def _apply_action(self) -> None:
        # path plan to target pose and execute
        # self.rmp_flow_controller.set_command(command=self.actions)
        # pos_target, vel_target = self.rmp_flow_controller.compute()
        # # print(f"RMP flow controller output: {res}")

        # self.robot.set_joint_position_target(pos_target)
        # self.robot.write_data_to_sim()
        # # env.unwrapped.sim.step()
        # self.robot.update(self.sim.get_physics_dt())

        # print(self.robot_entity_cfg)
        self.ik_controller.reset()
        self.ik_controller.set_command(self.actions) # (N, 7) target pose in world frame

        if self.robot.is_fixed_base:
            eef_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        else:
            eef_jacobi_idx = self.robot_entity_cfg.body_ids[0]

        jacobians = self.robot.root_physx_view.get_jacobians()[:, eef_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
        eef_pose_w = self.robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]]
        root_pose_w = self.robot.data.root_pose_w
        joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        # compute frame in base frame
        eef_pos_b, eef_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], eef_pose_w[:, 0:3], eef_pose_w[:, 3:7]
        )
        joint_pos_des = self.ik_controller.compute(eef_pos_b, eef_quat_b, jacobians, joint_pos)
        # apply actions
        self.robot.set_joint_position_target(joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids)
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.sim.get_physics_dt())


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
