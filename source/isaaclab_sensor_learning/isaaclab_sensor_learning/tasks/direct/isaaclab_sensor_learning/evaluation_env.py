#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import Camera, CameraCfg, MultiMeshRayCasterCamera, MultiMeshRayCasterCameraCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

import pose_data_capture as pdc
import pose_data_capture.utils.usd_utils as usd_utils
import pose_data_capture.utils.quaternion_utils as qutils
from .evaluation_env_cfg import PoseEvaluationEnvCfg

import os


class PoseEvaluationEnv(DirectRLEnv):
    cfg: PoseEvaluationEnvCfg

    def __init__(self, cfg: PoseEvaluationEnvCfg, render_mode: str | None = None, **kwargs):
        self.curr_tree_mesh_prim = None

        self.sensors = {}
        self._prims_removed = False
        super().__init__(cfg, render_mode, **kwargs)
        return

    def _setup_scene(self):
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # tree
        # self.cfg.tree_usd_path = os.path.join(pdc.PKG_DIR, "trees/models", "lpy_envy_00000_uv.usda")
        # self.curr_tree_mesh_prim = usd_utils.load_usd(usd_path=self.cfg.tree_usd_path, name="lpy_envy_00000", env=0)

        # robot
        self.robot = Articulation(
            cfg=self.cfg.robot_cfg.replace(
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.0, 1.0, 0.0), rot=tuple(qutils.xyzw_to_wxyz(np.asarray([0.0, 0.0, 0.0, 1.0])))
                )
            )
        )
        self.cfg.robot_cfg.init_state.pos = (0.0, 1.0, 0.0)
        self.cfg.robot_cfg.init_state.rot = tuple(qutils.xyzw_to_wxyz(np.asarray([0.0, 0.0, 0.0, 1.0])))
        self.robot = Articulation(cfg=self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # # controllers
        # self.ik_controller = DifferentialIKController(
        #     cfg=DifferentialIKControllerCfg(
        #         command_type="pose",
        #         use_relative_mode=False,
        #         ik_method="dls",
        #         ik_params={"lambda_val": 0.01}
        #     ),
        #     num_envs=self.cfg.n_envs,
        #     device=self.device,
        # )

        # Index of fr3_link7 in the body list
        # self.eef_body_idx = self.robot.find_bodies("fr3_link7")[0]

        # add sensors to robot. read rig data from h5 file and spawn cameras based on rig config

        return

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # update end-effector pose based on actions

        return

    def _apply_action(self) -> None:
        # path plan to target pose and execute

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
