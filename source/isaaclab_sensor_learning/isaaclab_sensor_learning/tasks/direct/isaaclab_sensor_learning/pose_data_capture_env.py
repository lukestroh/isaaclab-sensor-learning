# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import Camera, CameraCfg, MultiMeshRayCasterCamera, MultiMeshRayCasterCameraCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

import pose_data_capture as pdc
import pose_data_capture.utils.usd_utils as usd_utils
import pose_data_capture.sensor.yaml_to_cfg as yaml_to_cfg
from .pose_data_capture_env_cfg import PoseDataCaptureEnvCfg

# from pose_data_capture.sensor.yaml_to_cfg import rig_yaml_to_sensor_cfgs

import os


class PoseDataCaptureEnv(DirectRLEnv):
    cfg: PoseDataCaptureEnvCfg

    def __init__(self, cfg: PoseDataCaptureEnvCfg, render_mode: str | None = None, **kwargs):
        self._curr_tree_mesh_prim = None
        if not cfg.rig_yaml_path:
            raise ValueError("cfg.rig_yaml_path must be set before instantiating PoseDataCaptureEnv.")
        self._sensor_cfgs = yaml_to_cfg.rig_yaml_to_sensor_cfgs(rig_yaml_path=cfg.rig_yaml_path)

        self.camera_poses: np.ndarray
        self.curr_pose_idx = 0

        self.sensors = {}

        super().__init__(cfg, render_mode, **kwargs)
        return

    def _setup_scene(self):
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add dome light
        # light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0))
        # light_cfg.func("/World/Light", light_cfg)

        # add tree
        self.cfg.tree_usd_path = os.path.join(pdc.PKG_DIR, "trees/models", "lpy_envy_00000_uv.usda")
        self._curr_tree_mesh_prim = usd_utils.load_usd(usd_path=self.cfg.tree_usd_path, name="lpy_envy_00000", env=0)
        # add sensors
        for sensor_name, sensor_cfg in self._sensor_cfgs.items():
            if isinstance(sensor_cfg, CameraCfg):
                self.sensors[sensor_name] = Camera(cfg=sensor_cfg)
            elif isinstance(sensor_cfg, MultiMeshRayCasterCameraCfg):
                self.sensors[sensor_name] = MultiMeshRayCasterCamera(cfg=sensor_cfg)
            else:
                raise ValueError(f"Unsupported sensor config type: {type(sensor_cfg)} for sensor '{sensor_name}'")
        return

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().to(dtype=torch.float32)
        for sensor_name, sensor in self.sensors.items():
            sensor.set_world_poses(
                positions=self.actions[0, :3][torch.newaxis, :],
                orientations=self.actions[0, 3:][torch.newaxis, :],
                convention="ros",
            )
        return

    def _apply_action(self) -> None:
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
