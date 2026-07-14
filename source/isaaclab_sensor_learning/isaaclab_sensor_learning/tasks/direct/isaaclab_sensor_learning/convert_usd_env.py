# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence


from isaaclab.envs import DirectRLEnv
import isaaclab_sensor_learning.utils.usd_utils as usd_utils

from .convert_usd_env_cfg import ConvertUSDEnvCfg

import os


class ConvertUSDEnv(DirectRLEnv):
    cfg: ConvertUSDEnvCfg

    def __init__(self, cfg: ConvertUSDEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        return

    def _setup_scene(self):
        # self.robot = self.scene["robot"]

        return

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions[:] = actions.clone().to(dtype=torch.float32)
        return

    def _apply_action(self) -> None:
        

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
