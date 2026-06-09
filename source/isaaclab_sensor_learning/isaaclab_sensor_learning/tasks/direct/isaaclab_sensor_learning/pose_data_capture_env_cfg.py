# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, RenderCfg
from isaaclab.utils import configclass

from pathlib import Path


@configclass
class PoseDataCaptureEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 60.0
    # - spaces definition
    action_space = 1
    observation_space = 4
    state_space = 0
    n_envs = 1

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, render_interval=decimation, render=RenderCfg(antialiasing_mode="Off")
    )

    # scene
    scene = InteractiveSceneCfg(
        num_envs=n_envs,
        lazy_sensor_update=True,  # Change to false for evaluation
        replicate_physics=True,
        env_spacing=2.0,
    )

    rig_yaml_path: str = (
        Path(__file__).parent.parent.parent.parent / "config/rigs/rig0.yaml"
    )  # NOTE: Standin for now, will be replaced by arguments from evolutionary outputs
    # rig_yaml_path: str = "pose_data_capture/pose_data_capture/assets/rigs/rig0.yaml"
    tree_usd_path: str = ""
