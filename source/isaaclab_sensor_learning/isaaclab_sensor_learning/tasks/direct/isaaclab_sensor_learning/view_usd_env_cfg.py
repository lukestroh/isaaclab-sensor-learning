# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import (
    SimulationCfg,
    RenderCfg,
    DomeLightCfg
)
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg

import isaaclab_sensor_learning.utils.usd_utils as usd_utils
# from pose_data_capture.sensor.yaml_to_cfg import rig_yaml_to_sensor_cfgs
import isaaclab.sim as sim_utils
from isaaclab_sensor_learning.articulations.linear_slider import LINEAR_SLIDER_CFG
from isaaclab_sensor_learning.articulations.ag_robot import AG_ROBOT_CFG

from isaaclab.assets.articulation import ArticulationCfg, Articulation

import os


@configclass
class ViewUSDSceneCfg(InteractiveSceneCfg):
     # terrain
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", debug_vis=False)

    dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0)))

    robot = AG_ROBOT_CFG

@configclass
class ViewUSDEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 60.0
    # - spaces definition
    action_space = 7
    observation_space = 4
    state_space = 0
    num_envs = 1
    physics_dt = 1 / 120

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=physics_dt,
        render_interval=decimation,
        render=RenderCfg(enable_global_illumination=True, antialiasing_mode="Off"),  # , physics=NewtonCfg()
    )

    # scene
    scene = ViewUSDSceneCfg(
        num_envs=num_envs,
        lazy_sensor_update=False,  # Change to false for evaluation
        replicate_physics=False,
        env_spacing=5.0,
    )