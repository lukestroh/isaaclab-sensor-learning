# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg, RenderCfg, MultiUsdFileCfg, RigidBodyBaseCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import NewtonCfg

from isaaclab_sensor_learning import USD_DIR, URDF_DIR, TREES_DIR
from isaaclab_sensor_learning.articulations.ag_robot import AG_ROBOT_UR10e_CFG, AG_ROBOT_UR5e_CFG
from isaaclab_sensor_learning.sensor.generate_rig import RigGenerator
from isaaclab_sensor_learning.tasks.direct.isaaclab_sensor_learning.sensor_learning_scene_cfg import (
    SensorLearningSceneCfg,
)

import glob
import os
from pathlib import Path


@configclass
class PoseDataCaptureEnvCfg(DirectRLEnvCfg):
    """Configuration for the pose data capture environment."""

    # env
    decimation = 2
    episode_length_s = 60.0
    # - spaces definition
    action_space = 7
    observation_space = 4
    state_space = 0
    num_envs = 2
    physics_dt = 1 / 120

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=physics_dt,
        render_interval=decimation,
        render=RenderCfg(enable_global_illumination=True, antialiasing_mode="Off"),  # , physics=NewtonCfg()
    )

    # scene
    scene: SensorLearningSceneCfg = SensorLearningSceneCfg(
        num_envs=num_envs,
        lazy_sensor_update=True,  # Change to false for evaluation
        replicate_physics=False,
        env_spacing=5.0,
    )

    # rig
    rig_yaml_path: str = (
        Path(__file__).parent.parent.parent.parent / "config/rigs/test_rig0.yaml"
    )  # NOTE: Standin for now, will be replaced by arguments from evolutionary outputs
    # # rig_yaml_path: str = "pose_data_capture/pose_data_capture/assets/rigs/rig0.yaml"
