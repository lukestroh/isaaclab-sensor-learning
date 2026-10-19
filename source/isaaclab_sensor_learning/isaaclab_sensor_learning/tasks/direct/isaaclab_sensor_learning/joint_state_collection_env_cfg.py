#!/usr/bin/env python3
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg, RenderCfg, MultiUsdFileCfg, RigidBodyBaseCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_sensor_learning.tasks.direct.isaaclab_sensor_learning import common_env_cfg_params as env_params
from isaaclab_sensor_learning.tasks.direct.isaaclab_sensor_learning.sensor_learning_scene_cfg import (
    SensorLearningSceneCfg,
)


@configclass
class JointStateCollectionEnvCfg(DirectRLEnvCfg):
    """Configuration for the sensor learning environment."""

    # env
    decimation = env_params.decimation
    episode_length_s = env_params.episode_length_s
    # - spaces definition
    action_space = env_params.action_space
    observation_space = env_params.observation_space
    state_space = env_params.state_space
    num_envs = env_params.num_envs
    physics_dt = env_params.physics_dt

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=physics_dt,
        render_interval=decimation,
        render=RenderCfg(enable_global_illumination=True, antialiasing_mode="Off"),  # , physics=NewtonCfg()
    )

    # scene
    scene: SensorLearningSceneCfg = SensorLearningSceneCfg(
        num_envs=num_envs,
        lazy_sensor_update=False,  # Change to false for evaluation
        replicate_physics=False,
        env_spacing=5.0,
    )