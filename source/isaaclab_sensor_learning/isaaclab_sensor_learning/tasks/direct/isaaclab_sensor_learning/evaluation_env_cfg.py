#!/usr/bin/env python3

from isaaclab.assets import ArticulationCfg
# from isaaclab.controllers.config.rmp_flow import FR3_RMPFLOW_CFG, FRANKA_RMPFLOW_CFG, UR10_RMPFLOW_CFG
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, RenderCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from isaaclab_sensor_learning.robot.franka_cfg import FRANKA_FR3_CFG
from isaaclab_sensor_learning.motion_planning.rmp_flow import FR3_RMPFLOW_CFG
# from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG


from pathlib import Path


@configclass
class PoseEvaluationEnvCfg(DirectRLEnvCfg):
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

    # robot
    robot_cfg: ArticulationCfg = FRANKA_FR3_CFG.replace(prim_path="/World/robot")



    # sensors: NOTE: offsetconfigs for sensors are from the parent prim. this needs to be combined with internal sensor receiver offset defined in the sensor yaml.

    rig_yaml_path: str = ""
    # rig_yaml_path: str = "pose_data_capture/pose_data_capture/assets/rigs/rig0.yaml"
    tree_usd_path: str = ""
