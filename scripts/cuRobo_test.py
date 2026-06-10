#!/usr/bin/env python3
from isaaclab.app import AppLauncher
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="Pose evaluation script for pose data capture")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Template-Pose-Evaluation-Direct-v0", help="Name of the task.")
# parser.add_argument("--enable_cameras", action="store_true", default=True, help="Enable cameras in the environment.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch
import numpy as np

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.managers import SceneEntityCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_tasks.utils import parse_env_cfg

from pose_data_capture.robot import motion_planning as mp
import pose_data_capture.tasks  # noqa: F401
from pose_data_capture.utils import usd_utils
from pose_data_capture.utils import quaternion_utils as qutils

import data_utils.data_logger as data_logger
import data_utils.pose_generator as pose_generator
import pprint as pp
import os


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # reset environment
    env.reset()

    # robot = env.unwrapped.robot
    # ik_controller = env.unwrapped.ik_controller
    # scene = env.unwrapped.scene
    # sim_dt = env.unwrapped.sim.cfg.dt

    # print(scene)
    # print("Joint names:", robot.joint_names)
    # print("Body names:", robot.body_names)
    # print("Num joints:", robot.num_joints)
    # print("Num bodies:", robot.num_bodies)

    # # sim_dt = sim_utils.get_physics_dt()

    # robot_entity_cfg = SceneEntityCfg("robot", joint_names=["fr3_joint.*"], body_names=["fr3_link8"])
    # robot_entity_cfg.resolve(scene)
    # print(robot_entity_cfg)

    # # Markers
    # frame_marker_cfg = FRAME_MARKER_CFG.copy()
    # frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    # eef_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/eef_current"))
    # goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/eef_goal"))

    # count = 0
    # # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():

            actions = torch.zeros((env_cfg.n_envs, env_cfg.action_space), device=env.unwrapped.device)  # dummy actions
            observations, rewards, terminated, truncated, info = env.step(actions)

    return


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()
