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
parser.add_argument(
    "--allow-root", action="store_true", default=True, help="Allow running as root user (not recommended)."
)
parser.add_argument(
    "--robot",
    type=str,
    default="ur10e",
    help="Name of the robot to use in the evaluation.",
    choices=["fr3", "ur10e", "panda"],
)
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

# from isaaclab.markers import VisualizationMarkers
# from isaaclab.markers.config import FRAME_MARKER_CFG
# from isaaclab.managers import SceneEntityCfg
# import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab_tasks.utils import parse_env_cfg

# from isaaclab_sensor_learning.motion_planning import diff_ik as diff_ik_mp
import isaaclab_sensor_learning.tasks  # noqa: F401
from isaaclab_sensor_learning.utils import h5py_utils, usd_utils

import data_utils.data_logger as data_logger
import data_utils.pose_generator as pose_generator
import pprint as pp
import os

import time
import h5py as h5

import pprint as pp


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # reset environment
    env.reset()
    robot = env.unwrapped.robot
    num_envs = env.unwrapped.cfg.num_envs
    # env.unwrapped.goal_pose = torch.zeros((env.unwrapped.cfg.num_envs, 7), dtype=torch.float32, device=env.unwrapped.device)

    # load poses from file
    with h5.File("/home/luke/IsaacLab/source/isaaclab-sensor-learning/data/SL-PDC_20260721_140000.h5", "r") as f:
        trial_dict = h5py_utils.h5_to_dict(f)

    for key, val in trial_dict['trees'].items():
        pp.pprint(val['attrs'])
    
        

    goal_pose = torch.zeros((num_envs, 7), dtype=torch.float32)

    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            # check if the robot's end-effector is close to the goal pose
            robot_base_w = robot.data.root_pose_w.torch
            robot_eef_w = robot.data.body_pose_w.torch[:, robot.find_bodies("ur10e__tool0")[0]].reshape(-1, 7)
            # position
            robot_positions_b = torch.subtract(robot_eef_w[:, 0:3], robot_base_w[:, 0:3])
            positions_close = torch.isclose(robot_positions_b[:, 0:3], goal_pose[:, 0:3], atol=0.002)

            actions = ...
            observations, rewards, terminated, truncated, info = env.step(actions)
            break
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
