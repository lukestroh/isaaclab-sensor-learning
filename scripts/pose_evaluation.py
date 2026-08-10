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
import pxr

# from isaaclab.markers import VisualizationMarkers
# from isaaclab.markers.config import FRAME_MARKER_CFG
# from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene
import isaaclab.sim as sim_utils
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


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # reset environment
    env.reset()
    robot: Articulation = env.unwrapped.robot
    num_envs = env.unwrapped.cfg.num_envs
    scene: InteractiveScene = env.unwrapped.scene
    sim: sim_utils.SimulationContext = env.unwrapped.sim

    # # load poses from file
    with h5.File("/home/luke/IsaacLab/source/isaaclab-sensor-learning/data/SL-PDC_20260730_133953.h5", "r") as f:
        trial_dict = h5py_utils.h5_to_dict(f)

    # Set up data logger
    # dlog = data_logger.DataLogger() # TODO: add args for evaluation vs. capture

    # stage = scene.stage
    # for i in range(num_envs):
    #     tree_prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/tree")
    #     # pp.pprint(dir(tree_prim))
    #     trunk_prim = tree_prim.GetChildren()
    #     print(trunk_prim)

    # return

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel)
    sim.step()

    goal_pose = torch.zeros((num_envs, 7), dtype=torch.float32)

    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            # actions = ...
            # observations, rewards, terminated, truncated, info = env.step(actions)

            # # check if the robot's end-effector is close to the goal pose
            # robot_base_w = robot.data.root_pose_w.torch
            # robot_eef_w = robot.data.body_pose_w.torch[:, robot.find_bodies("ur10e__tool0")[0]].reshape(-1, 7)
            # # position
            # robot_positions_b = torch.subtract(robot_eef_w[:, 0:3], robot_base_w[:, 0:3])
            # positions_close = torch.isclose(robot_positions_b[:, 0:3], goal_pose[:, 0:3], atol=0.002)
            # # orientation
            # quat_diff_err = math_utils.quat_error_magnitude(actions[:, 3:7], robot_eef_w[:, 3:7])
            # orientations_close = torch.isclose(quat_diff_err, torch.tensor(0.0), atol=0.006)

            # # save observations and poses if the robot's end-effector is close to the goal pose
            # if positions_close.all() and orientations_close.all():
            #     if pose_idx >= len(discrete_poses):
            #         dlog.save_observations(observations=observations, poses=robot_eef_w, idx=pose_idx, last_obs=True)
            #         break
            #     else:
            #         dlog.save_observations(observations=observations, poses=robot_eef_w, idx=pose_idx)

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
