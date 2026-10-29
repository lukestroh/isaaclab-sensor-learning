#!/usr/bin/env python3
"""Collects joint states from the robot and saves them to a file."""

from isaaclab.app import AppLauncher
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="Pose data capture script")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Template-Joint-State-Collection-Direct-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=0, help="Seed argument for deterministic runs.")
# parser.add_argument("--enable_cameras", action="store_true", default=True, help="Enable cameras in the environment.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import numpy as np

import isaaclab.utils.math as math_utils

from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_sensor_learning.tasks  # noqa: F401

import data_utils.joint_state_data_logger as joint_state_data_logger
import data_utils.evaluation_data_logger as evaluation_data_logger
import utils.pose_generator as pose_generator
import pprint as pp
import os
import re

import time
import isaaclab.sim as sim_utils
import isaaclab.sim.views as sim_views
from isaaclab.sim.utils import stage as stage_utils
from isaaclab.sim.utils import semantics as semantic_utils
from pxr import Usd, UsdGeom

from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene

from isaaclab.utils.seed import configure_seed

# from isaaclab_sensor_learning.utils import logging_conf

def _map(
    values: torch.Tensor,
    base_lo: float | torch.Tensor,
    base_high: float | torch.Tensor,
    target_lo: torch.Tensor,
    target_high: torch.Tensor,
) -> torch.Tensor:
    """Linearly map `values` from [base_lo, base_hi] to [target_lo, target_hi], elementwise."""
    scaled = (values - base_lo) / (base_high - base_lo)
    return target_lo + scaled * (target_high - target_lo)

def main():
    configure_seed(seed=args_cli.seed)

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # reset environment
    env.reset()
    robot: Articulation = env.unwrapped.robot
    scene: InteractiveScene = env.unwrapped.scene
    sim: sim_utils.SimulationContext = env.unwrapped.sim
    cuda_device = env.unwrapped.device

    # Set up data logger
    dlog = joint_state_data_logger.JointStateDataLogger()


    # Generate discrete poses
    r_range = (0.9, 1.4)
    r_size = 5
    theta_cyl_range = (-np.pi / 8, np.pi / 8)
    theta_cyl_size = 10
    z_range = (1.0, 1.8)
    z_size = 5
    theta_ori_range = (-np.pi / 4, np.pi / 4)
    theta_ori_size = 5
    phi_ori_range = (-np.pi / 2, np.pi / 2)
    phi_ori_size = 5
    start_orientation = np.array([0, 1, 0])
    discrete_poses = pose_generator.generate_uniform_cylindrical_poses(
        r_range=r_range,
        r_size=r_size,
        theta_cyl_range=theta_cyl_range,
        theta_cyl_size=theta_cyl_size,
        z_range=z_range,
        z_size=z_size,
        theta_ori_range=theta_ori_range,
        theta_ori_size=theta_ori_size,
        phi_ori_range=phi_ori_range,
        phi_ori_size=phi_ori_size,
        start_orientation=start_orientation,
    )
    discrete_poses = np.asarray(discrete_poses)

    joint_states_list = []

    # simulate environment
    pose_idx = 0
    step_count = 0
    step_count_limit = 500

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel)
    sim.step()

    while simulation_app.is_running():
        with torch.inference_mode():
            if pose_idx >= len(discrete_poses):
                print("\n[INFO]: All poses completed, exiting simulation.")
                dlog.save_joint_states(joint_states=joint_states_list)
                break

            print(f"[INFO]: {pose_idx+1}/{len(discrete_poses)}", end="\r")

            goal_pose = torch.tensor(
                discrete_poses[pose_idx][np.newaxis, :], dtype=torch.float32, device=env.unwrapped.device
            )

            # Actions
            actions = torch.tensor(goal_pose.detach().clone(), dtype=torch.float32, device=env.unwrapped.device).repeat(
                env.unwrapped.num_envs, 1
            )
            # step
            observations, rewards, terminated, truncated, info = env.step(actions)
            step_count += 1

            # check if the robot's end-effector is close to the goal pose
            robot_base_w = robot.data.root_pose_w.torch
            robot_eef_w = robot.data.body_pose_w.torch[:, robot.find_bodies("ur10e__wrist_3_link")[0]].reshape(-1, 7)
            # position
            robot_eef_pos_b = torch.subtract(robot_eef_w[:, 0:3], robot_base_w[:, 0:3])
            positions_close = torch.isclose(robot_eef_pos_b[:, 0:3], goal_pose[:, 0:3], atol=0.002)
            # orientation
            quat_diff_err = math_utils.quat_error_magnitude(actions[:, 3:7], robot_eef_w[:, 3:7])
            orientations_close = torch.isclose(quat_diff_err, torch.tensor(0.0), atol=0.006)
            
            # save observations and poses if the robot's end-effector is close to the goal pose
            if positions_close.all() and orientations_close.all():
                # joint states
                robot_joint_pos = robot.data.joint_pos.torch
                joint_states_list.append(robot_joint_pos[0].detach().cpu()) # append the first env's joint states since all envs are the same
                joint_pos = robot.data.default_joint_pos.clone()
                joint_vel = robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel)
                robot.reset()
                step_count = 0
                pose_idx += 1
                continue

            if step_count >= step_count_limit:
                print(f"\n[INFO]: Step count limit reached for pose {pose_idx+1}/{len(discrete_poses)}. Resetting robot to a new random joint state.")
                joint_limits = robot.data.joint_limits.torch
                # try from a different start pose, randomize the robot's joint positions
                rand_joint_pos_normalized = torch.rand_like(robot.data.default_joint_pos)
                rand_joint_pos = _map(
                    rand_joint_pos_normalized,
                    base_lo=0.0,
                    base_high=1.0,
                    target_lo=joint_limits[..., 0],
                    target_high=joint_limits[..., 1],
                )
                
                joint_pos = rand_joint_pos.clone()
                joint_vel = robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel)
                robot.reset()
                step_count = 0
                continue

    print("\n[INFO]: Closing environment and simulation app.")
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

