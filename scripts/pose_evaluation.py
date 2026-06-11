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
parser.add_argument("--allow-root", action="store_true", default=True, help="Allow running as root user (not recommended).")
parser.add_argument("--robot", type=str, default="ur10e", help="Name of the robot to use in the evaluation.", choices=["fr3", "ur10e", "panda"])
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
from isaaclab_sensor_learning.utils import usd_utils
from isaaclab_sensor_learning.utils import quaternion_utils as qutils

import data_utils.data_logger as data_logger
import data_utils.pose_generator as pose_generator
import pprint as pp
import os

import time


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # reset environment
    env.reset()
    robot = env.unwrapped.robot
    # ik_controller = env.unwrapped.ik_controller
    # ik_controller.reset()



# rmp_flow_controller = env.unwrapped.rmp_flow_controller
    # rmp_flow_controller.initialize(prim_paths_expr="/World/envs/env_.*/robot")

    def get_tf(source_frame, target_frame, pose, quat):
        if source_frame == target_frame:
            return torch.cat([pose, quat], dim=-1)
        source_pos, source_quat = pose, quat
        print(f"Source pos: {source_pos}, source quat: {source_quat}")
        
        target_pos, target_quat = (
            robot.data.body_pos_w[:, robot.find_bodies(target_frame)[0]],
            robot.data.body_quat_w[:, robot.find_bodies(target_frame)[0]],
        )
        print(f"Target pos: {target_pos}, target quat: {target_quat}")
        relative_tf = math_utils.subtract_frame_transforms(source_pos, source_quat, target_pos, target_quat)
        return relative_tf

    # robot_base_pos = robot.data.body_pos_w[:, robot.find_bodies("fr3_link1")[0]]
    # robot_base_quat = robot.data.body_quat_w[:, robot.find_bodies("fr3_link1")[0]]

    target_pos = torch.tensor([[0.5, 0.5, 0.7]], device=env.unwrapped.device)
    target_quat = torch.tensor([[0.707, 0, 0.707, 0]], device=env.unwrapped.device)  # wxyz

    # target_pos_b, target_quat_b = get_tf(
    #     "world", "fr3_link1", target_pos[torch.newaxis, :], target_quat[torch.newaxis, :]
    # )

    # curr_goal_idx = 0
    # # Create buffers to store actions
    # ik_commands = torch.zeros(scene.num_envs, ik_controller.action_dim, device=robot.device)
    # ik_commands[:] = eef_goals[curr_goal_idx]

    # # reset joint state
    # joint_pos = robot.data.default_joint_pos.clone()
    # joint_vel = robot.data.default_joint_vel.clone()
    # robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    # ik_controller.reset()
    # env.unwrapped.sim.step()

    eef_goals = [
        [0.5, 0.5, 0.7, 0.707, 0, 0.707, 0],
        [0.5, -0.4, 0.6, 0.707, 0.707, 0.0, 0.0],
        [0.5, 0, 0.5, 0.0, 1.0, 0.0, 0.0],
    ]

    # if not args_cli.headless:
    # time.sleep(10)

    # count = 0
    # # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():

            # robot.set_joint_velocity_target(vel_target)

            #         print(f"Running IK to pose for goal index: {curr_goal_idx}")
            #         print(ik_commands)

            #         mp.run_ik_to_pose(
            #             robot=robot,
            #             ik_controller=ik_controller,
            #             robot_entity_cfg=robot_entity_cfg,
            #             ee_jacobi_idx=ee_jacobi_idx,
            #             scene=scene,
            #             sim=env.unwrapped.sim,
            #             goal_pos_w=ik_commands[:, 0:3],
            #             goal_quat_w=ik_commands[:, 3:7],
            #             num_steps=1000,
            #             eef_marker=eef_marker,
            #             goal_marker=goal_marker,
            #         )

            # rmp_flow_controller.set_command(command=actions)
            # pos_target, vel_target = rmp_flow_controller.compute()
            # # print(f"RMP flow controller output: {res}")

            # robot.set_joint_position_target(pos_target)
            # robot.write_data_to_sim()
            # # env.unwrapped.sim.step()
            # robot.update(env.unwrapped.sim.get_physics_dt())

            print(robot.data.body_pos_w[:, robot.find_bodies("wrist_3_link")[0]])
            print(robot.data.body_quat_w[:, robot.find_bodies("wrist_3_link")[0]])

            actions = torch.cat([target_pos, target_quat], dim=-1)
            actions = torch.tensor([[0.0, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0]], device=env.unwrapped.device)
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
