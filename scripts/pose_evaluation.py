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
from isaaclab_mimic.motion_planners.curobo import curobo_planner as crp
from isaaclab_tasks.utils import parse_env_cfg

from pose_data_capture.motion_planning import diff_ik as diff_ik_mp
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

    planner_cfg = crp.CuroboPlannerCfg.from_task_name(args_cli.task)


    planner = crp.CuroboPlanner(
        env=env,
        robot=env.unwrapped.robot,
        config=planner_cfg,
    )


    # curr_goal_idx = 0
    # # Create buffers to store actions
    # ik_commands = torch.zeros(scene.num_envs, ik_controller.action_dim, device=robot.device)
    # ik_commands[:] = eef_goals[curr_goal_idx]

    # # reset joint state
    # joint_pos = robot.data.default_joint_pos.clone()
    # joint_vel = robot.data.default_joint_vel.clone()
    # robot.write_joint_state_to_sim(joint_pos, joint_vel)
    # robot.reset()
    # ik_controller.reset()
    # env.unwrapped.sim.step()

    # count = 0
    # # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
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
