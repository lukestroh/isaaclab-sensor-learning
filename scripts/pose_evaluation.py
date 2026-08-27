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
parser.add_argument(
    "--trial-data-name",
    type=str,
    default=None,
    help="Name of the trial data file.",
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
from isaaclab.sim.utils import stage as stage_utils
import isaaclab.utils.math as math_utils
from isaaclab_tasks.utils import parse_env_cfg

# from isaaclab_sensor_learning.motion_planning import diff_ik as diff_ik_mp
from isaaclab_sensor_learning import DATA_DIR
from isaaclab_sensor_learning.evaluation.evaluation import CoverageEvaluator, ObservationConfig, GainConfig
import isaaclab_sensor_learning.tasks  # noqa: F401
from isaaclab_sensor_learning.utils import h5py_utils, usd_utils

import data_utils.data_logger as data_logger
import data_utils.pose_generator as pose_generator
import pprint as pp
import os

import time
import h5py

from isaaclab.utils.seed import configure_seed


def get_envs_to_move_intersection(a, b):
    return a[torch.isin(a, b)]


def main():
    # trial data path
    trial_data_path = os.path.join(DATA_DIR, "capture", f"{str(args_cli.trial_data_name)}.h5")
    with h5py.File(trial_data_path, "r") as f:
        trial_data_dict = h5py_utils.h5_to_dict(f)
    print(trial_data_dict.keys())
    seed = trial_data_dict['attrs']['seed']
    configure_seed(int(seed))
    
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
    device = env.unwrapped.device

    # # load poses from file
    if args_cli.trial_data_name is None:
        raise ValueError("Please provide a trial data name using the --trial-data-name argument.")

    # Set up data logger
    # dlog = data_logger.DataLogger() # TODO: add args for evaluation vs. capture

    

    # Read tree ids, make sure they are the same as the trial data
    tree_prims = [prim for prim in stage_utils.get_current_stage().Traverse() if prim.GetName() == "tree"]
    tree_ids = {}
    for tree_prim in tree_prims:
        env_id = tree_prim.GetParent().GetName()
        
        # pp.pprint(dir(tree_prim))
        usd_path = tree_prim.GetMetadata("references").prependedItems[0].assetPath
        tree_str = os.path.basename(usd_path).split(".")[0].strip("_uv")

        trial_tree_attrs = trial_data_dict["trees"][env_id]['attrs']
        trial_tree_str = "_".join([trial_tree_attrs['tree_namespace'], trial_tree_attrs['tree_type'], trial_tree_attrs['tree_id']])

        assert tree_str == trial_tree_str, f"Tree id mismatch for {env_id}: {tree_str} vs {trial_tree_str}"
        # tree_namespace, tree_type, tree_id = tree_str.split("_")




    # Coverage evaluator
    observation_cfg = ObservationConfig()
    gain_cfg = GainConfig()
    coverage_evaluator = CoverageEvaluator(
        n_envs=num_envs,
        trial_data=trial_data_dict,
        observation_config=observation_cfg,
        gain_config=gain_cfg,
    )


    # Get the robot's end-effector pose in world frame
    robot_eef_w = robot.data.body_pose_w.torch[:, robot.find_bodies("ur10e__tool0")[0]].reshape(-1, 7)
    robot_base_w = robot.data.root_pose_w.torch

    # compute the next best pose to move to based on the current end-effector pose and the trial data
    closest_idxs = coverage_evaluator.get_closest_pose_idxs(curr_poses=robot_eef_w)
    best_poses = coverage_evaluator.compute_env_info_gain_from_pose_idxs(
        env_ids=torch.arange(num_envs), closest_idxs=closest_idxs, trial_data_dict=trial_data_dict
    )
    env_idxs = best_poses[:, 0]
    pose_idxs = best_poses[:, 1]

    goal_pose_w = coverage_evaluator.poses[env_idxs, pose_idxs].clone().detach()

    robot_eef_positions_b = torch.subtract(goal_pose_w[:, 0:3], robot_base_w[:, 0:3])
    positions_close = torch.isclose(robot_eef_positions_b[:, 0:3], goal_pose_w[:, 0:3], atol=0.002)
    # orientation
    quat_diff_err = math_utils.quat_error_magnitude(goal_pose_w[:, 3:7], robot_eef_w[:, 3:7])
    orientations_close = torch.isclose(quat_diff_err, torch.tensor(0.0), atol=0.006)
    # print(orientations_close)

    env_pos_close_change_goal = torch.where(positions_close.all(dim=1))[0]
    env_orient_close_change_goal = torch.where(orientations_close)[0]
    envs_change_goal = get_envs_to_move_intersection(env_pos_close_change_goal, env_orient_close_change_goal)

    # Get actions as size of all envs
    goal_pose = torch.cat(
        (torch.subtract(goal_pose_w[:, 0:3], robot_base_w[env_idxs, 0:3]), goal_pose_w[env_idxs, 3:7]), dim=1
    )
    actions = goal_pose.clone().detach()
    # print(actions)

    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            # check if the robot's end-effector is close to the goal pose
            robot_base_w = robot.data.root_pose_w.torch
            robot_eef_w = robot.data.body_pose_w.torch[:, robot.find_bodies("ur10e__tool0")[0]].reshape(-1, 7)

            goal_pose_b = torch.subtract(goal_pose_w[:, 0:3], robot_base_w[env_idxs, 0:3])

            # position
            robot_eef_positions_b = torch.subtract(robot_eef_w[:, 0:3], robot_base_w[:, 0:3])
            positions_close = torch.isclose(robot_eef_positions_b[:, 0:3], goal_pose_b[:, 0:3], atol=0.002)
            # orientation
            quat_diff_err = math_utils.quat_error_magnitude(actions[:, 3:7], robot_eef_w[:, 3:7])
            orientations_close = torch.isclose(quat_diff_err, torch.tensor(0.0), atol=0.006)

            env_pos_close_change_goal = torch.where(positions_close.all(dim=1))[0]
            env_orient_close_change_goal = torch.where(orientations_close)[0]
            envs_change_goal = get_envs_to_move_intersection(env_pos_close_change_goal, env_orient_close_change_goal)
            # print(f"\nenv_pos_close_change_goal: {env_pos_close_change_goal}")
            # # print(robot_eef_positions_b[:, 0:3])
            # # print(goal_pose_b[:, 0:3])
            # print(f"env_orient_close_change_goal: {env_orient_close_change_goal}")
            # print(f"envs_change_goal: {envs_change_goal}")

            if not envs_change_goal.numel() == 0:
                print(f"Envs to change goal: {envs_change_goal}")

                # compute the next best pose to move to based on the current end-effector pose and the trial data
                closest_idxs = coverage_evaluator.get_closest_pose_idxs(curr_poses=robot_eef_w)
                best_poses = coverage_evaluator.compute_env_info_gain_from_pose_idxs(
                    env_ids=envs_change_goal, closest_idxs=closest_idxs, trial_data_dict=trial_data_dict
                )
                print(f"Best poses:\n{best_poses}")
                # print(coverage_evaluator.poses[best_poses[:, 0], best_poses[:, 1]])
                for env_key in range(num_envs):
                    print(f"Env {env_key} bin counts: {torch.sum(coverage_evaluator.binned_tree.bin_counts[f'env_{env_key}'])}")
                env_idxs = best_poses[:, 0]
                pose_idxs = best_poses[:, 1]

                goal_pose_w = coverage_evaluator.poses[env_idxs, pose_idxs].clone().detach()
                goal_pose_b = torch.subtract(goal_pose_w[:, 0:3], robot_base_w[env_idxs, 0:3])

                goal_pose = torch.cat((goal_pose_b, goal_pose_w[env_idxs, 3:7]), dim=1)

                # print(goal_pose.shape)
                # print(envs_change_goal)
                # print(actions)
                # print(goal_pose)

                actions[envs_change_goal] = goal_pose
                # print(actions)

                # import sys

                # sys.exit(0)


            observations, rewards, terminated, truncated, info = env.step(actions)
            info_gains, dense_bin_updates, sparse_bin_updates = (
                coverage_evaluator.update_bin_count_scores_from_observation(
                    env_ids=torch.arange(num_envs), observations=observations
                )
            )

            # for env_key in range(num_envs):
            #     print(env_key, torch.sum(coverage_evaluator.binned_tree.bin_counts[f"env_{env_key}"]))
            continue
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
