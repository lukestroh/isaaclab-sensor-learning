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
import h5py
import os
import numpy as np
import time
import torch

from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene
import isaaclab.sim as sim_utils
from isaaclab.sim.utils import stage as stage_utils
from isaaclab.utils.seed import configure_seed
import isaaclab.utils.math as math_utils
from isaaclab_tasks.utils import parse_env_cfg

# from isaaclab_sensor_learning.motion_planning import diff_ik as diff_ik_mp
from isaaclab_sensor_learning import DATA_DIR
from isaaclab_sensor_learning.evaluation.nbv import NBVEvaluator, ObservationConfig, GainConfig
import isaaclab_sensor_learning.tasks  # noqa: F401
from isaaclab_sensor_learning.utils import h5py_utils, usd_utils

import data_utils.evaluation_data_logger as data_logger
import data_utils.pose_generator as pose_generator

import pprint as pp


def get_envs_to_move_intersection(a, b):
    return a[torch.isin(a, b)]


def main():
    # trial data path
    trial_data_path = os.path.join(DATA_DIR, "capture", f"{str(args_cli.trial_data_name)}.h5")
    if not os.path.exists(trial_data_path):
        raise FileNotFoundError
    with h5py.File(trial_data_path, "r") as f:
        trial_data_dict = h5py_utils.h5_to_dict(f)

    # set seed
    seed = trial_data_dict["attrs"]["seed"]
    configure_seed(int(seed))

    # Create environment
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    robot: Articulation = env.unwrapped.robot
    num_envs = env.unwrapped.cfg.num_envs
    device = env.unwrapped.device
    physics_dt = env.unwrapped.physics_dt
    decimation = env.unwrapped.cfg.decimation

    # Get datetime as ID value connecting the capture and evaluation portions together
    capture_data_name = trial_data_dict["attrs"]["trial_name"]
    capture_data_split = capture_data_name.split("_")
    capture_data_dt = "_".join([capture_data_split[-2], capture_data_split[-1]])
    capture_data_dict = {
        "datetime_id": capture_data_dt,
        "n_envs": num_envs,
    }

    # data logger
    dlog = data_logger.EvaluationDataLogger(capture_data_dict=capture_data_dict)
    evaluation_metadata = {
        "seed": seed,
        "trial_name": dlog.trial_name,
        "associated_data": str(args_cli.trial_data_name),
    }
    timestamp_metadata = {"physics_dt": physics_dt, "decimation": decimation}
    dlog.save_evaluation_metadata(evaluation_metadata=evaluation_metadata)
    dlog.save_timestamp_metadata(timestamp_metadata=timestamp_metadata)
    dlog.save_joint_position_metadata(joint_position_metadata={"n_dof": robot.num_joints})
    dlog.save_nbv_goal_metadata()
    dlog.save_sparse_bin_count_metadata()
    dlog.save_pose_metadata()
    # import sys; sys.exit(0)

    # reset environment
    env.reset()

    # # load poses from file
    if args_cli.trial_data_name is None:
        raise ValueError("Please provide a trial data name using the --trial-data-name argument.")

    # Read tree ids, make sure they are the same as the trial data
    tree_prims = [prim for prim in stage_utils.get_current_stage().Traverse() if prim.GetName() == "tree"]
    for tree_prim in tree_prims:
        env_id = tree_prim.GetParent().GetName()
        # pp.pprint(dir(tree_prim))
        usd_path = tree_prim.GetMetadata("references").prependedItems[0].assetPath
        tree_str = os.path.basename(usd_path).split(".")[0].strip("_uv")

        trial_tree_attrs = trial_data_dict["trees"][env_id]["attrs"]
        trial_tree_str = "_".join(
            [trial_tree_attrs["tree_namespace"], trial_tree_attrs["tree_type"], trial_tree_attrs["tree_id"]]
        )
        assert tree_str == trial_tree_str, f"Tree id mismatch for {env_id}: {tree_str} vs {trial_tree_str}"

    # Coverage evaluator
    observation_cfg = ObservationConfig()
    gain_cfg = GainConfig()
    coverage_evaluator = NBVEvaluator(
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
    current_poses = best_poses.clone().detach()
    env_idxs = best_poses[:, 0]
    pose_idxs = best_poses[:, 1]

    goal_pose_w = coverage_evaluator.poses[env_idxs, pose_idxs].clone().detach()
    goal_position_b = torch.subtract(goal_pose_w[env_idxs, 0:3], robot_base_w[env_idxs, 0:3])

    actions = torch.cat((goal_position_b, goal_pose_w[env_idxs, 3:7]), dim=1)
    curr_time = 0
    for env_id in env_idxs:
        dlog.save_nbv_goal(env_id=env_id.item(), timestamp=curr_time, nbv_goal=actions[env_id])

    seen_pose_idxs = torch.empty(size=(0, 2), dtype=torch.int64, device=device)

    last_obs = False

    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            # Log all timestamps
            current_step = env.unwrapped.common_step_counter
            curr_time = physics_dt * decimation * current_step
            dlog.save_timestamp(timestamp=curr_time, last_obs=last_obs)

            # Get observations and save the sparse bin updates
            observations, rewards, terminated, truncated, info = env.step(actions)
            info_gains, dense_bin_updates, sparse_bin_updates = (
                coverage_evaluator.update_bin_count_scores_from_observation(
                    env_ids=torch.arange(num_envs), observations=observations
                )
            )
            dlog.save_sparse_bin_counts(timestamp=curr_time, sparse_bin_updates=sparse_bin_updates, last_obs=last_obs)

            # #################################################################
            # check if the robot's end-effector is close to the goal pose
            # #################################################################
            robot_base_w = robot.data.root_pose_w.torch
            robot_eef_w = robot.data.body_pose_w.torch[:, robot.find_bodies("ur10e__tool0")[0]].reshape(-1, 7)

            # Save eef poses in base frame
            robot_position_b = robot_eef_w[:, 0:3] - robot_base_w[:, 0:3]
            robot_eef_b = torch.cat([robot_position_b, robot_eef_w[:, 3:7]], axis=1)
            dlog.save_poses(poses=robot_eef_b, last_obs=last_obs)

            # Save joint positions
            robot_joint_pos = robot.data.joint_pos.torch
            dlog.save_joint_positions(joint_positions=robot_joint_pos, last_obs=last_obs)

            goal_position_w = actions.clone().detach()[:, 0:3] + robot_base_w[:, 0:3]
            # position
            positions_close = torch.isclose(goal_position_w[:, 0:3], robot_eef_w[:, 0:3], rtol=0.0, atol=0.003)
            # orientation
            quat_diff_err = math_utils.quat_error_magnitude(actions[:, 3:7], robot_eef_w[:, 3:7])
            orientations_close = torch.isclose(quat_diff_err, torch.tensor(0.0), rtol=0.0, atol=0.0065)
            # find which envs need a new goal
            env_pos_close_change_goal = torch.where(positions_close.all(dim=1))[0]
            env_orient_close_change_goal = torch.where(orientations_close)[0]
            envs_change_goal = get_envs_to_move_intersection(env_pos_close_change_goal, env_orient_close_change_goal)
            # print("\n------------------------------------------------------------------------\n")
            # print("p_close:\n ", positions_close)
            # print("o_close:\n ", orientations_close)
            # print(f"actions\n{actions}")
            # robot_eef_b = robot_eef_w[:, 0:3] - robot_base_w[:, 0:3]
            # goal_diff = robot_eef_b - actions[:, 0:3]
            # print("robot_eef_b\n", robot_eef_b)
            # print("goal_diff\n", goal_diff)
            # print("envs change goal\n", envs_change_goal)

            if envs_change_goal.numel() != 0:
                # Keep a list of visited poses by idx
                # a[~a.unsqueeze(1).eq(b).all(-1).any(-1)] # https://stackoverflow.com/questions/62588779/pytorch-differences-between-two-tensors
                unseen_current_poses = current_poses[~current_poses.unsqueeze(1).eq(seen_pose_idxs).all(-1).any(-1)]
                if torch.any(unseen_current_poses):
                    seen_pose_idxs = torch.cat([seen_pose_idxs, unseen_current_poses], dim=0)

                # Compute the next best pose to move to based on the current end-effector pose and the trial data.
                # Remove seen poses so we don't get redundant locations.
                # Update actions
                closest_idxs = coverage_evaluator.get_closest_pose_idxs(curr_poses=robot_eef_w)
                unseen_closest_idxs = closest_idxs[~closest_idxs.unsqueeze(1).eq(seen_pose_idxs).all(-1).any(-1)]
                if unseen_closest_idxs.numel() == 0:
                    raise ValueError("no unseen_closes_idxs")
                best_pose_idxs = coverage_evaluator.compute_env_info_gain_from_pose_idxs(
                    env_ids=envs_change_goal, closest_idxs=unseen_closest_idxs, trial_data_dict=trial_data_dict
                )
                if best_pose_idxs.numel() == 0:
                    print(unseen_closest_idxs)
                    raise ValueError("no best poses")
                env_idxs = best_pose_idxs[:, 0]
                pose_idxs = best_pose_idxs[:, 1]
                goal_pose_w = coverage_evaluator.poses[env_idxs, pose_idxs].clone().detach()
                goal_position_b = torch.subtract(goal_pose_w[:, 0:3], robot_base_w[env_idxs, 0:3])
                goal_pose_b = torch.cat((goal_position_b, goal_pose_w[:, 3:7]), dim=1)
                actions[env_idxs] = goal_pose_b

                # Save nbv goal on update
                for env_id in env_idxs:
                    dlog.save_nbv_goal(env_id=env_id.item(), timestamp=curr_time, nbv_goal=actions[env_id])

                # update the current pose to the new best pose
                current_poses[env_idxs] = best_pose_idxs

            if last_obs:
                break

            for env_key, env_bin_counts in coverage_evaluator.binned_tree.bin_counts.items():
                more_than_5_counts_mask = torch.nonzero(env_bin_counts >= 5, as_tuple=False)
                num_completed_bins = torch.numel(more_than_5_counts_mask)
                env_total_bins = np.prod(tuple(env_bin_counts.shape))

                seen_bins = num_completed_bins / env_total_bins

                print(current_step, ": ", env_key, seen_bins)

            # Terminating condition
            if current_step == 10_000:
                last_obs = True

    print("[INFO]: Simulation completed")
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
