#!/usr/bin/env python3

from isaaclab.app import AppLauncher
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="Pose evaluation script for pose data capture")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Template-Sensor-Learning-Direct-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=0, help="Seed argument for deterministic runs.")
parser.add_argument(
    "--allow-root", action="store_true", default=True, help="Allow running as root user (not recommended)."
)
parser.add_argument("--rig_yaml_path", type=str, default="/home/luke/IsaacLab/source/isaaclab-sensor-learning/data/rigs/test_rig0.yaml", help="Path to the rig yaml file.")
parser.add_argument(
    "--robot",
    type=str,
    default="ur10e",
    help="Name of the robot to use in the evaluation.",
    choices=["fr3", "ur10e", "panda"],
)


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import data_utils.capture_data_logger as cdata_logger
import data_utils.evaluation_data_logger as edata_logger
import gymnasium as gym
import h5py
import isaaclab.sim as sim_utils
import isaaclab.sim.views as sim_views
import isaaclab.utils.math as math_utils
import isaaclab_sensor_learning.tasks  # noqa: F401
import numpy as np
import pprint as pp
import os
import re
import time
import torch
import utils.pose_generator as pose_generator
import utils.trial_setup as trial_setup


from isaaclab.sim.utils import stage as stage_utils
from isaaclab.sim.utils import semantics as semantic_utils
from pxr import Usd, UsdGeom

from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene
from isaaclab.utils.seed import configure_seed

from isaaclab_sensor_learning import DATA_DIR
from isaaclab_sensor_learning.nbv.nbv import NBVEvaluator, ObservationConfig, GainConfig
from isaaclab_sensor_learning.utils import h5py_utils
from isaaclab_tasks.utils import parse_env_cfg




def main():
    configure_seed(seed=args_cli.seed)

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, rig_yaml_path=args_cli.rig_yaml_path)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # reset environment
    env.reset()
    robot: Articulation = env.unwrapped.robot
    sensors: dict = env.unwrapped.sensors
    scene: InteractiveScene = env.unwrapped.scene
    sim: sim_utils.SimulationContext = env.unwrapped.sim
    cuda_device = env.unwrapped.device
    n_envs = env.unwrapped.num_envs
    physics_dt = env.unwrapped.physics_dt
    decimation = env.unwrapped.cfg.decimation

    discrete_poses, trial_metadata = trial_setup.default_discrete_poses(seed=args_cli.seed, n_envs=n_envs, robot=robot)
    tree_metadata = trial_setup.get_tree_metadata(env=env)
    sensor_metadata = trial_setup.get_sensor_metadata(sensors=sensors)

    # Set up capture data logger
    cdlog = cdata_logger.CaptureDataLogger()
    cdlog.save_trial_metadata(trial_metadata=trial_metadata)
    cdlog.save_tree_metadata(tree_metadata=tree_metadata)
    cdlog.save_sensor_metadata(sensor_metadata=sensor_metadata, n_poses=discrete_poses.shape[0])

    # simulate environment
    pose_idx = 0
    step_count = 0
    step_count_limit = 500
    max_steps = 0
    avg_step_count = 0
    total_steps = 0
    skipped_poses = []
    last_obs = False

    # Data collection loop
    while simulation_app.is_running():
        # check if poses completed. save any skipped poses
        if last_obs or pose_idx >= len(discrete_poses):
            print("\n[INFO]: All poses completed, exiting simulation.")
            cdlog.save_skipped_poses(skipped_poses=skipped_poses)
            cdlog.save_labels_dict()
            break
        print(f"\rpose iter: {pose_idx+1}/{len(discrete_poses)}", end="")
        # run everything in inference mode
        with torch.inference_mode():
            goal_pose = torch.tensor(
                discrete_poses[pose_idx][np.newaxis, :], dtype=torch.float32, device=cuda_device
            )

            # Actions
            actions = torch.tensor(goal_pose.detach().clone(), dtype=torch.float32, device=cuda_device).repeat(
                n_envs, 1
            )
            # step
            observations, rewards, terminated, truncated, info = env.step(actions)
            step_count += 1

            # check if the robot's end-effector is close to the goal pose
            robot_base_w = robot.data.root_pose_w.torch
            robot_eef_w = robot.data.body_pose_w.torch[:, robot.find_bodies("ur10e__tool0")[0]].reshape(-1, 7)
            # position
            robot_eef_pos_b = torch.subtract(robot_eef_w[:, 0:3], robot_base_w[:, 0:3])
            positions_close = torch.isclose(robot_eef_pos_b[:, 0:3], goal_pose[:, 0:3], atol=0.002)
            # orientation
            quat_diff_err = math_utils.quat_error_magnitude(actions[:, 3:7], robot_eef_w[:, 3:7])
            orientations_close = torch.isclose(quat_diff_err, torch.tensor(0.0), atol=0.006)
            # joint states
            robot_joint_pos = robot.data.joint_pos.torch

            # print("\n", goal_pose[:, 0:3] - robot_eef_pos_b, quat_diff_err)

            if step_count > max_steps:
                max_steps = step_count
            
            # save observations and poses if the robot's end-effector is close to the goal pose
            if positions_close.all() and orientations_close.all():
                if pose_idx >= len(discrete_poses):
                    cdlog.save_pose_observations(observations=observations, idx=pose_idx, last_obs=True)
                    cdlog.save_actual_poses(poses=robot_eef_w, idx=pose_idx, last_obs=True)
                    cdlog.save_joint_state(joint_state=robot_joint_pos, idx=pose_idx, last_obs=True)
                    last_obs = True
                    continue
                else:
                    cdlog.save_pose_observations(observations=observations, idx=pose_idx)
                    cdlog.save_actual_poses(poses=robot_eef_w, idx=pose_idx)
                    cdlog.save_joint_state(joint_state=robot_joint_pos, idx=pose_idx)
                # break  # remove after testing

                print(f"  --  avg steps: {avg_step_count:.2f} -- max steps: {max_steps}", end="")
                joint_pos = robot.data.default_joint_pos.clone()
                joint_vel = robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel)
                robot.reset()
                total_steps += step_count
                avg_step_count = total_steps / (pose_idx + 1)
                step_count = 0
                pose_idx += 1
            else:
                if step_count > step_count_limit:
                    print(
                        f"\n[WARNING]: Step count exceeded {step_count_limit} for pose {pose_idx}, continuing to next pose."
                    )
                    cdlog.save_pose_observations(
                        observations=observations, idx=pose_idx
                    )  # save observations even if not reached goal. change later?
                    cdlog.save_actual_poses(poses=robot_eef_w, idx=pose_idx)
                    cdlog.save_joint_state(joint_state=robot_joint_pos, idx=pose_idx)
                    skipped_poses.append(pose_idx)
                    total_steps += step_count
                    step_count = 0
                    pose_idx += 1
                    continue
    # end of data collection loop
    print(f"\n[INFO]: Pose evaluation completed. Total poses: {len(discrete_poses)}, Skipped poses: {len(skipped_poses)}")
    print(f"[INFO]: Proceeding to NBV evaluation loop.")


    # reset environment
    env.reset()

    # trial data path
    trial_data_path = os.path.join(DATA_DIR, "capture", f"SL_PDC_{trial_metadata['datetime_id']}.h5")
    if not os.path.exists(trial_data_path):
        raise FileNotFoundError
    with h5py.File(trial_data_path, "r") as f:
        trial_data_dict = h5py_utils.h5_to_dict(f)

    # Set up nbv evaluation data logger
    capture_data_dict = {"n_envs": n_envs, "datetime_id": trial_metadata["datetime_id"]}
    evaluation_metadata = {
        "seed": args_cli.seed,
        "datetime_id": trial_metadata["datetime_id"],
        "associated_data": f"SL_PDC_{trial_metadata['datetime_id']}",
    }
    timestamp_metadata = {"physics_dt": physics_dt, "decimation": decimation}
    evlog = edata_logger.EvaluationDataLogger(capture_data_dict=capture_data_dict)
    evlog.save_evaluation_metadata(evaluation_metadata=evaluation_metadata)
    evlog.save_timestamp_metadata(timestamp_metadata=timestamp_metadata)
    evlog.save_joint_position_metadata(joint_position_metadata={"n_dof": robot.num_joints})
    evlog.save_nbv_goal_metadata()
    evlog.save_sparse_bin_count_metadata()
    evlog.save_pose_metadata()

    # Coverage evaluator
    observation_cfg = ObservationConfig()
    gain_cfg = GainConfig()
    nbv_evaluator = NBVEvaluator(
        n_envs=n_envs,
        trial_data=trial_data_dict,
        observation_config=observation_cfg,
        gain_config=gain_cfg,
    )

    # NBV sim setup
    # Reset robot to default joint positions and velocities
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel)
    robot.reset()
    # Get the robot's end-effector pose in world frame
    robot_eef_w = robot.data.body_pose_w.torch[:, robot.find_bodies("ur10e__tool0")[0]].reshape(-1, 7)
    robot_base_w = robot.data.root_pose_w.torch

    # compute the next best pose to move to based on the current end-effector pose and the trial data
    closest_idxs = nbv_evaluator.get_closest_pose_idxs(curr_poses=robot_eef_w)
    best_poses = nbv_evaluator.compute_env_info_gain_from_pose_idxs(
        env_ids=torch.arange(n_envs), closest_idxs=closest_idxs, trial_data_dict=trial_data_dict
    )
    current_poses = best_poses.clone().detach()
    env_idxs = best_poses[:, 0]
    pose_idxs = best_poses[:, 1]

    goal_pose_w = nbv_evaluator.poses[env_idxs, pose_idxs].clone().detach()
    goal_position_b = torch.subtract(goal_pose_w[env_idxs, 0:3], robot_base_w[env_idxs, 0:3])

    actions = torch.cat((goal_position_b, goal_pose_w[env_idxs, 3:7]), dim=1)
    curr_time = 0
    for env_id in env_idxs:
        evlog.save_nbv_goal(env_id=env_id.item(), timestamp=curr_time, nbv_goal=actions[env_id])

    seen_pose_idxs = torch.empty(size=(0, 2), dtype=torch.int64, device=cuda_device)

    last_obs = False

    def get_envs_to_move_intersection(a, b):
        return a[torch.isin(a, b)]

    # NBV loop
    while simulation_app.is_running():
        with torch.inference_mode():
            # Log all timestamps
            current_step = env.unwrapped.common_step_counter
            curr_time = physics_dt * decimation * current_step
            evlog.save_timestamp(timestamp=curr_time, last_obs=last_obs)

            # Get observations and save the sparse bin updates
            observations, rewards, terminated, truncated, info = env.step(actions)
            info_gains, dense_bin_updates, sparse_bin_updates = nbv_evaluator.update_bin_count_scores_from_observation(
                env_ids=torch.arange(n_envs), observations=observations
            )
            evlog.save_sparse_bin_counts(timestamp=curr_time, sparse_bin_updates=sparse_bin_updates, last_obs=last_obs)

            # #################################################################
            # check if the robot's end-effector is close to the goal pose
            # #################################################################
            robot_base_w = robot.data.root_pose_w.torch
            robot_eef_w = robot.data.body_pose_w.torch[:, robot.find_bodies("ur10e__tool0")[0]].reshape(-1, 7)

            # Save eef poses in base frame
            robot_position_b = robot_eef_w[:, 0:3] - robot_base_w[:, 0:3]
            robot_eef_b = torch.cat([robot_position_b, robot_eef_w[:, 3:7]], axis=1)
            evlog.save_poses(poses=robot_eef_b, last_obs=last_obs)

            # Save joint positions
            robot_joint_pos = robot.data.joint_pos.torch
            evlog.save_joint_positions(joint_positions=robot_joint_pos, last_obs=last_obs)

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

            if envs_change_goal.numel() != 0:
                # Keep a list of visited poses by idx
                # a[~a.unsqueeze(1).eq(b).all(-1).any(-1)] # https://stackoverflow.com/questions/62588779/pytorch-differences-between-two-tensors
                unseen_current_poses = current_poses[~current_poses.unsqueeze(1).eq(seen_pose_idxs).all(-1).any(-1)]
                if torch.any(unseen_current_poses):
                    seen_pose_idxs = torch.cat([seen_pose_idxs, unseen_current_poses], dim=0)

                # Compute the next best pose to move to based on the current end-effector pose and the trial data.
                # Remove seen poses so we don't get redundant locations.
                # Update actions
                closest_idxs = nbv_evaluator.get_closest_pose_idxs(curr_poses=robot_eef_w)
                unseen_closest_idxs = closest_idxs[~closest_idxs.unsqueeze(1).eq(seen_pose_idxs).all(-1).any(-1)]
                if unseen_closest_idxs.numel() == 0:
                    raise ValueError("no unseen_closes_idxs")
                best_pose_idxs = nbv_evaluator.compute_env_info_gain_from_pose_idxs(
                    env_ids=envs_change_goal, closest_idxs=unseen_closest_idxs, trial_data_dict=trial_data_dict
                )
                if best_pose_idxs.numel() == 0:
                    print(unseen_closest_idxs)
                    raise ValueError("no best poses")
                env_idxs = best_pose_idxs[:, 0]
                pose_idxs = best_pose_idxs[:, 1]
                goal_pose_w = nbv_evaluator.poses[env_idxs, pose_idxs].clone().detach()
                goal_position_b = torch.subtract(goal_pose_w[:, 0:3], robot_base_w[env_idxs, 0:3])
                goal_pose_b = torch.cat((goal_position_b, goal_pose_w[:, 3:7]), dim=1)
                actions[env_idxs] = goal_pose_b

                # Save nbv goal on update
                for env_id in env_idxs:
                    evlog.save_nbv_goal(env_id=env_id.item(), timestamp=curr_time, nbv_goal=actions[env_id])

                # update the current pose to the new best pose
                current_poses[env_idxs] = best_pose_idxs

            if last_obs:
                evlog.save_bin_counts(bin_counts=nbv_evaluator.binned_tree.bin_counts)
                break

            for env_key, env_bin_counts in nbv_evaluator.binned_tree.bin_counts.items():
                more_than_5_counts_mask = torch.nonzero(env_bin_counts >= 5, as_tuple=False)
                num_completed_bins = torch.numel(more_than_5_counts_mask)
                env_total_bins = np.prod(tuple(env_bin_counts.shape))

                seen_bins = num_completed_bins / env_total_bins

                print(current_step, ": ", env_key, seen_bins)

            # Terminating condition
            if current_step == 10_000:
                last_obs = True
    

    env.close()

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


