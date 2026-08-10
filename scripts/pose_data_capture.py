#!/usr/bin/env python3
from isaaclab.app import AppLauncher
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="Pose data capture script")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Template-Pose-Data-Capture-Direct-v0", help="Name of the task.")
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

import data_utils.data_logger as data_logger
import data_utils.pose_generator as pose_generator
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

# from isaaclab_sensor_learning.utils import logging_conf


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()

    robot: Articulation = env.unwrapped.robot
    sensors: dict = env.unwrapped.sensors
    scene: InteractiveScene = env.unwrapped.scene
    sim: sim_utils.SimulationContext = env.unwrapped.sim
    cuda_device = env.unwrapped.device

    # Set up data logger
    dlog = data_logger.DataLogger()
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

    # trial metadata
    trial_metadata = {
        "trial_name": dlog.trial_name,
        "num_envs": env.unwrapped.cfg.num_envs,
        "r_range": r_range,
        "r_size": r_size,
        "theta_cyl_range": theta_cyl_range,
        "theta_cyl_size": theta_cyl_size,
        "z_range": z_range,
        "z_size": z_size,
        "theta_ori_range": theta_ori_range,
        "theta_ori_size": theta_ori_size,
        "phi_ori_range": phi_ori_range,
        "phi_ori_size": phi_ori_size,
        "poses": discrete_poses,
        "robot_joint_names": robot.data.joint_names,
        "robot_body_names": robot.data.body_names,
    }
    # tree metadata
    tree_metadata = {}
    tree_prims = [prim for prim in stage_utils.get_current_stage().Traverse() if prim.GetName() == "tree"]
    for tree_prim in tree_prims:
        # print(pp.pformat(dir(tree_prim)))
        prim_path = tree_prim.GetPath()
        match = re.search(r"env_(\d+)", str(prim_path))
        if match:
            env_idx = int(match.group(1))
        usd_path = tree_prim.GetMetadata("references").prependedItems[0].assetPath
        # tree_pose = tree_prim.GetAttribute("xformOp:transform").Get()
        xformable = UsdGeom.Xformable(tree_prim)
        tree_pose = xformable.GetLocalTransformation()
        tree_pose_np = np.array(tree_pose).reshape(4, 4).T
        tree_pos, tree_rot_mat = math_utils.unmake_pose(torch.tensor(tree_pose_np, device=env.unwrapped.device))
        tree_str = os.path.basename(usd_path).split(".")[0].strip("_uv")
        tree_namespace, tree_type, tree_id = tree_str.split("_")
        tree_metadata[env_idx] = {
            "tree_usd_path": usd_path,
            "prim_path": str(prim_path),
            "env_idx": env_idx,
            "tree_namespace": tree_namespace,
            "tree_type": tree_type,
            "tree_id": tree_id,
            "position": tree_pos.detach().cpu().numpy(),
            "orientation": tree_rot_mat.detach().cpu().numpy(),
        }

    # sensors
    sensor_metadata = {}
    for sensor_name, sensor in sensors.items():
        # print(sensor.cfg)
        # print(sensor.cfg.update_period)
        # pp.pprint(sensor.cfg.asdict())
        sensor_metadata[sensor_name] = {
            "class_type": str(sensor.cfg.class_type),
            "width": sensor.cfg.width,
            "height": sensor.cfg.height,
            "data_rate_hz": 1 / sensor.cfg.update_period,
            "receiver_offset": np.concatenate([sensor.cfg.offset.pos, sensor.cfg.offset.rot]),
            "depth_clipping_behavior": sensor.cfg.depth_clipping_behavior,
            "data_types": sensor.cfg.data_types,
            "z_near": sensor.cfg.spawn.clipping_range[0],
            "z_far": sensor.cfg.spawn.clipping_range[1],
            "focal_length": sensor.cfg.spawn.focal_length,
            "focus_distance": sensor.cfg.spawn.focus_distance,
            "f_stop": sensor.cfg.spawn.f_stop,
            "horizontal_aperture": sensor.cfg.spawn.horizontal_aperture,
            "vertical_aperture": sensor.cfg.spawn.vertical_aperture,
        }
    dlog.save_trial_metadata(trial_metadata=trial_metadata)
    dlog.save_tree_metadata(tree_metadata=tree_metadata)
    dlog.save_sensor_metadata(
        sensor_metadata=sensor_metadata, n_poses=discrete_poses.shape[0], num_envs=env.unwrapped.cfg.num_envs
    )

    # simulate environment
    pose_idx = 0
    step_count = 0
    step_count_limit = 500
    max_steps = 0
    avg_step_count = 0
    total_steps = 0
    skipped_poses = []
    # avg_step_time = 0.0

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel)
    sim.step()

    # robot.reset()
    while simulation_app.is_running():

        # check if poses completed. save any skipped poses
        if pose_idx >= len(discrete_poses):
            print("\n[INFO]: All poses completed, exiting simulation.")
            dlog.save_skipped_poses(skipped_poses=skipped_poses)
            dlog.save_labels_dict()
            break

        print(f"\rpose iter: {pose_idx+1}/{len(discrete_poses)}", end="")

        # run everything in inference mode
        with torch.inference_mode():
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
            robot_eef_w = robot.data.body_pose_w.torch[:, robot.find_bodies("ur10e__tool0")[0]].reshape(-1, 7)
            # position
            robot_eef_pos_b = torch.subtract(robot_eef_w[:, 0:3], robot_base_w[:, 0:3])
            positions_close = torch.isclose(robot_eef_pos_b[:, 0:3], goal_pose[:, 0:3], atol=0.002)
            # orientation
            quat_diff_err = math_utils.quat_error_magnitude(actions[:, 3:7], robot_eef_w[:, 3:7])
            orientations_close = torch.isclose(quat_diff_err, torch.tensor(0.0), atol=0.006)

            for sensor_name, sensor in sensors.items():
                print(sensor.data.pos_w.torch)

            # print(positions_close, orientations_close, step_count, end="\r")
            # print(goal_pose[:, 0:3])
            # print(robot_eef_pos_b[:, 0:3])
            if step_count > max_steps:
                max_steps = step_count

            # save observations and poses if the robot's end-effector is close to the goal pose
            if positions_close.all() and orientations_close.all():
                print("HELLOW ROLD", observations['tof0'].output['normals'].torch)

                if pose_idx >= len(discrete_poses):
                    dlog.save_pose_observations(
                        observations=observations, poses=robot_eef_w, idx=pose_idx, last_obs=True
                    )
                    break
                else:
                    dlog.save_pose_observations(observations=observations, poses=robot_eef_w, idx=pose_idx)
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
                    dlog.save_pose_observations(
                        observations=observations, poses=robot_eef_w, idx=pose_idx
                    )  # save observations even if not reached goal. change later?
                    skipped_poses.append(pose_idx)
                    total_steps += step_count
                    step_count = 0
                    pose_idx += 1
                    continue

    # close the simulator
    env.close()
    # print("\n[INFO]: Simulation finished, data saved to: ", datafile_path)
    return


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception as e:
        # print(f"[ERROR]: {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()
