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
from isaaclab.sim.utils import stage as stage_utils

from isaaclab.assets import Articulation

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
    print(robot.data.body_names)

    # Set up data logger
    dlog = data_logger.DataLogger()

    r_range = (0.8, 1.4)
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
    # discrete_poses = discrete_poses[1687:]

    # print(discrete_poses)

    # # a test pose for debugging
    # discrete_poses = np.array([[0.5, 0.5, 0.7, 0.707, 0, 0.707, 0]])
    # print(f"JOINT_NAMES: {robot.data.joint_names}")
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
        tree_str = os.path.basename(usd_path).split(".")[0].strip("_uv")
        tree_namespace, tree_type, tree_id = tree_str.split("_")
        tree_metadata[env_idx] = {
            "tree_usd_path": usd_path,
            "prim_path": str(prim_path),
            "env_idx": env_idx,
            "tree_namespace": tree_namespace,
            "tree_type": tree_type,
            "tree_id": tree_id,
            "pose": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }
    # pp.pprint(tree_metadata)

    # pp.pprint(env.unwrapped.scene.sensors)

    # print(f"DISCRETE POSES: {discrete_poses.shape}")

    # # print all robot prims in the stage
    for prim in stage_utils.get_current_stage().Traverse():
        if "robot" in str(prim.GetPath()):
            print(prim.GetPath())

    # from pxr import UsdPhysics, UsdGeom
    # import omni

    # stage = omni.usd.get_context().get_stage()

    # for prim in stage.Traverse():
    #     if UsdPhysics.ArticulationRootAPI.Get(stage, prim.GetPath()):
    #         print(prim.GetPath())

    # for path in [
    #     "/World/envs/env_0/robot",
    #     "/World/envs/env_0/robot/Geometry",
    #     "/World/envs/env_0/robot/Geometry/amiga__base",
    # ]:
    #     prim = stage.GetPrimAtPath(path)
    #     xf = UsdGeom.Xformable(prim)
    #     print(path)
    #     print(xf.GetLocalTransformation())

    from pxr import UsdGeom, Usd, UsdPhysics
    stage = env.unwrapped.sim.stage
    root = stage.GetPrimAtPath("/World/envs/env_0/robot")

    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            print(prim.GetPath())


    # sensors
    sensor_metadata = {}
    for sensor_name, sensor in env.unwrapped.sensors.items():
        # pp.pprint(sensor.cfg)
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
    max_steps = 0
    avg_step_count = 0
    # avg_step_time = 0.0
    while simulation_app.is_running():
        # print(f"\rpose iter: {pose_idx+1}/{len(discrete_poses)}", end="")
        # run everything in inference mode
        with torch.inference_mode():
            goal_pose = torch.tensor(
                discrete_poses[pose_idx][np.newaxis, :], dtype=torch.float32, device=env.unwrapped.device
            )
            # Actions
            actions = torch.tensor(goal_pose.detach().clone(), dtype=torch.float32, device=env.unwrapped.device).repeat(
                env.unwrapped.num_envs, 1
            )

            observations, rewards, terminated, truncated, info = env.step(actions)
            step_count += 1
            # # print(observations['tof0'].output['rgb'].shape)
            # print("Goal: ", torch.tensor(discrete_poses[pose_idx][0:3][np.newaxis, :], dtype=torch.float32, device=env.unwrapped.device))
            # print("Curr: ", robot.data.body_pos_w[:, robot.find_bodies("ur5e__wrist_3_link")[0]][0])
            robot_base_w = robot.data.root_pose_w.torch
            robot_eef_w = robot.data.body_pose_w.torch[:, robot.find_bodies("ur10e__wrist_3_link")[0]].reshape(-1, 7)
            # check if eef is close to goal pose
            # position
            robot_positions_b = torch.subtract(robot_eef_w[:, 0:3], robot_base_w[:, 0:3])
            positions_close = torch.isclose(robot_positions_b[:, 0:3], goal_pose[:, 0:3], atol=0.002)
            # orientation
            quat_diff_err = math_utils.quat_error_magnitude(actions[:, 3:7], robot_eef_w[:, 3:7])
            orientations_close = torch.isclose(quat_diff_err, torch.tensor(0.0), atol=0.005)
            # print(positions_close, orientations_close)

            if positions_close.all() and orientations_close.all():
                if pose_idx >= len(discrete_poses):
                    dlog.save_observations(observations=observations, poses=robot_eef_w, idx=pose_idx, last_obs=True)
                    break
                else:
                    dlog.save_observations(observations=observations, poses=robot_eef_w, idx=pose_idx)
                # break  # remove after testing
                
                joint_pos = robot.data.default_joint_pos.clone()
                joint_vel = robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel)
                robot.reset()
                avg_step_count = (avg_step_count + step_count) / (pose_idx + 1)
                if step_count > max_steps:
                    max_steps = step_count
                step_count = 0
                pose_idx += 1
                # print(f"  --  avg steps: {avg_step_count:.2f} -- max steps: {max_steps}", end="")



            # Remove after testing
            # sensors = env.unwrapped.sensors
            # for sensor_name, sensor in sensors.items():
            #     # print sensor location
            #     sensor_pos_w = sensor.data.pos_w.torch
            #     print(sensor_pos_w)
            # print(f"\nGoal pos: {discrete_poses[pose_idx][0:3]}, goal quat: {discrete_poses[pose_idx][3:7]}")
            # print(f"Curr pos: {robot.data.body_pos_w[:, robot.find_bodies('ur10e__wrist_3_link')[0]]}")
            # print(f"Curr quat: {robot.data.body_quat_w[:, robot.find_bodies('ur10e__wrist_3_link')[0]]}")
            # print(f"Root pos: {robot.data.root_pos_w.torch}")
            # print(f"Root quat: {robot.data.root_quat_w.torch}")
            # print(f"scene origins: {env.unwrapped.scene.env_origins}")

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
