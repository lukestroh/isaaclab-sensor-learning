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

from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_sensor_learning.tasks  # noqa: F401
from isaaclab_sensor_learning.utils import quaternion_utils as qutils

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

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()

    print(env.unwrapped.cfg.__dict__.keys())

    # set up data file paths and metadata
    # tree_name = os.path.splitext(os.path.basename(env_cfg.tree_usd_path))[
    #     0
    # ]  # TODO: move tree path as arg to set up multiple envs
    tree_name = "test_tree_00000"  # TODO: remove after testing
    _tree = tree_name.split("_")
    tree_namespace, tree_type, tree_id = _tree[0], _tree[1], _tree[2]
    dlog = data_logger.DataLogger(tree_name=tree_name)
    datafile_path, trial_name = dlog.get_file_path(tree_name=tree_name)

    # generate discrete poses
    x_range = (-1.0, 1.0)
    y_range = (-1.0, 1.0)
    z_range = (0.0, 2.0)
    theta_range = (-np.pi / 4, np.pi / 4)
    phi_range = (-np.pi / 4, np.pi / 4)
    x_size = 5
    y_size = 5
    z_size = 5
    angles_size = 5
    start_orientation = np.array([0, 0, 1])
    discrete_poses = pose_generator.generate_discrete_poses(
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        theta_range=theta_range,
        phi_range=phi_range,
        x_size=x_size,
        y_size=y_size,
        z_size=z_size,
        angles_size=angles_size,
        start_orientation=start_orientation,
    )

    # metadata dicts
    trial_metadata = {
        "trial_name": trial_name,
        "n_envs": env.unwrapped.cfg.n_envs,
        "x_range": x_range,
        "y_range": y_range,
        "z_range": z_range,
        "theta_range": theta_range,
        "phi_range": phi_range,
        "x_size": x_size,
        "y_size": y_size,
        "z_size": z_size,
        "angles_size": angles_size,
        "poses": discrete_poses,
    }
    tree_metadata = {
        "tree_usd_path": "testtesttesttest",
        "tree_namespace": tree_namespace,
        "tree_type": tree_type,
        "tree_id": tree_id,
        "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    }
    
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
            "receiver_offset": np.concatenate(
                [sensor.cfg.offset.pos, qutils.wxyz_to_xyzw(np.asarray(sensor.cfg.offset.rot))]
            ),
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
        sensor_metadata=sensor_metadata, n_poses=discrete_poses.shape[0], n_envs=env.unwrapped.cfg.n_envs
    )

    # simulate environment
    pose_idx = 0
    while simulation_app.is_running():
        print(f"\rpose idx: {pose_idx}/{len(discrete_poses)-1}", end="")
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.tensor(discrete_poses[pose_idx][np.newaxis, :], device=env.unwrapped.device)
            observations, rewards, terminated, truncated, info = env.step(actions)

            pose_idx += 1
            if pose_idx >= len(discrete_poses):
                dlog.save_observations(observations=observations, last_obs=True)
                break
            else:
                dlog.save_observations(observations=observations)
            break

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
