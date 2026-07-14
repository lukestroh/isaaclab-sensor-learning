#!/usr/bin/env python3
from isaaclab.app import AppLauncher
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="Test rig generator script")
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


import gymnasium as gym
import torch
import numpy as np

from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_sensor_learning.tasks  # noqa: F401
import data_utils.data_logger as data_logger
import data_utils.pose_generator as pose_generator
import pprint as pp
import os
import re

import time
from isaacsim.core.utils import stage as stage_utils

from isaaclab_sensor_learning import CFG_DIR
from isaaclab_sensor_learning.sensor.yaml_to_cfg import load_sensor_yaml, load_rig_yaml
from isaaclab_sensor_learning.sensor.spherical_sensor_layout_generator import LloydSphereSensorLayout
from isaaclab_sensor_learning.sensor.planar_sensor_layout_generator import PlaneSensorLayout
from isaaclab_sensor_learning.sensor.generate_rig import RigGenerator
import isaaclab_sensor_learning.sensor.camera_utils as camera_utils
import isaaclab_sensor_learning.sensor.lidar_utils as lidar_utils
import isaaclab_sensor_learning.sensor.rig_utils as rig_utils


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

    # # print(stage_utils.print_stage_prim_paths())

    # test_rig = os.path.join(CFG_DIR, "rigs", "test_rig0.yaml")
    # rig_cfg = load_rig_yaml(test_rig)
    # rig_generator = RigGenerator(sensor_cfgs=rig_cfg["sensors"])
    # layout_dict = {
    #     # "sensor_model": "filler_sensor_model",
    #     "layout_type": "sphere",
    #     "radius": 0.2,
    #     "colatitude": 25, # degrees
    #     "max_iter": 200,
    #     "tol": 1e-6,
    #     "mesh_points": 8000
    # }
    # eef_sensor_layout = rig_generator.generate_eef_layout(layout_dict=layout_dict)
    # rig_generator.generate_eef_sensor_cfgs(sensor_layout=eef_sensor_layout)
    # print(eef_sensor_layout)
    # rig_generator.generate_rig(layout_type="sphere")

    while simulation_app.is_running():
        with torch.inference_mode():
            # compute zero actions
            # actions = torch.zeros((env.unwrapped.num_envs, env.action_space.shape[0]), dtype=torch.float32, device=env.unwrapped.device)
            # # print(actions.shape)
            # observations, rewards, terminated, truncated, info = env.step(actions)

            break

    env.close()

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
