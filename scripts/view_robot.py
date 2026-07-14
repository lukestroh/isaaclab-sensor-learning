#!/usr/bin/env python3
from isaaclab.app import AppLauncher
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="Convert URDF to USD script for pose data capture")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Template-View-USD-Direct-v0", help="Name of the task.")
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
from isaaclab_tasks.utils import parse_env_cfg

# from pose_data_capture.robot import motion_planning as mp
import isaaclab_sensor_learning.tasks  # noqa: F401
from isaaclab_sensor_learning.utils import usd_utils


import omni.usd
from pxr import UsdPhysics
import os
import pprint as pp

def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # reset environment
    env.reset()

    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        print(prim.GetPath(), prim.GetTypeName())

    # # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros(size=(1,7), dtype=torch.float32)
            env.step(actions)

    # close the simulator
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
