#!/usr/bin/env python3

# from isaaclab.sensors import CameraCfg, MultiMeshRayCasterCameraCfg, OffsetCfg
# from isaaclab.sim.spawners.sensors import PinholeCameraCfg
# from pathlib import Path

# from isaaclab_sensor_learning.sensor import camera_utils, lidar_utils


ALLOWED_SENSOR_POSITIONS = {
    "camera": ["eef_link", "mobile_base_link", "slider_link"],
    "tof": ["eef_link"],
    "lidar": ["mobile_base_link", "slider_link"],
}

eef_link_names = {
    "universal_robotics": "wrist_3_link",
    "panda": "panda_hand",
    "fr3": "fr3_link8",
}
