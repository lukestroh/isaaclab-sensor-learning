#!/usr/bin/env python3
from isaaclab.sensors import CameraCfg, MultiMeshRayCasterCameraCfg, OffsetCfg
from isaaclab.sim.spawners.sensors import PinholeCameraCfg
from pathlib import Path
from isaaclab_sensor_learning.sensor import camera_utils, lidar_utils, rig_utils
from scipy.spatial.transform import Rotation as R

import os
import numpy as np
import yaml

from isaaclab_sensor_learning import CFG_DIR

RIG_CFG_DIR = os.path.join(CFG_DIR, "rigs")
SENSOR_CFG_DIR = os.path.join(CFG_DIR, "sensors")

SensorCfg = CameraCfg | MultiMeshRayCasterCameraCfg



def load_yaml(yaml_path: Path) -> dict:
    """Load the yaml file."""
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_sensor_yaml(sensor_rig_data: dict) -> dict:
    # print(sensor_rig_data)
    sensor_metadata_path = os.path.join(
        SENSOR_CFG_DIR, sensor_rig_data["sensor_type"], f"{sensor_rig_data['model']}.yaml"
    )
    with open(sensor_metadata_path, "r") as f:
        sensor_metadata = yaml.safe_load(f)
    return sensor_metadata


def group_sensors_by_type(sensor_rig_data: dict) -> dict[str, list[str]]:
    """Group sensors by type. Return a dictionary of sensor type to list of sensor names."""
    sensor_groups = {
        "camera": [],
        "lidar": [],
        "tof": [],
    }
    for sensor in sensor_rig_data["sensors"]:
        sensor_type = sensor["sensor_type"]
        if sensor_type not in sensor_groups:
            raise ValueError(f"Unsupported sensor type: {sensor_type}")
        sensor_groups[sensor_type].append(sensor["name"])
    return sensor_groups


def rig_dict_to_sensor_cfgs(rig_dict: dict) -> dict[str, SensorCfg]:
    """Assume the same format as the rig yaml file, but already loaded as a dictionary. Convert the rig dict to a dictionary of sensor name to sensor config."""
    sensor_metadatas = {}
    sensor_cfgs = {}
    # sensor_groups = group_sensors_by_type(rig_dict)
    # print(f"[INFO]: Sensor groups: {sensor_groups}")

    # for sensor_rig_data in rig_dict["sensors"]:
    #     name = sensor_rig_data["name"]
    #     sensor_metadata = load_sensor_yaml(sensor_rig_data)
    #     sensor_metadatas[name] = sensor_metadata
    #     sensor_groups[sensor_metadata["sensor_type"]].append(name)

    # # Assign sensor locations based on allowed sensor positions and rig configuration. This is needed to determine the correct offsets for each sensor.
    # # for sensor_type, sensor_names in sensor_groups.items():
    # #     if sensor_type == "camera":
    # #         ...
    # #     elif sensor_type == "tof":
    # #         rig_generator = rig_utils.RigGenerator(sensor_cfgs=sensor_metadatas)

    # #
    # for name, sensor_metadata in sensor_metadatas.items():

    #     ##########################################################
    #     # Camera config
    #     ##########################################################
    #     if sensor_metadata["sensor_type"] == "camera":
    #         sensor_cfgs[name] = CameraCfg()

    #     ##########################################################
    #     # Depth camera config
    #     ##########################################################
    #     if sensor_metadata["sensor_type"] == "depth_camera":
    #         sensor_cfgs[name+"_rgb"] = CameraCfg()
    #         sensor_cfgs[name+"_depth"] = CameraCfg()

    #     ##########################################################
    #     # ToF sensor config
    #     ##########################################################
    #     elif sensor_metadata["sensor_type"] == "tof":
    #         sensor_groups[sensor_metadata["sensor_type"]].append(name)

    #         depth_sensor_metadata = sensor_metadata["depth"]
    #         sensing_unit_offset = depth_sensor_metadata["sensing_unit_offset"]
    #         offset_pos = (
    #             sensing_unit_offset["x"],
    #             sensing_unit_offset["y"],
    #             sensing_unit_offset["z"],
    #         )
    #         offset_rot = R.from_euler(
    #             "xyz",
    #             [
    #                 sensing_unit_offset["roll"],
    #                 sensing_unit_offset["pitch"],
    #                 sensing_unit_offset["yaw"],
    #             ],
    #             degrees=True,
    #         ).as_quat()
    #         offset = OffsetCfg(
    #             pos=offset_pos,
    #             rot=qutils.xyzw_to_wxyz(offset_rot),
    #         )
    #         offset.convention = "ros"
    #         sensor_cfgs[name] = {
    #             "type": sensor_metadata["sensor_type"],
    #             "cfg":CameraCfg(
    #                 prim_path=f"/World/envs/env_.*/robot/{name}",  # TODO: need to handle multiple envs
    #                 width=depth_sensor_metadata["width"],
    #                 height=depth_sensor_metadata["height"],
    #                 data_types=["rgb", "depth", "normals", "semantic_segmentation", "instance_segmentation_fast"],
    #                 spawn=PinholeCameraCfg.from_intrinsic_matrix(
    #                     intrinsic_matrix=camera_utils.get_intrinsic_matrix_from_dfov(
    #                         dfov=depth_sensor_metadata["dfov"],
    #                         width=depth_sensor_metadata["width"],
    #                         height=depth_sensor_metadata["height"],
    #                         degrees=True,
    #                     ).flatten(),
    #                     width=depth_sensor_metadata["width"],
    #                     height=depth_sensor_metadata["height"],
    #                     clipping_range=(depth_sensor_metadata["z_near"], depth_sensor_metadata["z_far"]),
    #                 ),
    #                 depth_clipping_behavior="max",
    #                 offset=offset,
    #                 update_period=1 / sensor_metadata["data_rate_hz"],
    #                 debug_vis=True,
    #             ),
    #         }

    #     ##########################################################
    #     # LiDAR config
    #     ##########################################################
    #     elif sensor_metadata["sensor_type"] == "lidar":
    #         sensor_cfgs[name] = MultiMeshRayCasterCameraCfg()
    #     else:
    #         raise ValueError(f"Unsupported sensor type: {sensor_metadata['sensor_type']}")
    return sensor_cfgs


def rig_yaml_to_sensor_cfgs(rig_yaml_path: Path) -> dict[str, SensorCfg]:
    """Convert the rig yaml file to a dictionary of sensor name to sensor config."""
    rig_cfg = load_rig_yaml(rig_yaml_path)
    sensor_cfgs = rig_dict_to_sensor_cfgs(rig_cfg)
    return sensor_cfgs


def main():
    rig_yaml_path = RIG_CFG_DIR / "rig0.yaml"
    sensor_cfgs = rig_yaml_to_sensor_cfgs(rig_yaml_path)
    return


if __name__ == "__main__":
    main()
