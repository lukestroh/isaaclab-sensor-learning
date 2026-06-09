#!/usr/bin/env python3
from isaaclab.sensors import CameraCfg, MultiMeshRayCasterCameraCfg, OffsetCfg
from isaaclab.sim.spawners.sensors import PinholeCameraCfg
from pathlib import Path
from pose_data_capture.sensor import camera_utils, lidar_utils
from scipy.spatial.transform import Rotation as R

import numpy as np
import yaml

CFG_DIR = Path(__file__).parent.parent / "config"
RIG_CFG_DIR = CFG_DIR / "rigs"
SENSOR_CFG_DIR = CFG_DIR / "sensors"


def load_rig_yaml(rig_yaml_path: Path) -> dict:
    """Load the rig yaml file."""
    with open(rig_yaml_path, "r") as f:
        rig_cfg = yaml.safe_load(f)
    return rig_cfg


def load_sensor_yaml(sensor_rig_data: dict) -> dict:
    sensor_metadata_path = SENSOR_CFG_DIR / sensor_rig_data["sensor_type"] / f"{sensor_rig_data['model']}.yaml"
    with open(sensor_metadata_path, "r") as f:
        sensor_metadata = yaml.safe_load(f)
    return sensor_metadata


def rig_yaml_to_sensor_cfgs(rig_yaml_path: Path) -> dict[str, CameraCfg | MultiMeshRayCasterCameraCfg]:
    """Convert the rig yaml file to a dictionary of sensor name to sensor config."""
    rig_cfg = load_rig_yaml(rig_yaml_path)
    sensor_cfgs = {}
    for rig_sensor_data in rig_cfg["sensors"]:
        name = rig_sensor_data["name"]
        sensor_metadata = load_sensor_yaml(rig_sensor_data)
        ##########################################################
        # Camera config
        ##########################################################
        if sensor_metadata["sensor_type"] == "camera":

            sensor_cfgs[name] = CameraCfg()
        ##########################################################
        # ToF sensor config
        ##########################################################
        elif sensor_metadata["sensor_type"] == "tof":
            depth_sensor_metadata = sensor_metadata["depth"]
            # vfov, hfov = camera_utils.get_fov_from_dfov(
            #     dfov=depth_sensor_metadata["dfov"],
            #     width=depth_sensor_metadata["width"],
            #     height=depth_sensor_metadata["height"]
            # )
            sensing_unit_offset = depth_sensor_metadata["sensing_unit_offset"]
            offset_pos = (
                sensing_unit_offset["x"],
                sensing_unit_offset["y"],
                sensing_unit_offset["z"],
            )
            offset_rot = R.from_euler(
                "xyz",
                [
                    sensing_unit_offset["roll"],
                    sensing_unit_offset["pitch"],
                    sensing_unit_offset["yaw"],
                ],
                degrees=True,
            ).as_quat()
            offset = OffsetCfg(
                pos=offset_pos,
                rot=[offset_rot[3], offset_rot[0], offset_rot[1], offset_rot[2]],
                # convention="ros" # +Z forward, -Y up
            )
            offset.convention = "ros"
            sensor_cfgs[name] = CameraCfg(
                prim_path=f"/World/envs/env_0/{name}",
                width=depth_sensor_metadata["width"],
                height=depth_sensor_metadata["height"],
                data_types=["rgb", "depth", "normals", "semantic_segmentation", "instance_segmentation_fast"],
                spawn=PinholeCameraCfg.from_intrinsic_matrix(
                    intrinsic_matrix=camera_utils.get_intrinsic_matrix_from_dfov(
                        dfov=depth_sensor_metadata["dfov"],
                        width=depth_sensor_metadata["width"],
                        height=depth_sensor_metadata["height"],
                        degrees=True,
                    ).flatten(),
                    width=depth_sensor_metadata["width"],
                    height=depth_sensor_metadata["height"],
                    clipping_range=(depth_sensor_metadata["z_near"], depth_sensor_metadata["z_far"]),
                ),
                depth_clipping_behavior="max",
                offset=offset,
                update_period=1 / sensor_metadata["data_rate_hz"],
                debug_vis=True,
            )
        ##########################################################
        # LiDAR config
        ##########################################################
        elif sensor_metadata["sensor_type"] == "lidar":
            sensor_cfgs[name] = MultiMeshRayCasterCameraCfg()
        else:
            raise ValueError(f"Unsupported sensor type: {sensor_metadata['sensor_type']}")
    return sensor_cfgs


def main():
    rig_yaml_path = RIG_CFG_DIR / "rig0.yaml"
    sensor_cfgs = rig_yaml_to_sensor_cfgs(rig_yaml_path)
    return


if __name__ == "__main__":
    main()
