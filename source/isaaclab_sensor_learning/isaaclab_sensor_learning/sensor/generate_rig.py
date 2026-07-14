#!/usr/bin/env python3
from isaaclab.sensors import Camera, CameraCfg, MultiMeshRayCasterCamera, MultiMeshRayCasterCameraCfg, OffsetCfg
from isaaclab.sim.spawners.sensors import PinholeCameraCfg
import isaaclab.utils.math as math_utils

from isaaclab_sensor_learning import CFG_DIR
from isaaclab_sensor_learning.sensor.yaml_to_cfg import load_sensor_yaml, rig_dict_to_sensor_cfgs
from isaaclab_sensor_learning.sensor.spherical_sensor_layout_generator import LloydSphereSensorLayout
from isaaclab_sensor_learning.sensor.planar_sensor_layout_generator import PlaneSensorLayout
import isaaclab_sensor_learning.sensor.camera_utils as camera_utils
import isaaclab_sensor_learning.sensor.lidar_utils as lidar_utils
import isaaclab_sensor_learning.sensor.rig_utils as rig_utils

import torch

Sensor = Camera | MultiMeshRayCasterCamera
SensorCfg = CameraCfg | MultiMeshRayCasterCameraCfg


class RigGenerator:
    def __init__(self, rig_sensor_cfgs: list[dict]):
        self._rig_sensor_cfgs = rig_sensor_cfgs
        self._sensor_groups = self.group_sensors_by_type()
        print(f"[INFO]: Sensor groups: {self._sensor_groups}")
        # self.lloyd_layout_generator: LloydSphereSensorLayout
        # self.planar_layout_generator: PlaneSensorLayout
        return

    def group_sensors_by_type(self) -> dict[str, list[str]]:
        """Group sensors by type. Return a dictionary of sensor type to list of sensor names."""
        sensor_groups = {
            "camera": [],
            "lidar": [],
            "tof": [],
        }
        print(f"[INFO]: Sensor rig cfgs: {self._rig_sensor_cfgs}")
        for rig_sensor_cfg in self._rig_sensor_cfgs:
            sensor_name = rig_sensor_cfg["name"]
            sensor_type = rig_sensor_cfg["sensor_type"]
            if sensor_type not in sensor_groups:
                raise ValueError(f"Unsupported sensor type: {sensor_type}")
            sensor_groups[sensor_type].append(rig_sensor_cfg)
        return sensor_groups

    def generate_eef_layout(self, layout_dict: dict):
        # for now, let's combine cameras and tofs. may change later
        optical_sensors = self._sensor_groups["camera"] + self._sensor_groups["tof"]
        if layout_dict["layout_type"] == "sphere":
            self.lloyd_layout_generator = LloydSphereSensorLayout(
                sensors=optical_sensors,
                n_sensors=len(optical_sensors),
                radius=layout_dict["radius"],
                colatitude=layout_dict["colatitude"],
                max_iter=layout_dict.get("max_iter", 200),
                tol=layout_dict.get("tol", 1e-6),
                mesh_points=layout_dict.get("mesh_points", 8000),
            )
            sensor_layout = self.lloyd_layout_generator.generate_sensor_layout()

        # elif layout_dict["layout_type"] == "plane":
        #     self.planar_layout_generator = PlaneSensorLayout(
        #         n_sensors=len(optical_sensors),
        #         width=layout_dict.get("width", 0.5),
        #         height=layout_dict.get("height", 0.5),
        #         # spacing=layout_dict.get("spacing", 0.1),
        #     )
        #     sensor_layout = self.planar_layout_generator.generate_sensor_layout()

        print(f"[INFO]: Generated sensor layout: {sensor_layout}")

        return sensor_layout

    def generate_eef_sensor_cfgs(self, sensor_layout: dict):
        sensor_cfgs = {}
        """/World/envs/env_1/robot/wrist_3_link/flange"""
        for i, sensor_rig_data in enumerate(sensor_layout["sensors"]):
            sensor_name = sensor_rig_data["name"]
            sensor_type = sensor_rig_data["sensor_type"]
            sensor_rig_pose = sensor_layout["poses"][
                i
            ]  # NOTE: iteratively assigns sensors. Does this need to be randomized?

            sensor_metadata = load_sensor_yaml(sensor_rig_data)

            if sensor_type == "tof":
                depth_sensor_metadata = sensor_metadata["depth"]
                # Get internal sensor offset
                sensing_unit_offset = depth_sensor_metadata["sensing_unit_offset"]
                offset_pos = torch.tensor(
                    [sensing_unit_offset["x"], sensing_unit_offset["y"], sensing_unit_offset["z"]], dtype=torch.float32
                )
                offset_rot_euler = torch.tensor(
                    [sensing_unit_offset["roll"], sensing_unit_offset["pitch"], sensing_unit_offset["yaw"]],
                    dtype=torch.float32,
                )
                internal_offset_mat = math_utils.make_pose(
                    pos=offset_pos, rot=math_utils.matrix_from_euler(offset_rot_euler, convention="XYZ")
                )
                # Rig generator offset
                sensor_offset_mat = math_utils.make_pose(
                    pos=torch.tensor(sensor_rig_pose[0:3], dtype=torch.float32),
                    rot=math_utils.matrix_from_quat(torch.tensor(sensor_rig_pose[3:7], dtype=torch.float32)),
                )
                full_offset_mat = sensor_offset_mat @ internal_offset_mat
                full_offset_pos, full_offset_quat = math_utils.unmake_pose(full_offset_mat)
                offset_cfg = OffsetCfg()
                offset_cfg.pos = full_offset_pos.tolist()
                offset_cfg.quat = full_offset_quat.tolist()
                offset_cfg.convention = "ros"

                # prim_path = f"/World/envs/env_.*/robot/wrist_3_link/tool0/{sensor_name}"
                # prim_path = f"/World/envs/env_.*/robot/Geometry/amiga__base/amiga__body/amiga__tool0/linear_slider__base/linear_slider__base_link/linear_slider__moving_base/linear_slider__flange/linear_slider__tool0/ur10e__base_link/ur10e__base_link_inertia/ur10e__shoulder_link/ur10e__upper_arm_link/ur10e__forearm_link/ur10e__wrist_1_link/ur10e__wrist_2_link/ur10e__wrist_3_link/ur10e__flange/ur10e__tool0/{sensor_name}"
                prim_path = f"/World/envs/env_.*/robot/ur10e__tool0/{sensor_name}"
                # Create sensor config
                sensor_cfgs[sensor_name] = {
                    "type": sensor_type,
                    "cfg": CameraCfg(
                        prim_path=prim_path,
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
                        offset=offset_cfg,
                        update_period=(1 / sensor_metadata["data_rate_hz"]),
                        update_latest_camera_pose=True,
                        debug_vis=True,
                    ),
                }

        return sensor_cfgs

    def generate_rig(self, sensor_cfgs: dict[str, SensorCfg]) -> dict[str, Sensor]:
        """Generate a rig based on sensor configs. Small form-factor sensors (i.e. tofs) are arranged together. All sensors must adhere to allowed sensor positions"""
        sensors = {}
        for sensor_name, sensor_dict in sensor_cfgs.items():
            sensor_cfg = sensor_dict["cfg"]
            sensor_type = sensor_dict["type"]
            if sensor_type == "camera":
                ...
            elif sensor_type == "tof":
                sensors[sensor_name] = Camera(cfg=sensor_cfg)
            elif sensor_type == "lidar":
                ...
            else:
                raise ValueError(f"Unsupported sensor type: {sensor_type}")

        return sensors


def main():
    rig_yaml_path = CFG_DIR / "rigs" / "test_rig0.yaml"

    sensor_cfgs = load_sensor_yaml(rig_yaml_path)
    rig_generator = RigGenerator(sensor_cfgs=sensor_cfgs)
    rig_cfg = rig_generator.generate_rig(layout_type="sphere")

    return


if __name__ == "__main__":
    main()
