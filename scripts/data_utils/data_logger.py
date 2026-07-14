#!/usr/bin/env python3
import h5py as h5
import os
from datetime import datetime as dt
import numpy as np
import torch
import warp

from isaaclab.sensors import CameraData
import pprint as pp


class DataLogger:
    def __init__(self):
        self.TARGET_CHUNK_BYTES = 64 * 1024 * 1024  # 64 MB
        self.datafile_path, self.trial_name = self.get_file_path()
        self.sensor_obs_buf = {}
        self.actual_poses_buf = []
        self.idx_buf = []
        # self.row_cursor = 0
        self.flush_every = None
        return

    def _compute_flush_every(self, observations: CameraData) -> dict[str, int]:
        MIN_FLUSH = 16
        MAX_FLUSH = 8192

        flush_every = {}
        for sensor_name, sensor_data in observations.items():
            fields = {
                "position_w": sensor_data.pos_w,
                "quat_w_world": sensor_data.quat_w_world,
                "intrinsic_matrix": sensor_data.intrinsic_matrices,
                "rgb": sensor_data.output["rgb"],
                "depth": sensor_data.output["depth"],
                "instance_segmentation": sensor_data.output["instance_segmentation_fast"],
                "semantic_segmentation": sensor_data.output["semantic_segmentation"],
                "normals": sensor_data.output["normals"],
            }

            total_bytes_per_entry = sum(t.numel() * t.element_size() for t in fields.values())
            n = max(MIN_FLUSH, min(self.TARGET_CHUNK_BYTES // total_bytes_per_entry, MAX_FLUSH))
            flush_every[sensor_name] = n

            print(f"[DataLogger] {sensor_name}: {total_bytes_per_entry/1024:.1f} KB/entry → flush_every={n}")

        return flush_every

    def _get_camera_observation_buffer(self) -> dict:
        buf = {
            "position_w": [],
            "quat_w_world": [],
            "joint_state": [],
            "intrinsic_matrix": [],
            "rgb": [],
            "depth": [],
            "instance_segmentation": [],
            "semantic_segmentation": [],
            "normals": [],
        }
        return buf

    def get_file_path(self) -> str:
        pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(pkg_dir, "data")
        trial_name = f"SL-PDC_{dt.now().strftime('%Y%m%d_%H%M%S')}"
        file_path = os.path.join(data_dir, f"{trial_name}.h5")
        if not os.path.exists(data_dir):
            print(f"Creating data directory at {data_dir}")
            os.makedirs(data_dir)
        if os.path.exists(file_path):
            print(f"File {file_path} already exists. It will be overwritten.")
        return file_path, trial_name

    def save_trial_metadata(self, trial_metadata: dict) -> None:
        with h5.File(self.datafile_path, "w") as file:
            file.attrs["trial_name"] = trial_metadata["trial_name"]
            file.attrs["num_envs"] = trial_metadata["num_envs"]
            # desired poses
            desired_poses_group = file.require_group("poses/desired_poses")
            desired_poses_group.attrs["r_range"] = trial_metadata["r_range"]
            desired_poses_group.attrs["r_size"] = trial_metadata["r_size"]
            desired_poses_group.attrs["theta_cyl_range"] = trial_metadata["theta_cyl_range"]
            desired_poses_group.attrs["theta_cyl_size"] = trial_metadata["theta_cyl_size"]
            desired_poses_group.attrs["z_range"] = trial_metadata["z_range"]
            desired_poses_group.attrs["z_size"] = trial_metadata["z_size"]
            desired_poses_group.attrs["theta_ori_range"] = trial_metadata["theta_ori_range"]
            desired_poses_group.attrs["theta_ori_size"] = trial_metadata["theta_ori_size"]
            desired_poses_group.attrs["phi_ori_range"] = trial_metadata["phi_ori_range"]
            desired_poses_group.attrs["phi_ori_size"] = trial_metadata["phi_ori_size"]

            desired_poses_group.create_dataset("eef_poses", data=trial_metadata["poses"], compression="gzip")

            # actual poses
            actual_poses_group = file.require_group("poses/actual_poses")
            actual_poses_group.create_dataset(
                name="eef_poses",
                shape=(trial_metadata["num_envs"], len(trial_metadata["poses"]), 7),
                dtype=np.float32,
                compression="gzip",
            )

            # joint states
            joint_states_group = file.require_group("joint_states")
            joint_states_group.require_dataset(
                name="joint_state",
                shape=(
                    trial_metadata["num_envs"],
                    len(trial_metadata["poses"]),
                    len(trial_metadata["robot_joint_names"]),
                ),
                chunks=self._get_entries_per_chunk(
                    trial_metadata["num_envs"], (len(trial_metadata["poses"]), len(trial_metadata["robot_joint_names"]))
                ),
                dtype=np.float32,
                compression="gzip",
            )

        return

    def save_tree_metadata(self, tree_metadata: dict) -> None:
        with h5.File(self.datafile_path, "a") as file:
            tree_group = file.require_group("trees")
            for env_idx, tree_info in tree_metadata.items():
                env_group = tree_group.require_group(f"env_{env_idx}")
                env_group.attrs["env_idx"] = tree_info["env_idx"]
                env_group.attrs["prim_path"] = tree_info["prim_path"]
                env_group.attrs["tree_usd_path"] = tree_info["tree_usd_path"]
                env_group.attrs["tree_namespace"] = tree_info["tree_namespace"]
                env_group.attrs["tree_type"] = tree_info["tree_type"]
                env_group.attrs["tree_id"] = tree_info["tree_id"]
                env_group.attrs["pose"] = tree_info["pose"]
        return

    def save_sensor_metadata(self, sensor_metadata: dict, n_poses: int, num_envs: int) -> None:
        with h5.File(self.datafile_path, "a") as file:
            for sensor_name, sensor_info in sensor_metadata.items():
                self.sensor_obs_buf[sensor_name] = self._get_camera_observation_buffer()
                sensor_group = file.require_group(f"sensors/{sensor_name}")
                for key, value in sensor_info.items():
                    sensor_group.attrs[key] = value

                sensor_group.require_dataset(
                    name="pose",
                    shape=(num_envs, n_poses, 7),
                    chunks=self._get_entries_per_chunk(num_envs, (n_poses, 7)),
                    dtype=np.float32,
                    compression="gzip",
                )
                sensor_group.require_dataset(
                    name="intrinsic_matrix",
                    shape=(num_envs, n_poses, 3, 3),
                    chunks=self._get_entries_per_chunk(num_envs, (n_poses, 3, 3)),
                    dtype=np.float32,
                    compression="gzip",
                )
                sensor_group.require_dataset(
                    name="rgb",
                    shape=(num_envs, n_poses, sensor_info["height"], sensor_info["width"], 3),
                    chunks=self._get_entries_per_chunk(
                        num_envs, (n_poses, sensor_info["height"], sensor_info["width"], 3)
                    ),
                    dtype=np.float32,
                    compression="gzip",
                )
                sensor_group.require_dataset(
                    name="depth",
                    shape=(num_envs, n_poses, sensor_info["height"], sensor_info["width"]),
                    chunks=self._get_entries_per_chunk(
                        num_envs, (n_poses, sensor_info["height"], sensor_info["width"])
                    ),
                    dtype=np.float32,
                    compression="gzip",
                )
                sensor_group.require_dataset(
                    name="instance_segmentation",
                    shape=(num_envs, n_poses, sensor_info["height"], sensor_info["width"], 4),
                    chunks=self._get_entries_per_chunk(
                        num_envs, (n_poses, sensor_info["height"], sensor_info["width"], 4)
                    ),
                    dtype=np.float32,
                    compression="gzip",
                )
                sensor_group.require_dataset(
                    name="semantic_segmentation",
                    shape=(num_envs, n_poses, sensor_info["height"], sensor_info["width"], 4),
                    chunks=self._get_entries_per_chunk(
                        num_envs, (n_poses, sensor_info["height"], sensor_info["width"], 4)
                    ),
                    dtype=np.float32,
                    compression="gzip",
                )
                sensor_group.require_dataset(
                    name="normals",
                    shape=(num_envs, n_poses, sensor_info["height"], sensor_info["width"], 3),
                    chunks=self._get_entries_per_chunk(
                        num_envs, (n_poses, sensor_info["height"], sensor_info["width"], 3)
                    ),
                    dtype=np.float32,
                    compression="gzip",
                )
        return

    def _get_entries_per_chunk(self, n_poses: int, data_shape: tuple, dtype=np.float32) -> tuple:
        bytes_per_entry = np.empty(data_shape, dtype=dtype).itemsize * 16  # 4x4 matrix
        entries_per_chunk = min(n_poses, self.TARGET_CHUNK_BYTES // bytes_per_entry)
        return tuple([entries_per_chunk] + list(data_shape))

    def _reshape_tensor_to_numpy(self, tensors: list[torch.Tensor], shape: tuple = (0, 1)) -> np.ndarray:
        return torch.stack(tensors).transpose(*shape).detach().cpu().numpy()

    def save_observations(self, observations: CameraData, poses: torch.Tensor, idx: int, last_obs: bool = False) -> None:
        """Save observations and poses to HDF5 file. Flushes buffers every N observations or if last_obs is True."""
        # print(poses.shape)

        # # append actual poses to buffer
        # print(f"POSES: ", poses.detach().cpu().numpy())
        self.actual_poses_buf.append(poses.detach().cpu().numpy())
        self.idx_buf.append(idx)

        # append sensor data to buffers
        for i, (sensor_name, sensor_data) in enumerate(observations.items()):
            self.sensor_obs_buf[sensor_name]["position_w"].append(sensor_data.pos_w.torch)
            self.sensor_obs_buf[sensor_name]["quat_w_world"].append(sensor_data.quat_w_world.torch)
            self.sensor_obs_buf[sensor_name]["intrinsic_matrix"].append(sensor_data.intrinsic_matrices.torch)
            self.sensor_obs_buf[sensor_name]["rgb"].append(sensor_data.output["rgb"].torch)
            self.sensor_obs_buf[sensor_name]["depth"].append(sensor_data.output["depth"].torch)
            self.sensor_obs_buf[sensor_name]["instance_segmentation"].append(
                sensor_data.output["instance_segmentation_fast"].torch
            )
            self.sensor_obs_buf[sensor_name]["semantic_segmentation"].append(
                sensor_data.output["semantic_segmentation"].torch
            )
            self.sensor_obs_buf[sensor_name]["normals"].append(sensor_data.output["normals"].torch)

        flush = False
        for sensor_name in observations.keys():

            len_buffers = len(self.sensor_obs_buf[sensor_name]["position_w"])
            # print(self.sensor_obs_buf[sensor_name]["position_w"][0].warp)
            if len_buffers >= 2 or last_obs:  # flush every N observations or if it's the last observation
                flush = True
                # reshape data to (num_envs, n_poses, ...) and save to HDF5
                chunked_positions = self._reshape_tensor_to_numpy(self.sensor_obs_buf[sensor_name]["position_w"])
                chunked_quats = self._reshape_tensor_to_numpy(self.sensor_obs_buf[sensor_name]["quat_w_world"])
                chunked_poses = np.concatenate([chunked_positions, chunked_quats], axis=-1)
                chunked_intrinsics = self._reshape_tensor_to_numpy(self.sensor_obs_buf[sensor_name]["intrinsic_matrix"])
                chunked_rgbs = self._reshape_tensor_to_numpy(self.sensor_obs_buf[sensor_name]["rgb"])
                chunked_depths = self._reshape_tensor_to_numpy(self.sensor_obs_buf[sensor_name]["depth"]).squeeze(
                    axis=-1
                )
                chunked_instance_segs = self._reshape_tensor_to_numpy(
                    self.sensor_obs_buf[sensor_name]["instance_segmentation"]
                )
                chunked_semantic_segs = self._reshape_tensor_to_numpy(
                    self.sensor_obs_buf[sensor_name]["semantic_segmentation"]
                )
                chunked_normals = self._reshape_tensor_to_numpy(self.sensor_obs_buf[sensor_name]["normals"])

                self.sensor_obs_buf[sensor_name]["position_w"].clear()
                self.sensor_obs_buf[sensor_name]["quat_w_world"].clear()
                self.sensor_obs_buf[sensor_name]["intrinsic_matrix"].clear()
                self.sensor_obs_buf[sensor_name]["rgb"].clear()
                self.sensor_obs_buf[sensor_name]["depth"].clear()
                self.sensor_obs_buf[sensor_name]["instance_segmentation"].clear()
                self.sensor_obs_buf[sensor_name]["semantic_segmentation"].clear()
                self.sensor_obs_buf[sensor_name]["normals"].clear()

                # end = self.row_cursor + len_buffers
                # print(f"\n[DataLogger] {self.idx_buf}")
                # print(chunked_depths.shape)
                # print(chunked_depths)

                # Save the buffered observations
                with h5.File(self.datafile_path, "a") as file:
                    sensor_group = file.require_group(f"sensors/{sensor_name}")
                    sensor_group["pose"][:, self.idx_buf[0]:self.idx_buf[-1] + 1, :] = chunked_poses
                    sensor_group["intrinsic_matrix"][:, self.idx_buf[0]:self.idx_buf[-1] + 1, :] = chunked_intrinsics
                    sensor_group["rgb"][:, self.idx_buf[0]:self.idx_buf[-1] + 1, :] = chunked_rgbs
                    sensor_group["depth"][:, self.idx_buf[0]:self.idx_buf[-1] + 1, :] = chunked_depths
                    sensor_group["instance_segmentation"][:, self.idx_buf[0]:self.idx_buf[-1] + 1, :] = chunked_instance_segs
                    sensor_group["semantic_segmentation"][:, self.idx_buf[0]:self.idx_buf[-1] + 1, :] = chunked_semantic_segs
                    sensor_group["normals"][:, self.idx_buf[0]:self.idx_buf[-1] + 1, :] = chunked_normals

        if flush:
            # Save the actual poses
            stacked_poses = np.stack(self.actual_poses_buf, axis=1)  # shape: (num_envs, n_poses, 7)
            # print(self.row_cursor, end, stacked_poses.shape)
            with h5.File(self.datafile_path, "a") as file:
                pose_group = file.require_group("poses/actual_poses")
                pose_group["eef_poses"][:, self.idx_buf[0]:self.idx_buf[-1] + 1, :] = stacked_poses
                self.actual_poses_buf.clear()
                self.idx_buf.clear()
            flush = False

            # self.row_cursor += len_buffers
        return

    def _flush_buffers(self):
        # This method can be used to implement any additional flushing logic if needed
        pass
