#!/usr/bin/env python3
from turtle import update

import h5py as h5
import os
from datetime import datetime as dt
import numpy as np
import torch
import warp

from isaaclab.sensors import CameraData
import pprint as pp


class DataLogger:
    def __init__(self, process_type: str):
        self.TARGET_CHUNK_BYTES = 64 * 1024 * 1024  # 64 MB
        self.process_types_to_label = {
            "capture": "PDC",
            "evaluation": "PDEVAL",
        }
        self.datafile_path, self.trial_name = self.get_file_path(process_type=process_type)
        self.sensor_obs_buf = {}
        self.actual_poses_buf = []
        self.idx_buf = []
        # self.row_cursor = 0
        self.flush_every = None

        self.labels_dict = {"semantic_labels": {}, "instance_labels": {}}

        

        return

    # def _compute_flush_every(self, observations: CameraData) -> dict[str, int]:
    #     MIN_FLUSH = 16
    #     MAX_FLUSH = 8192

    #     flush_every = {}
    #     for sensor_name, sensor_data in observations.items():
    #         fields = {
    #             "position_w": sensor_data.pos_w,
    #             "quat_w_world": sensor_data.quat_w_world,
    #             "intrinsic_matrix": sensor_data.intrinsic_matrices,
    #             "rgb": sensor_data.output["rgb"],
    #             "depth": sensor_data.output["depth"],
    #             "instance_segmentation": sensor_data.output["instance_segmentation_fast"],
    #             "semantic_segmentation": sensor_data.output["semantic_segmentation"],
    #             "normals": sensor_data.output["normals"],
    #         }

    #         total_bytes_per_entry = sum(t.numel() * t.element_size() for t in fields.values())
    #         n = max(MIN_FLUSH, min(self.TARGET_CHUNK_BYTES // total_bytes_per_entry, MAX_FLUSH))
    #         flush_every[sensor_name] = n

    #         print(f"[DataLogger] {sensor_name}: {total_bytes_per_entry/1024:.1f} KB/entry → flush_every={n}")

    #     return flush_every

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

    def get_file_path(self, process_type: str) -> tuple[str, str]:
        """
        Get the file path for the data file.
        """
        if process_type not in ["capture", "evaluation"]:
            raise ValueError(f"Invalid process_type: {process_type}. Must be 'capture' or 'evaluation'.")

        pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        data_dir = os.path.join(pkg_dir, "data", process_type)
        trial_name = f"SL-{self.process_types_to_label[process_type]}_{dt.now().strftime('%Y%m%d_%H%M%S')}"
        file_path = os.path.join(data_dir, f"{trial_name}.h5")
        if not os.path.exists(data_dir):
            print(f"Creating data directory at {data_dir}")
            os.makedirs(data_dir)
        if os.path.exists(file_path):
            print(f"File {file_path} already exists. It will be overwritten.")
        return file_path, trial_name

    def save_trial_metadata(self, trial_metadata: dict) -> None:
        # Create labels_dict for each environment to store semantic and instance labels
        for i in range(trial_metadata["num_envs"]):
            self.labels_dict["instance_labels"][f"env_{i}"] = {}

        # Save trial metadata to HDF5 file
        with h5.File(self.datafile_path, "w") as file:
            file.attrs["trial_name"] = trial_metadata["trial_name"]
            file.attrs["seed"] = trial_metadata["seed"]
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
                env_group.attrs["position"] = tree_info["position"]
                env_group.attrs["orientation"] = tree_info["orientation"]

                for instance_path, instance_id in tree_info["instance_path_to_id_map"].items():
                    env_group.attrs[instance_path] = instance_id
            
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
                    shape=(num_envs, n_poses, sensor_info["height"], sensor_info["width"]),
                    chunks=self._get_entries_per_chunk(
                        num_envs, (n_poses, sensor_info["height"], sensor_info["width"])
                    ),
                    dtype=np.float32,
                    compression="gzip",
                )
                sensor_group.require_dataset(
                    name="semantic_segmentation",
                    shape=(num_envs, n_poses, sensor_info["height"], sensor_info["width"]),
                    chunks=self._get_entries_per_chunk(
                        num_envs, (n_poses, sensor_info["height"], sensor_info["width"])
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
        """Reshape a tensor of (n_poses, n_envs, ...) to numpy array of shape (n_envs, n_poses, ...)."""
        return torch.stack(tensors).transpose(*shape).detach().cpu().numpy()

    # def save_observations(self, observations: dict[str, CameraData], poses: torch.Tensor, idx: int) -> None:
    #     for sensor_name, sensor_data in observations.items():
    #         # print(list(zip(sensor_data.info["instance_segmentation_fast"].keys(), sensor_data.info["instance_segmentation_fast"].values())))
    #         with h5.File(self.datafile_path, "a") as file:
    #             sensor_group = file.require_group(f"sensors/{sensor_name}")
    #             sensor_group["pose"][:, idx, :] = torch.cat([sensor_data.pos_w.torch, sensor_data.quat_w_world.torch], dim=-1).detach().cpu().numpy()
    #             sensor_group["intrinsic_matrix"][:, idx, :, :] = sensor_data.intrinsic_matrices.torch.detach().cpu().numpy()
    #             sensor_group["rgb"][:, idx, :, :, :] = sensor_data.output["rgb"].torch.detach().cpu().numpy()
    #             sensor_group["depth"][:, idx, :, :] = sensor_data.output["depth"].torch.squeeze(-1).detach().cpu().numpy()
    #             sensor_group["instance_segmentation"][:, idx, :, :] = sensor_data.output[
    #                 "instance_segmentation_fast"
    #             ].torch.squeeze(-1).detach().cpu().numpy()
    #             id_to_labels = sensor_data.info["instance_segmentation_fast"]['idToLabels']
    #             sensor_group["instance_labels"][:, idx, :, :] = list(zip(id_to_labels.keys(), id_to_labels.values()))
    #             sensor_group["semantic_segmentation"][:, idx, :, :] = sensor_data.output[
    #                 "semantic_segmentation"
    #             ].torch.squeeze(-1).detach().cpu().numpy()
    #             # sensor_group["semantic_segmentation"].attrs["semantic_labels"] = sensor_data.info["semantic_segmentation"]
    #             sensor_group["normals"][:, idx, :, :, :] = sensor_data.output["normals"].torch.detach().cpu().numpy()

    #     return

    def save_pose_observations(
        self, observations: dict[str, CameraData], poses: torch.Tensor, idx: int, last_obs: bool = False
    ) -> None:
        """Save observations and poses to HDF5 file. Flushes buffers every N observations or if last_obs is True."""
        # print(poses.shape)

        # # append actual poses to buffer
        # print(f"POSES: ", poses.detach().cpu().numpy())
        self.actual_poses_buf.append(poses.detach().cpu().numpy())
        self.idx_buf.append(idx)

        # add semantic and instance labels to labels_dict
        for sensor_name, sensor_data in observations.items():
            semantic_labels_dict = sensor_data.info["semantic_segmentation"]
            instance_labels_dict = sensor_data.info["instance_segmentation_fast"]
            for rgba_id, label in instance_labels_dict["idToLabels"].items():
                if label not in ["BACKGROUND", "UNLABELLED"]:  # Does isaac sim have a typo??
                    env_id = label.split("/")[3].split("_")[1]
                    self.labels_dict["instance_labels"][f"env_{env_id}"][rgba_id] = label

            for rgba_id, label_dict in semantic_labels_dict["idToLabels"].items():
                self.labels_dict["semantic_labels"][rgba_id] = label_dict["class"]

        # append sensor data to buffers
        for sensor_name, sensor_data in observations.items():
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
        len_buffers = len(self.sensor_obs_buf[sensor_name]["position_w"])
        if len_buffers >= 1 or last_obs:  # flush every N observations or if it's the last observation
            for sensor_name in observations.keys():
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
                ).squeeze(axis=-1)
                chunked_semantic_segs = self._reshape_tensor_to_numpy(
                    self.sensor_obs_buf[sensor_name]["semantic_segmentation"]
                ).squeeze(axis=-1)
                chunked_normals = self._reshape_tensor_to_numpy(self.sensor_obs_buf[sensor_name]["normals"])

                self.sensor_obs_buf[sensor_name]["position_w"].clear()
                self.sensor_obs_buf[sensor_name]["quat_w_world"].clear()
                self.sensor_obs_buf[sensor_name]["intrinsic_matrix"].clear()
                self.sensor_obs_buf[sensor_name]["rgb"].clear()
                self.sensor_obs_buf[sensor_name]["depth"].clear()
                self.sensor_obs_buf[sensor_name]["instance_segmentation"].clear()
                self.sensor_obs_buf[sensor_name]["semantic_segmentation"].clear()
                self.sensor_obs_buf[sensor_name]["normals"].clear()

                start_idx = self.idx_buf[0]
                end_idx = self.idx_buf[-1] + 1  # +1 because slicing

                # Save the buffered observations
                with h5.File(self.datafile_path, "a") as file:
                    sensor_group = file.require_group(f"sensors/{sensor_name}")
                    sensor_group["pose"][:, start_idx:end_idx, :] = chunked_poses
                    sensor_group["intrinsic_matrix"][:, start_idx:end_idx, :] = chunked_intrinsics
                    sensor_group["rgb"][:, start_idx:end_idx, :] = chunked_rgbs
                    sensor_group["depth"][:, start_idx:end_idx, :] = chunked_depths
                    sensor_group["instance_segmentation"][:, start_idx:end_idx, :] = chunked_instance_segs
                    sensor_group["semantic_segmentation"][:, start_idx:end_idx, :] = chunked_semantic_segs
                    sensor_group["normals"][:, start_idx:end_idx, :] = chunked_normals

        if flush:
            # Save the actual poses
            stacked_poses = np.stack(self.actual_poses_buf, axis=1)  # shape: (num_envs, n_poses, 7)
            with h5.File(self.datafile_path, "a") as file:
                pose_group = file.require_group("poses/actual_poses")
                pose_group["eef_poses"][:, start_idx:end_idx, :] = stacked_poses

            self.actual_poses_buf.clear()
            self.idx_buf.clear()
            flush = False

        return

    def save_skipped_poses(self, skipped_poses: list[int]) -> None:
        """Save the list of skipped poses to the HDF5 file."""
        with h5.File(self.datafile_path, "a") as file:
            pose_group = file.require_group("poses/actual_poses")
            pose_group.create_dataset(
                name="skipped_poses",
                data=np.array(skipped_poses, dtype=np.int32),
                compression="gzip",
            )
        return

    def save_labels_dict(self) -> None:
        """Save the labels_dict to the HDF5 file."""
        with h5.File(self.datafile_path, "a") as file:
            semantic_labels_group = file.require_group("trees/semantic_labels")
            for rgba_id, label in self.labels_dict["semantic_labels"].items():
                semantic_labels_group.attrs[str(rgba_id)] = label

            for env_name, env_labels in self.labels_dict["instance_labels"].items():
                tree_env_group = file.require_group(f"trees/{env_name}")
                instance_labels_group = tree_env_group.require_group("instance_labels")
                for rgba_id, label in env_labels.items():
                    instance_labels_group.attrs[str(rgba_id)] = label
        return

    # def _flush_buffers(self):
    #     # This method can be used to implement any additional flushing logic if needed
    #     pass
