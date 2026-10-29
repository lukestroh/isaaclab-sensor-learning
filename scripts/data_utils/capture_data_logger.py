#!/usr/bin/env python3
from turtle import update

import h5py as h5
import os
from datetime import datetime as dt
import numpy as np
import torch
from typing import Any
from isaaclab.sensors import CameraData
import pprint as pp


class CaptureDataLogger:
    def __init__(
        self,
        process_type: str = "capture",
        compression_level: int = 9,
        datetime_id: str | None = None,
        file_path: str | None = None,
    ):
        """
        :param datetime_id: trial id used in the file name (SL-PDC_<datetime_id>.h5). Pass the trial's own id so the
            capture and evaluation files share it; defaults to the current time.
        :param file_path: explicit output file (e.g. ``<out_dir>/capture.h5`` for NSGA evaluations); overrides the
            default ``data/capture/SL-PDC_<datetime_id>.h5``.
        """
        self.process_type = process_type
        if file_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            self.datafile_path, self.trial_name = file_path, os.path.splitext(os.path.basename(file_path))[0]
        else:
            self.datafile_path, self.trial_name = self.get_file_path(datetime_id)

        # sensor data stuff
        self.sensor_obs_buf = {}
        self.sensor_chunk_size_dict = {}
        self.sensor_curr_idx_tracker = {}
        self._actual_poses_buf = []
        self._actual_poses_idx_buf = []
        self._joint_state_buf = []
        self._joint_state_idx_buf = []

        # self.row_cursor = 0
        self.flush_every = None
        self.labels_dict = {"semantic_labels": {}, "instance_labels": {}}
        self.compression_level = compression_level
        return

    def _get_chunk_steps(self, data_size: tuple, dtype: np.dtype, target_chunk_bytes=1024**2) -> int:
        row_bytes = np.prod(data_size) * np.dtype(dtype).itemsize
        chunk_steps = int(min(self.n_poses, max(1, target_chunk_bytes // row_bytes)))
        return chunk_steps

    def _make_camera_observation_buffer(self) -> dict[str, list]:
        buf = {
            "pose_w": [],
            "intrinsic_matrix": [],
            "albedo": [],
            "depth": [],
            "instance_segmentation": [],
            "semantic_segmentation": [],
            "normals": [],
        }
        return buf

    def _make_sensor_chunk_size_dict(self) -> dict[str, int]:
        _dict = {
            "pose_w": 0,
            "intrinsic_matrix": 0,
            "albedo": 0,
            "depth": 0,
            "instance_segmentation": 0,
            "semantic_segmentation": 0,
            "normals": 0,
        }
        return _dict

    def _make_sensor_idx_tracker_dict(self) -> dict[str, list]:
        _dict = {
            "pose_w": [],
            "intrinsic_matrix": [],
            "albedo": [],
            "depth": [],
            "instance_segmentation": [],
            "semantic_segmentation": [],
            "normals": [],
        }
        return _dict

    def _make_data_shapes_dict(self) -> dict[str, tuple]:
        _dict = {"pose_w": (self.n_envs, self.n_poses, 7)}
        return _dict

    def _reshape_tensor_to_numpy(self, tensors: list[torch.Tensor], shape: tuple = (0, 1)) -> np.ndarray:
        """Reshape a tensor of (n_poses, n_envs, ...) to numpy array of shape (n_envs, n_poses, ...)."""
        return torch.stack(tensors).transpose(*shape).cpu().numpy()

    def get_file_path(self, datetime_id: str | None = None) -> tuple[str, str]:
        """
        Get the file path for the data file.
        """
        pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        data_dir = os.path.join(pkg_dir, "data", self.process_type)
        trial_name = f"SL-PDC_{datetime_id or dt.now().strftime('%Y%m%d_%H%M%S')}"
        file_path = os.path.join(data_dir, f"{trial_name}.h5")
        if not os.path.exists(data_dir):
            print(f"[INFO] Creating data directory at {data_dir}")
            os.makedirs(data_dir)
        if os.path.exists(file_path):
            print(f"[WARNING] File {file_path} already exists. It will be overwritten.")
        return file_path, trial_name

    def save_trial_metadata(self, trial_metadata: dict) -> None:
        # Create labels_dict for each environment to store semantic and instance labels
        for i in range(trial_metadata["num_envs"]):
            self.labels_dict["instance_labels"][f"env_{i}"] = {}

        self.n_envs = trial_metadata["num_envs"]
        self.n_poses = trial_metadata["n_poses"]

        # Save trial metadata to HDF5 file
        with h5.File(self.datafile_path, "w") as file:
            file.attrs["datetime_id"] = trial_metadata["datetime_id"]
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
            desired_poses_group.create_dataset(
                name="eef_poses",
                data=trial_metadata["poses"],
                compression="gzip",
                compression_opts=self.compression_level,
            )

            # actual poses
            actual_poses_group = file.require_group("poses/actual_poses")
            data_shape = (self.n_envs, len(trial_metadata["poses"]), 7)
            self.actual_poses_chunk_steps = self._get_chunk_steps(data_size=(self.n_envs, 7), dtype=np.float32)
            actual_poses_group.require_dataset(
                name="eef_poses",
                shape=(self.n_envs, self.n_poses, 7),
                dtype=np.float32,
                chunks=(self.n_envs, self.actual_poses_chunk_steps, 7),
                compression="gzip",
                compression_opts=self.compression_level,
            )

            # joint states
            joint_states_group = file.require_group("joint_states")
            n_joints = len(trial_metadata["robot_joint_names"])
            data_shape = (self.n_envs, self.n_poses, len(trial_metadata["robot_joint_names"]))
            self.joint_state_chunk_steps = self._get_chunk_steps(data_size=(self.n_envs, n_joints), dtype=np.float32)
            joint_states_group.require_dataset(
                name="joint_state",
                shape=data_shape,
                chunks=(self.n_envs, self.joint_state_chunk_steps, n_joints),
                dtype=np.float32,
                compression="gzip",
                compression_opts=self.compression_level,
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

    def save_sensor_metadata(self, sensor_metadata: dict, n_poses: int) -> None:
        """I apologize to future me for this function"""
        with h5.File(self.datafile_path, "a") as file:
            for sensor_name, sensor_info in sensor_metadata.items():
                # Set up buffer dicts
                self.sensor_obs_buf[sensor_name] = self._make_camera_observation_buffer()
                self.sensor_chunk_size_dict[sensor_name] = self._make_sensor_chunk_size_dict()
                self.sensor_curr_idx_tracker[sensor_name] = self._make_sensor_idx_tracker_dict()

                # Sensor group
                sensor_group = file.require_group(f"sensors/{sensor_name}")
                for key, value in sensor_info.items():
                    sensor_group.attrs[key] = value
                sensor_group.attrs["pose_w_convention"] = "ros"  # read by NBVEvaluator to pick the optical axis

                self.sensor_chunk_size_dict[sensor_name]["pose_w"] = self._get_chunk_steps(
                    data_size=(self.n_envs, 7), dtype=np.float32
                )
                self.sensor_chunk_size_dict[sensor_name]["intrinsic_matrix"] = self._get_chunk_steps(
                    data_size=(self.n_envs, 3, 3), dtype=np.float32
                )
                self.sensor_chunk_size_dict[sensor_name]["albedo"] = self._get_chunk_steps(
                    data_size=(self.n_envs, n_poses, sensor_info["height"], sensor_info["width"], 3), dtype=np.uint8
                )
                self.sensor_chunk_size_dict[sensor_name]["depth"] = self._get_chunk_steps(
                    data_size=(self.n_envs, n_poses, sensor_info["height"], sensor_info["width"]), dtype=np.float32
                )
                self.sensor_chunk_size_dict[sensor_name]["instance_segmentation"] = self._get_chunk_steps(
                    data_size=(self.n_envs, n_poses, sensor_info["height"], sensor_info["width"]), dtype=np.float32
                )
                self.sensor_chunk_size_dict[sensor_name]["semantic_segmentation"] = self._get_chunk_steps(
                    data_size=(self.n_envs, n_poses, sensor_info["height"], sensor_info["width"]), dtype=np.float32
                )
                self.sensor_chunk_size_dict[sensor_name]["normals"] = self._get_chunk_steps(
                    data_size=(self.n_envs, n_poses, sensor_info["height"], sensor_info["width"], 3), dtype=np.float32
                )
                sensor_group.require_dataset(
                    name="pose_w",
                    shape=(self.n_envs, n_poses, 7),
                    chunks=(self.n_envs, self.sensor_chunk_size_dict[sensor_name]["pose_w"], 7),
                    dtype=np.float32,
                    compression="gzip",
                    compression_opts=self.compression_level,
                )
                sensor_group.require_dataset(
                    name="intrinsic_matrix",
                    shape=(self.n_envs, n_poses, 3, 3),
                    chunks=(self.n_envs, self.sensor_chunk_size_dict[sensor_name]["intrinsic_matrix"], 3, 3),
                    dtype=np.float32,
                    compression="gzip",
                    compression_opts=self.compression_level,
                )
                # Unlit UV texture colors (RGB of the RGBA albedo output); the NBV evaluator bins from these
                sensor_group.require_dataset(
                    name="albedo",
                    shape=(self.n_envs, n_poses, sensor_info["height"], sensor_info["width"], 3),
                    chunks=(
                        self.n_envs,
                        self.sensor_chunk_size_dict[sensor_name]["albedo"],
                        sensor_info["height"],
                        sensor_info["width"],
                        3,
                    ),
                    dtype=np.uint8,
                    compression="gzip",
                    compression_opts=self.compression_level,
                )
                sensor_group.require_dataset(
                    name="depth",
                    shape=(self.n_envs, n_poses, sensor_info["height"], sensor_info["width"]),
                    chunks=(
                        self.n_envs,
                        self.sensor_chunk_size_dict[sensor_name]["depth"],
                        sensor_info["height"],
                        sensor_info["width"],
                    ),
                    dtype=np.float32,
                    compression="gzip",
                    compression_opts=self.compression_level,
                )
                sensor_group.require_dataset(
                    name="instance_segmentation",
                    shape=(self.n_envs, n_poses, sensor_info["height"], sensor_info["width"]),
                    chunks=(
                        self.n_envs,
                        self.sensor_chunk_size_dict[sensor_name]["instance_segmentation"],
                        sensor_info["height"],
                        sensor_info["width"],
                    ),
                    dtype=np.float32,
                    compression="gzip",
                    compression_opts=self.compression_level,
                )
                sensor_group.require_dataset(
                    name="semantic_segmentation",
                    shape=(self.n_envs, n_poses, sensor_info["height"], sensor_info["width"]),
                    chunks=(
                        self.n_envs,
                        self.sensor_chunk_size_dict[sensor_name]["semantic_segmentation"],
                        sensor_info["height"],
                        sensor_info["width"],
                    ),
                    dtype=np.float32,
                    compression="gzip",
                    compression_opts=self.compression_level,
                )
                sensor_group.require_dataset(
                    name="normals",
                    shape=(self.n_envs, n_poses, sensor_info["height"], sensor_info["width"], 3),
                    chunks=(
                        self.n_envs,
                        self.sensor_chunk_size_dict[sensor_name]["normals"],
                        sensor_info["height"],
                        sensor_info["width"],
                        3,
                    ),
                    dtype=np.float32,
                    compression="gzip",
                    compression_opts=self.compression_level,
                )
        return

    def save_pose_observations(self, observations: dict[str, CameraData], idx: int, last_obs: bool = False) -> None:
        """
        Save observations and poses to HDF5 file. Flushes buffers every chunk_step observations or if last_obs is True.
        """
        # add semantic and instance labels to labels_dict
        for sensor_name, sensor_data in observations.items():
            semantic_labels_dict = sensor_data.info["semantic_segmentation"]
            instance_labels_dict = sensor_data.info["instance_segmentation"]
            for rgba_id, label in instance_labels_dict["idToLabels"].items():
                if label not in ["BACKGROUND", "UNLABELLED"]:  # Does isaac sim have a typo??
                    env_id = label.split("/")[3].split("_")[1]
                    self.labels_dict["instance_labels"][f"env_{env_id}"][rgba_id] = label

            for rgba_id, label_dict in semantic_labels_dict["idToLabels"].items():
                self.labels_dict["semantic_labels"][rgba_id] = label_dict["class"]

        # append sensor data to buffers
        for sensor_name, sensor_data in observations.items():
            # ROS convention: optical axis is +Z (matches OffsetCfg.convention = "ros" in generate_rig)
            sensor_pose_w = torch.cat([sensor_data.pos_w.torch, sensor_data.quat_w_ros.torch], axis=1)
            self.sensor_obs_buf[sensor_name]["pose_w"].append(sensor_pose_w.detach())
            self.sensor_obs_buf[sensor_name]["intrinsic_matrix"].append(sensor_data.intrinsic_matrices.torch.detach())
            self.sensor_obs_buf[sensor_name]["albedo"].append(sensor_data.output["albedo"].torch[..., :3].detach())
            self.sensor_obs_buf[sensor_name]["depth"].append(sensor_data.output["depth"].torch.detach())
            self.sensor_obs_buf[sensor_name]["instance_segmentation"].append(
                sensor_data.output["instance_segmentation"].torch.detach()
            )
            self.sensor_obs_buf[sensor_name]["semantic_segmentation"].append(
                sensor_data.output["semantic_segmentation"].torch.detach()
            )
            self.sensor_obs_buf[sensor_name]["normals"].append(sensor_data.output["normals"].torch.detach())

        for sensor_name in observations.keys():
            for data_type, target_chunk_size in self.sensor_chunk_size_dict[sensor_name].items():
                self.sensor_curr_idx_tracker[sensor_name][data_type].append(idx)
                if (len(self.sensor_obs_buf[sensor_name][data_type]) >= target_chunk_size) or last_obs:
                    if data_type in ["depth", "instance_segmentation", "semantic_segmentation"]:
                        chunked_data = self._reshape_tensor_to_numpy(
                            self.sensor_obs_buf[sensor_name][data_type]
                        ).squeeze(-1)
                    else:
                        chunked_data = self._reshape_tensor_to_numpy(self.sensor_obs_buf[sensor_name][data_type])

                    start_idx = self.sensor_curr_idx_tracker[sensor_name][data_type][0]
                    end_idx = self.sensor_curr_idx_tracker[sensor_name][data_type][-1] + 1  # +1 because slicing

                    # Save the buffered observations
                    with h5.File(self.datafile_path, "a") as file:
                        sensor_group = file.require_group(f"sensors/{sensor_name}")
                        sensor_group[data_type][:, start_idx:end_idx, :] = chunked_data
                    self.sensor_curr_idx_tracker[sensor_name][data_type].clear()
                    self.sensor_obs_buf[sensor_name][data_type].clear()
        return

    def save_actual_poses(self, poses: torch.Tensor, idx: int, last_obs: bool = False) -> None:
        self._actual_poses_buf.append(poses.cpu().numpy())
        self._actual_poses_idx_buf.append(idx)

        # print("      ", self._actual_poses_buf[-1][0])

        if (len(self._actual_poses_buf) >= self.actual_poses_chunk_steps) or last_obs:
            # print("FLUSH: poses")
            start_idx = self._actual_poses_idx_buf[0]
            end_idx = self._actual_poses_idx_buf[-1] + 1
            stacked_poses = np.stack(self._actual_poses_buf, axis=1)
            with h5.File(self.datafile_path, "a") as file:
                pose_group = file.require_group("poses/actual_poses")
                dset_poses = pose_group["eef_poses"]
                dset_poses[:, start_idx:end_idx, :] = stacked_poses

            self._actual_poses_buf.clear()
            self._actual_poses_idx_buf.clear()
        return

    def save_joint_state(self, joint_state: torch.Tensor, idx: int, last_obs: bool = False) -> None:
        self._joint_state_buf.append(joint_state.cpu().numpy())
        self._joint_state_idx_buf.append(idx)
        if (len(self._joint_state_buf) >= self.joint_state_chunk_steps) or last_obs:
            # print("FLUSH: Joint state")
            start_idx = self._joint_state_idx_buf[0]
            end_idx = self._joint_state_idx_buf[-1] + 1
            stacked_joint_states = np.stack(self._joint_state_buf, axis=1)
            with h5.File(self.datafile_path, 'a') as file:
                group_joint_states = file.require_group("joint_states")
                dset_joint_states = group_joint_states["joint_state"]
                dset_joint_states[:, start_idx:end_idx, :] = stacked_joint_states

            self._joint_state_buf.clear()
            self._joint_state_idx_buf.clear()
        return

    def save_skipped_poses(self, skipped_poses: list[int]) -> None:
        """Save the list of skipped poses to the HDF5 file."""
        with h5.File(self.datafile_path, "a") as file:
            pose_group = file.require_group("poses/actual_poses")
            pose_group.create_dataset(
                name="skipped_poses",
                data=np.array(skipped_poses, dtype=np.int32),
                compression="gzip",
                compression_opts=self.compression_level,
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

    def save_scene_frames(self, env_origins: torch.Tensor, robot_base_pose_w: torch.Tensor) -> None:
        """
        Save the frames needed to relate logged poses across envs and to the trees, under ``scene_frames``:

        - ``env_origins`` (n_envs, 3): world position of each env (``InteractiveScene.env_origins``).
        - ``robot_base_pose_w`` (n_envs, 7): robot root pose in world frame, (x, y, z, qx, qy, qz, qw). The base is
          fixed, so one snapshot describes the whole run.

        Tree poses in ``trees/env_<i>`` attrs are relative to the env origin; EEF poses in the evaluation file are
        ``position - robot base position`` (world orientation).
        """
        with h5.File(self.datafile_path, "a") as file:
            group = file.require_group("scene_frames")
            for name, data in (("env_origins", env_origins), ("robot_base_pose_w", robot_base_pose_w)):
                if name in group:
                    del group[name]
                group.create_dataset(name, data=data.detach().cpu().numpy().astype(np.float32))
            group.attrs["quaternion_order"] = "xyzw"
        return
