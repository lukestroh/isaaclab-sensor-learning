#!/usr/bin/env python3
# from datetime import datetime as dt
import h5py as h5
import numpy as np
import os
import torch
from typing import Any


class EvaluationDataLogger:
    def __init__(self, capture_data_dict):
        self.process_type = "evaluation"
        self.capture_data_dict = capture_data_dict
        self.n_envs = capture_data_dict['n_envs']
        self.datafile_path, self.trial_name = self.get_file_path()
        # buffers
        self._timestamp_buf = []
        self._joint_position_buf = []
        self._pose_buf = []
        return

    def _get_chunk_steps(self, data_size: tuple, dtype: np.dtype, target_chunk_bytes: int = 512_000) -> int:
        row_bytes = np.prod(data_size) * np.dtype(dtype).itemsize
        chunk_steps = max(1, target_chunk_bytes // row_bytes)
        return chunk_steps

    def get_file_path(self) -> tuple[str, str]:
        """
        Get the file path for the data file.
        """
        pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        datetime_id = self.capture_data_dict["datetime_id"]

        data_dir = os.path.join(pkg_dir, "data", self.process_type)
        trial_name = f"SL-PDEVAL_{datetime_id}"
        file_path = os.path.join(data_dir, f"{trial_name}.h5")
        if not os.path.exists(data_dir):
            print(f"[INFO] Creating data directory at {data_dir}")
            os.makedirs(data_dir)
        if os.path.exists(file_path):
            print(f"[WARNING] File {file_path} already exists. It will be overwritten.")
        return file_path, trial_name

    def _save_data(self, data: np.ndarray, key: str) -> None:
        return

    def _save_metadata(self, metadata: dict) -> None:
        return

    def save_evaluation_metadata(self, evaluation_metadata: dict[str, Any]) -> None:
        with h5.File(self.datafile_path, 'w') as file:
            file.attrs['seed'] = evaluation_metadata['seed']
            file.attrs['trial_name'] = evaluation_metadata['trial_name']
            file.attrs['associated_data'] = evaluation_metadata['associated_data']
        return

    def save_timestamp_metadata(self, timestamp_metadata: dict) -> None:
        data_size = (1,)
        self.timestamp_chunk_steps = self._get_chunk_steps(data_size=data_size, dtype=np.float32)
        with h5.File(self.datafile_path, 'a') as file:
            ts_dset = file.require_dataset(
                name="timestamp",
                shape=(0,),
                maxshape=(None,),
                chunks=(self.timestamp_chunk_steps,),
                dtype=np.float32
            )
            ts_dset.attrs['physics_dt'] = timestamp_metadata['physics_dt']
            ts_dset.attrs['decimation'] = timestamp_metadata['decimation']
        return

    def save_timestamp(self, timestamp: float, last_obs: bool = False) -> None:
        self._timestamp_buf.append(timestamp)
        if (len(self._timestamp_buf) >= self.timestamp_chunk_steps) or (last_obs):
            print("FLUSH: timestamp")
            new_data = np.vstack(self._timestamp_buf)
            new_data_len = new_data.shape[0]
            with h5.File(self.datafile_path, 'a') as file:
                dset_timestamps = file['timestamp']
                row_count = dset_timestamps.shape[0]
                dset_timestamps.resize(row_count + new_data_len, axis=0)
                dset_timestamps[row_count : row_count+new_data_len] = new_data
            self._timestamp_buf.clear()
        return

    def save_joint_position_metadata(self, joint_position_metadata: dict[str, Any]) -> None:
        n_dof = joint_position_metadata['n_dof']
        data_size = (self.n_envs, n_dof)
        self.joint_position_chunk_steps = self._get_chunk_steps(data_size=data_size, dtype=np.float32)

        with h5.File(self.datafile_path, 'a') as file:
            file.require_dataset(
                name="joint_positions",
                shape=(self.n_envs, 0, n_dof),   # TODO: get num. joints dynamically
                maxshape=(self.n_envs, None, n_dof),
                chunks=(self.n_envs, self.joint_position_chunk_steps, n_dof),
                dtype=np.float32
            )
        return

    def save_joint_positions(self, joint_positions: torch.Tensor, last_obs: bool = False) -> None:
        self._joint_position_buf.append(joint_positions.detach().cpu().numpy())
        if (len(self._joint_position_buf) >= self.joint_position_chunk_steps) or (last_obs):
            print("FLUSH: joint positions")
            new_data = np.stack(self._joint_position_buf, axis=1)
            new_data_len = new_data.shape[1]
            with h5.File(self.datafile_path, 'a') as file:
                dset_joint_positions = file['joint_positions']
                row_count = dset_joint_positions.shape[1]
                dset_joint_positions.resize(row_count + new_data_len, axis=1)
                dset_joint_positions[:, row_count:row_count+new_data_len, :] = new_data
            self._joint_position_buf.clear()
        return

    def save_pose_metadata(self) -> None:
        data_size = (self.n_envs, 7) # (x, y, z, qx, qy, qz, qw)
        self.pose_chunk_steps = self._get_chunk_steps(data_size=data_size, dtype=np.float32)
        with h5.File(self.datafile_path, 'a') as file:
            file.require_dataset(
                name="pose",
                shape=(self.n_envs, 0, 7),
                maxshape=(self.n_envs, None, 7),
                chunks=(self.n_envs, self.pose_chunk_steps, 7),
                dtype=np.float32
            )
        return

    def save_poses(self, poses: torch.Tensor, last_obs: bool = False) -> None:
        self._pose_buf.append(poses.detach().cpu().numpy())
        if (len(self._pose_buf) >= self.pose_chunk_steps) or (last_obs):
            print("FLUSH: poses")
            new_data = np.stack(self._pose_buf, axis=1)
            new_data_len = new_data.shape[1]
            with h5.File(self.datafile_path, 'a') as file:
                dset_poses = file['pose']
                row_count = dset_poses.shape[1]
                dset_poses.resize(row_count + new_data_len, axis=1)
                dset_poses[:, row_count:row_count+new_data_len, :] = new_data
            self._pose_buf.clear()
        return

    def save_nbv_goal_metadata(self, nbv_goal_metadata: dict = {}) -> None:
        data_size = (1,)
        self.nbv_goal_chunk_steps = self._get_chunk_steps(data_size=data_size, dtype=np.float32)
        nbv_goal_dtype = np.dtype([
            ("timestamp", np.float32),
            ("pose", np.float32, (7,)) # (x, y, z, qx, qy, qz, qw)
        ])
        with h5.File(self.datafile_path, 'a') as file:
            group_nbv_goal = file.require_group(name="nbv_goal")
            for env_id in range(self.n_envs):
                group_nbv_goal.require_dataset(
                    name=f"{env_id}",
                    shape=(0,),
                    maxshape=(None,),
                    chunks=(self.nbv_goal_chunk_steps,),
                    dtype=nbv_goal_dtype
                )
        return

    def save_nbv_goal(self, env_id: int, timestamp: float, nbv_goal: torch.Tensor) -> None:
        """Save a goal update directly"""
        with h5.File(self.datafile_path, 'a') as file:
            group_nbv_goal = file.require_group(name="nbv_goal")
            dset_env_nbv_goal = group_nbv_goal[f"{env_id}"]
            row_count = dset_env_nbv_goal.shape[0]
            dset_env_nbv_goal.resize(row_count + 1, axis=0)
            dset_env_nbv_goal[row_count] = (timestamp, nbv_goal.detach().cpu().numpy())
        return

    def save_bin_count_metadata(self, bin_count_metadata: dict) -> None:
        return

    def save_bin_counts(self, bin_counts: torch.Tensor, last_obs: bool = False) -> None:
        raise NotImplementedError
