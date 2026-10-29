#!/usr/bin/env python3
# from datetime import datetime as dt
import h5py as h5
import numpy as np
import os
import torch
from typing import Any


class EvaluationDataLogger:
    def __init__(self, capture_data_dict, compression_level: int = 9, file_path: str | None = None):
        """
        :param file_path: explicit output file (e.g. ``<out_dir>/nbv.h5`` for NSGA evaluations); defaults to
            ``data/evaluation/SL-PDEVAL_<datetime_id>.h5``.
        """
        self.process_type = "evaluation"
        self.capture_data_dict = capture_data_dict
        self.n_envs = capture_data_dict["n_envs"]
        self.datafile_path, self.trial_name = self.get_file_path(file_path)
        # buffers
        self._timestamp_buf = []
        self._joint_position_buf = []
        self._pose_buf = []

        self.compression_level = compression_level
        return

    def _get_chunk_steps(self, data_size: tuple, dtype: np.dtype, target_chunk_bytes: int = 1024**2) -> int:
        row_bytes = np.prod(data_size) * np.dtype(dtype).itemsize
        chunk_steps = int(max(1, target_chunk_bytes // row_bytes))
        return chunk_steps

    def get_file_path(self, file_path: str | None = None) -> tuple[str, str]:
        """
        Get the file path for the data file.
        """
        if file_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            return file_path, os.path.splitext(os.path.basename(file_path))[0]
        pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        datetime_id = self.capture_data_dict["datetime_id"]

        data_dir = os.path.join(pkg_dir, "data", self.process_type)
        eval_trial_name = f"SL-PDEVAL_{datetime_id}"
        file_path = os.path.join(data_dir, f"{eval_trial_name}.h5")
        if not os.path.exists(data_dir):
            print(f"[INFO] Creating data directory at {data_dir}")
            os.makedirs(data_dir)
        if os.path.exists(file_path):
            print(f"[WARNING] File {file_path} already exists. It will be overwritten.")
        return file_path, eval_trial_name

    # def _save_data(self, data: np.ndarray, key: str) -> None:
    #     return

    # def _save_metadata(self, metadata: dict) -> None:
    #     return

    def save_evaluation_metadata(self, evaluation_metadata: dict[str, Any]) -> None:
        with h5.File(self.datafile_path, "w") as file:
            file.attrs["seed"] = evaluation_metadata["seed"]
            file.attrs["datetime_id"] = evaluation_metadata["datetime_id"]
            file.attrs["associated_data"] = evaluation_metadata["associated_data"]
        return

    def save_timestamp_metadata(self, timestamp_metadata: dict) -> None:
        data_size = (1,)
        self.timestamp_chunk_steps = self._get_chunk_steps(data_size=data_size, dtype=np.float32)
        with h5.File(self.datafile_path, "a") as file:
            ts_dset = file.require_dataset(
                name="timestamp",
                shape=(0,),
                maxshape=(None,),
                chunks=(self.timestamp_chunk_steps,),
                dtype=np.float32,
                compression="gzip",
                compression_opts=self.compression_level,
            )
            ts_dset.attrs["physics_dt"] = timestamp_metadata["physics_dt"]
            ts_dset.attrs["decimation"] = timestamp_metadata["decimation"]
        return

    def save_timestamp(self, timestamp: float, last_obs: bool = False) -> None:
        self._timestamp_buf.append(timestamp)
        if (len(self._timestamp_buf) >= self.timestamp_chunk_steps) or (last_obs):
            print("FLUSH: timestamp")
            new_data = np.array(self._timestamp_buf)
            new_data_len = new_data.shape[0]
            with h5.File(self.datafile_path, "a") as file:
                dset_timestamps = file["timestamp"]
                row_count = dset_timestamps.shape[0]
                dset_timestamps.resize(row_count + new_data_len, axis=0)
                dset_timestamps[row_count : row_count + new_data_len] = new_data
            self._timestamp_buf.clear()
        return

    def save_joint_position_metadata(self, joint_position_metadata: dict[str, Any]) -> None:
        n_dof = joint_position_metadata["n_dof"]
        data_size = (self.n_envs, n_dof)
        self.joint_position_chunk_steps = self._get_chunk_steps(data_size=data_size, dtype=np.float32)

        with h5.File(self.datafile_path, "a") as file:
            file.require_dataset(
                name="joint_positions",
                shape=(self.n_envs, 0, n_dof),  # TODO: get num. joints dynamically
                maxshape=(self.n_envs, None, n_dof),
                chunks=(self.n_envs, self.joint_position_chunk_steps, n_dof),
                dtype=np.float32,
                compression="gzip",
                compression_opts=self.compression_level,
            )
        return

    def save_joint_positions(self, joint_positions: torch.Tensor, last_obs: bool = False) -> None:
        self._joint_position_buf.append(joint_positions.cpu().numpy())
        if (len(self._joint_position_buf) >= self.joint_position_chunk_steps) or (last_obs):
            print("FLUSH: joint positions")
            new_data = np.stack(self._joint_position_buf, axis=1)
            new_data_len = new_data.shape[1]
            with h5.File(self.datafile_path, "a") as file:
                dset_joint_positions = file["joint_positions"]
                row_count = dset_joint_positions.shape[1]
                dset_joint_positions.resize(row_count + new_data_len, axis=1)
                dset_joint_positions[:, row_count : row_count + new_data_len, :] = new_data
            self._joint_position_buf.clear()
        return

    def save_pose_metadata(self) -> None:
        data_size = (self.n_envs, 7)  # (x, y, z, qx, qy, qz, qw)
        self.pose_chunk_steps = self._get_chunk_steps(data_size=data_size, dtype=np.float32)
        with h5.File(self.datafile_path, "a") as file:
            file.require_dataset(
                name="pose",
                shape=(self.n_envs, 0, 7),
                maxshape=(self.n_envs, None, 7),
                chunks=(self.n_envs, self.pose_chunk_steps, 7),
                dtype=np.float32,
                compression="gzip",
                compression_opts=self.compression_level,
            )
        return

    def save_poses(self, poses: torch.Tensor, last_obs: bool = False) -> None:
        self._pose_buf.append(poses.cpu().numpy())
        if (len(self._pose_buf) >= self.pose_chunk_steps) or (last_obs):
            print("FLUSH: poses")
            new_data = np.stack(self._pose_buf, axis=1)
            new_data_len = new_data.shape[1]
            with h5.File(self.datafile_path, "a") as file:
                dset_poses = file["pose"]
                row_count = dset_poses.shape[1]
                dset_poses.resize(row_count + new_data_len, axis=1)
                dset_poses[:, row_count : row_count + new_data_len, :] = new_data
            self._pose_buf.clear()
        return

    def save_nbv_goal_metadata(self, nbv_goal_metadata: dict = {}) -> None:
        data_size = (1,)
        nbv_goal_dtype = np.dtype([("timestamp", np.float32), ("pose", np.float32, (7,))])  # (x, y, z, qx, qy, qz, qw)
        self.nbv_goal_chunk_steps = self._get_chunk_steps(data_size=data_size, dtype=nbv_goal_dtype)
        with h5.File(self.datafile_path, "a") as file:
            group_nbv_goal = file.require_group(name="nbv_goal")
            for env_id in range(self.n_envs):
                group_nbv_goal.require_dataset(
                    name=f"{env_id}",
                    shape=(0,),
                    maxshape=(None,),
                    chunks=(self.nbv_goal_chunk_steps,),
                    dtype=nbv_goal_dtype,
                    compression="gzip",
                    compression_opts=self.compression_level,
                )
        return

    def save_nbv_goal(self, env_id: int, timestamp: float, nbv_goal: torch.Tensor) -> None:
        """Save a goal update directly"""
        with h5.File(self.datafile_path, "a") as file:
            # print("FLUSH: nbv_goal")
            group_nbv_goal = file.require_group(name="nbv_goal")
            dset_env_nbv_goal = group_nbv_goal[f"{env_id}"]
            row_count = dset_env_nbv_goal.shape[0]
            dset_env_nbv_goal.resize(row_count + 1, axis=0)
            dset_env_nbv_goal[row_count] = (timestamp, nbv_goal.cpu().numpy())
        return

    def save_sparse_bin_count_metadata(self, bin_count_metadata: dict = {}) -> None:
        data_size = (1,)
        self.sparse_bin_count_dtype = np.dtype(
            [
                ("timestamp", np.float32),
                ("instance_idx", np.int32),
                ("length_bin", np.int32),
                ("theta_bin", np.int32),
                ("count_delta", np.int32),
            ]
        )
        self.sparse_bin_count_chunk_steps = self._get_chunk_steps(
            data_size=data_size, dtype=self.sparse_bin_count_dtype
        )

        with h5.File(self.datafile_path, "a") as file:
            sparse_bin_updates_group = file.require_group("sparse_bin_updates")
            for env_id in range(self.n_envs):
                sparse_bin_updates_group.require_dataset(
                    name=f"{env_id}",
                    shape=(0,),
                    maxshape=(None,),
                    chunks=(self.sparse_bin_count_chunk_steps,),
                    dtype=self.sparse_bin_count_dtype,
                    compression="gzip",
                    compression_opts=self.compression_level,
                )
        return

    def save_sparse_bin_counts(
        self, timestamp: float, sparse_bin_updates: dict[dict[str, torch.Tensor]], last_obs: bool = False
    ) -> None:
        # print("FLUSH: sparse_bin_counts")
        with h5.File(self.datafile_path, "a") as file:
            group_sparse_bin_updates = file["sparse_bin_updates"]
            for env_key, env_data in sparse_bin_updates.items():
                env_id = int(env_key.split("_")[1])
                len_data = torch.numel(env_data["instance_idx"])
                if len_data == 0:
                    continue

                rows = np.zeros(len_data, dtype=self.sparse_bin_count_dtype)
                rows["timestamp"] = timestamp  # broadcast timestamp to all rowws
                rows["instance_idx"] = env_data["instance_idx"].cpu().numpy()
                rows["length_bin"] = env_data["length_bin"].cpu().numpy()
                rows["theta_bin"] = env_data["theta_bin"].cpu().numpy()
                rows["count_delta"] = env_data["count_delta"].cpu().numpy()

                env_dset = group_sparse_bin_updates[f"{env_id}"]
                row_count = env_dset.shape[0]
                env_dset.resize(row_count + len_data, axis=0)
                env_dset[row_count : row_count + len_data] = rows
        return

    # def save_bin_counts_metadata(self, bin_count_metadata: dict) -> None:
    #     with h5.File(self.datafile_path, 'a') as file:
    #         group_bin_counts = file.require_group("bin_counts")
    #         for env_key, bin_metadata in bin_count_metadata.items():
    #             group_bin_counts.require_dataset(
    #                 name=env_key,
    #                 shape=bin_metadata["shape"],
    #                 dtype=np.float32
    #             )
    #     return

    def save_bin_counts(self, bin_counts: dict[str, torch.Tensor]) -> None:
        with h5.File(self.datafile_path, "a") as file:
            group_bin_counts = file.require_group("bin_counts")
            for env_key, env_bin_data in bin_counts.items():
                group_bin_counts.require_dataset(
                    name=env_key,
                    shape=tuple(env_bin_data.shape),
                    data=env_bin_data.cpu().numpy(),
                    dtype=np.int32,
                    compression="gzip",
                    compression_opts=self.compression_level,
                )
        return

    def save_instance_coverage(self, instance_coverage: dict[str, dict[str, dict]]) -> None:
        """
        Save NBVEvaluator.instance_coverage() output as ``instance_coverage/<env>/<class>`` groups.

        Each class group holds attrs ``n_instances`` and ``seen_ge_<k>`` (fraction of instances with >= k hits).
        """
        with h5.File(self.datafile_path, "a") as file:
            root = file.require_group("instance_coverage")
            for env_key, by_class in instance_coverage.items():
                env_group = root.require_group(env_key)
                for cls, stats in by_class.items():
                    group = env_group.require_group(cls)
                    group.attrs["n_instances"] = stats["n_instances"]
                    for k, fraction in stats["seen"].items():
                        group.attrs[f"seen_ge_{k}"] = fraction
        return

    def save_instance_extent(
        self,
        instance_extent: dict[str, dict[str, dict]],
        per_instance: dict[str, dict[int, dict[str, torch.Tensor]]],
        instance_class_ids: dict[str, torch.Tensor],
        class_names: tuple[str, ...],
        config: dict[str, Any],
    ) -> None:
        """
        Save NBVEvaluator extent metrics under ``instance_extent/``:

        - ``<env>/<class>/ge_<k>`` attrs: ``length_coverage``, ``theta_coverage``, ``continuous`` (class means).
        - ``<env>/per_instance/ge_<k>/{length_coverage, theta_coverage, longest_gap, continuous}`` datasets, indexed by
          custom instance id, plus ``<env>/per_instance/class_id`` (index into the root attr ``class_names``, -1 unknown).
        - root attrs: ``class_names`` and the continuity config (L, g, thresholds).
        """
        with h5.File(self.datafile_path, "a") as file:
            root = file.require_group("instance_extent")
            root.attrs["class_names"] = list(class_names)
            for key, value in config.items():
                if isinstance(value, dict):  # per-class thresholds
                    for cls, cls_value in value.items():
                        root.attrs[f"{key}/{cls}"] = cls_value
                else:
                    root.attrs[key] = value
            for env_key, by_class in instance_extent.items():
                env_group = root.require_group(env_key)
                for cls, by_k in by_class.items():
                    for k, stats in by_k.items():
                        group = env_group.require_group(f"{cls}/ge_{k}")
                        for name, value in stats.items():
                            group.attrs[name] = value
                inst_group = env_group.require_group("per_instance")
                inst_group.create_dataset("class_id", data=instance_class_ids[env_key].cpu().numpy())
                for k, arrays in per_instance[env_key].items():
                    k_group = inst_group.require_group(f"ge_{k}")
                    for name, values in arrays.items():
                        k_group.create_dataset(name, data=values.cpu().numpy(), compression="gzip")
        return

    def save_termination(self, rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
        """
        Save TerminationTracker.summary() as a ``termination`` table: one dataset per column, one row per env.

        ``reason`` holds TerminationReason codes; root attrs map each code to its name (``reason_<code>``) and record
        the TerminationConfig values. ``t_end`` / ``step_end`` are NBV-relative (0 at the start of the NBV loop).
        """
        from isaaclab_sensor_learning.nbv.termination import TerminationReason

        with h5.File(self.datafile_path, "a") as file:
            group = file.require_group("termination")
            for reason in TerminationReason:
                group.attrs[f"reason_{int(reason)}"] = reason.name
            for key, value in config.items():
                group.attrs[f"config/{key}"] = "none" if value is None else value  # h5py can't store None
            for column in rows[0].keys():
                values = [row[column] for row in rows]
                data = np.array(values, dtype=h5.string_dtype()) if isinstance(values[0], str) else np.array(values)
                group.create_dataset(column, data=data)
        return

    def save_run_timing(self, timing: dict[str, Any]) -> None:
        """Save wall-clock timing of the run (setup / capture / NBV / total, seconds) as ``timing`` group attrs."""
        with h5.File(self.datafile_path, "a") as file:
            group = file.require_group("timing")
            for key, value in timing.items():
                group.attrs[key] = value
        return

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
