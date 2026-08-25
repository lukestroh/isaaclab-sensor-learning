#!/usr/bin/env python3
from dataclasses import dataclass

import h5py as h5
import numpy as np
import plotly.graph_objects as go
import scipy.spatial as sps
from plotly import data

# from pybullet_tree_sim.sensors.sensor import st
# from pybullet_tree_sim.tree_metadata import Face
# from scipy.constants import c
import os
import ast

# from sensor_learning.plot import evaluation_plots as eval_plots
# from sensor_learning.utils import pose_generator
import pxr
import cv2
import torch

from isaaclab.utils import math as math_utils


@dataclass
class ObservationConfig:
    # min_distance: float  # meters, get as input from z_near
    # max_distance: float  # meters, get as input from z_far
    max_incidence_angle: float = 45.0  # degrees
    pose_search_radius: float = 0.1  # radius for finding neighboring poses in kd-tree search
    # length_bin_size: float = 0.05  # meters, size of bins for distance histogram
    # theta_bin_size: float = 5.0  # degrees, size of bins for angle histogram


@dataclass
class GainConfig:
    saturation_lambda: float = 1.0  # controls how fast gain saturates
    n_midpoint: int = 5  # observations beyond n_max add little value
    steepness: float = 2.0  # controls steepness of gain curve around n_midpoint
    branch_gains = {
        "trunk": 1.0,
        "branch": 5.0,
        "spur": 10.0,
    }


class BinnedTree:
    def __init__(
        self,
        n_envs: int,
        unique_instances: dict,
        n_length_bins: int = 20,
        n_theta_bins: int = 36,
        uv_bin_img: str = os.path.join(os.path.dirname(__file__), "uv.png"),
        device: str = "cuda:0",
    ):
        self.uv_bin_img = uv_bin_img
        self.uv_img = cv2.imread(self.uv_bin_img, cv2.IMREAD_UNCHANGED)
        self.device = device

        self.n_length_bins = n_length_bins
        self.n_theta_bins = n_theta_bins

        self.rgb_bins = self._create_rgb_bins()
        # print(self.rgb_bins.shape)
        # print(self.rgb_bins.reshape(-1, 2).shape)
        # print(np.unique(self.rgb_bins.reshape(-1, 2), axis=0).shape)
        self.rgb_to_bin_lut = self._create_rgb_to_bin_lut()
        # print(self.rgb_to_bin_lut)

        self.instance_ids = {}  # env_id -> 1D int64 tensor of instance ids
        self.instance_idx_map = {}

        self.unique_instances = unique_instances
        # for env_id, ids in self.unique_instances.items():
        #     ids_t = torch.tensor([ast.literal_eval(x) for x in ids], dtype=torch.int64, device=self.device)
        #     self.instance_ids[env_id] = ids_t
        #     print(self.instance_ids[env_id])

        #     map_t = -torch.ones(, dtype=torch.int64, device=self.device)
        #     map_t[ids_t] = torch.arange(ids_t.shape[0], dtype=torch.int64, device=self.device)
        #     self.instance_idx_map[env_id] = map_t
        # self.instance_to_idx = {
        #     f"env_{i}": {str(instance): idx for idx, instance in enumerate(unique_instances[f"env_{i}"])}
        #     for i in range(n_envs)
        # }

        # self.bins = {f"{instance}": np.zeros((self.n_length_bins, self.n_theta_bins), dtype=np.int32) for instance in unique_instances}
        # self.instance_to_idx = {str(instance): idx for idx, instance in enumerate(unique_instances)}
        self.bin_counts = {
            f"env_{i}": torch.zeros(
                (len(unique_instances[f"env_{i}"]), self.n_length_bins, self.n_theta_bins),
                dtype=torch.int32,
                device=self.device,
            )
            for i in range(n_envs)
        }
        # self.bin_scores = {
        #     f"env_{i}": torch.zeros(
        #         (len(unique_instances[f"env_{i}"]), self.n_length_bins, self.n_theta_bins),
        #         dtype=torch.float32,
        #         device=self.device,
        #     )
        #     for i in range(n_envs)
        # }
        return

    def _create_rgb_bins(self):
        h, w = self.uv_img.shape[:2]

        length_edges = np.linspace(0, h, self.n_length_bins + 1, dtype=int)
        theta_edges = np.linspace(0, w, self.n_theta_bins + 1, dtype=int)

        pixel_bins = np.empty((h, w, 2), dtype=np.uint16)

        for i in range(self.n_length_bins):
            y0, y1 = length_edges[i], length_edges[i + 1]

            for j in range(self.n_theta_bins):
                x0, x1 = theta_edges[j], theta_edges[j + 1]

                pixel_bins[y0:y1, x0:x1] = (i, j)
        return pixel_bins

    def _create_rgb_to_bin_lut(self):
        """
        Create a lookup table (LUT) that maps RGB values to bin indices based on the provided UV image. The LUT is a 4D tensor where the first three dimensions correspond to the RGB values (0-255) and the last dimension contains the corresponding bin indices (length_bin, theta_bin). The LUT is initialized with -1 for all entries, indicating that no bin is assigned

        NOTE: keeps whatever channel order cv2 gave you (BGR if 3-channel, BGRA if 4).
        Whatever order you use here MUST match the order of the batch tensor you query with later.
        """
        if self.uv_bin_img is None:
            raise ValueError("UV bin img is None.")
        else:
            # cv2.imread returns BGR order; convert to RGB so the LUT matches sensor RGB order
            img = self.uv_img[..., :3][..., ::-1].astype(np.int64)  # drop alpha if present and reorder to RGB

            lut = torch.full((256, 256, 256, 2), -1, dtype=torch.int16)

            c0 = torch.from_numpy(img[..., 0].reshape(-1))
            c1 = torch.from_numpy(img[..., 1].reshape(-1))
            c2 = torch.from_numpy(img[..., 2].reshape(-1))
            bins_flat = torch.from_numpy(self.rgb_bins.reshape(-1, 2).astype(np.int16))

            lut[c0, c1, c2] = bins_flat
            return lut.to(self.device)

    def rgb_to_bin(self, rgb_batch: torch.Tensor) -> torch.Tensor:
        """
        Convert an RGB tensor to bin indices using the precomputed LUT.

        :param rgb_tensor: A tensor of shape (..., 3) containing RGB values in the range [0, 255].
        :type rgb_tensor: torch.Tensor
        :return: A tensor of shape (..., 2) containing the corresponding bin indices.
        :rtype: torch.Tensor
        """
        # Accept either float [0..1] or int [0..255] RGB inputs. Normalize to integer 0-255.
        rgb = rgb_batch
        if torch.is_floating_point(rgb):
            # if values are normalized to [0,1], scale up
            try:
                maxv = float(rgb.max())
            except Exception:
                maxv = None
            if maxv is not None and maxv <= 1.01:
                rgb = (rgb * 255.0).round()

        rgb = rgb.clamp(0, 255).long()

        # Use advanced indexing to get the bin indices from the LUT
        bin_indices = self.rgb_to_bin_lut[rgb[..., 0], rgb[..., 1], rgb[..., 2]]
        return bin_indices


class CoverageEvaluator:
    def __init__(
        self,
        trial_data: dict,
        observation_config: ObservationConfig,
        gain_config: GainConfig,
        device: str = "cuda:0",
        n_envs: int = 1,
    ) -> None:
        self.device = device
        self.n_envs = n_envs
        self.poses = torch.tensor(
            trial_data["poses"]["actual_poses"]["eef_poses"], dtype=torch.float32, device=self.device
        )
        self.trial_data = trial_data

        # print(self.kd_tree)
        self.observation_config = observation_config
        self.gain_config = gain_config

        # Unique instances (by sensor)
        for i in range(self.n_envs):
            self.unique_instances = (
                {  # NOTE: This only shows unique instances that have BEEN SEEN. NEED TO REDO, GET FROM TREE INFO
                    f"env_{i}": trial_data["trees"][f"env_{i}"]["instance_labels"]["attrs"] for i in range(self.n_envs)
                }
            )

        # Unique semantic classes (by sensor)
        self.unique_semantics = trial_data["trees"]["semantic_labels"]["attrs"]

        # Binned tree for mapping RGB values to bins
        self.binned_tree = BinnedTree(unique_instances=self.unique_instances, n_envs=self.n_envs, device=self.device)
        return

    def _filter_pose_candidates(
        self, env_ids: torch.Tensor, closest_idxs: torch.Tensor, trial_data_dict: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate env IDs and keep only candidate rows for requested envs and non-skipped poses."""
        if env_ids.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=self.device), torch.empty(
                (0, 2), dtype=torch.long, device=self.device
            )

        if env_ids.max() >= self.n_envs or env_ids.min() < 0:
            raise ValueError(f"Invalid env_ids: {env_ids}. All envs must be in range [0, {self.n_envs - 1}].")

        env_ids = env_ids.to(device=self.device).reshape(-1)
        skipped_poses = torch.tensor(trial_data_dict["poses"]["actual_poses"]["skipped_poses"], device=self.device)
        filtered = closest_idxs[~torch.isin(closest_idxs[:, 1], skipped_poses)]
        filtered = filtered[torch.isin(filtered[:, 0], env_ids)]
        return env_ids, filtered

    def _build_instance_to_idx_by_env(self, selected_envs: list[int]) -> dict:
        """Create per-env mapping from raw instance label values to contiguous instance indices."""
        instance_to_idx_by_env = {}
        for i in selected_envs:
            env_key = f"env_{i}"
            env_instances = self.unique_instances[env_key]
            instance_to_idx = {}

            for idx, raw_instance in enumerate(env_instances):
                parsed_instance = raw_instance
                if isinstance(raw_instance, str):
                    try:
                        parsed_instance = ast.literal_eval(raw_instance)
                    except (ValueError, SyntaxError):
                        parsed_instance = raw_instance

                if isinstance(parsed_instance, np.ndarray):
                    if parsed_instance.size == 1:
                        instance_to_idx[int(parsed_instance.item())] = idx
                    continue

                if isinstance(parsed_instance, (int, np.integer)):
                    instance_to_idx[int(parsed_instance)] = idx
                elif isinstance(parsed_instance, (list, tuple)) and len(parsed_instance) > 0:
                    first_val = parsed_instance[0]
                    if isinstance(first_val, (int, np.integer)):
                        instance_to_idx[int(first_val)] = idx

            instance_to_idx_by_env[env_key] = instance_to_idx

        return instance_to_idx_by_env

    def _build_masked_dict(
        self, selected_envs: list[int], closest_idxs_env: torch.Tensor, trial_data_dict: dict
    ) -> dict:
        """
        Build per-env/per-sensor masked RGB, instance segmentation, semantic segmentation, and validity tensors.

        :param selected_envs: Environment IDs to keep.
        :param closest_idxs_env: Candidate (env_id, pose_id) pairs filtered to selected_envs.
        :param trial_data_dict: Trial metadata dictionary.
        :return: Nested dict keyed by env and sensor name.
        """
        masked_dict = {f"env_{i}": {} for i in selected_envs}
        sensor_data_dict = trial_data_dict["sensors"]
        env_idxs = closest_idxs_env[:, 0]
        pose_idxs = closest_idxs_env[:, 1]

        for sensor_name, sensor_data in sensor_data_dict.items():
            sensor_height = sensor_data["attrs"]["height"]
            sensor_width = sensor_data["attrs"]["width"]

            sensor_pose_data = torch.tensor(sensor_data["pose"], device=self.device)[env_idxs, pose_idxs].flatten(
                start_dim=0, end_dim=-2
            )

            sensor_orientations = sensor_pose_data[:, 3:7]
            sensor_orientations_matrix = math_utils.matrix_from_quat(sensor_orientations)
            sensor_orientation_v = sensor_orientations_matrix @ torch.tensor([[0.0, 0.0, 1.0]], device=self.device).T
            sensor_orientation_v = (
                sensor_orientation_v.transpose(1, 2)
                .reshape(-1, 3)
                .repeat_interleave(sensor_height * sensor_width, dim=0)
            )

            normals_data = torch.tensor(sensor_data["normals"] * -1, device=self.device)[env_idxs, pose_idxs]
            flattened_normals = normals_data.flatten(start_dim=0, end_dim=-2)
            angles = torch.acos((flattened_normals * sensor_orientation_v).sum(dim=-1))
            normals_mask = angles <= np.radians(self.observation_config.max_incidence_angle)

            depth_data = torch.tensor(sensor_data["depth"], device=self.device)[env_idxs, pose_idxs].flatten()
            depth_mask = (depth_data >= sensor_data["attrs"]["z_near"]) & (depth_data < sensor_data["attrs"]["z_far"])

            rgb_data = torch.tensor(sensor_data["rgb"], device=self.device)[env_idxs, pose_idxs].flatten(
                start_dim=0, end_dim=-2
            )
            instance_data = torch.tensor(sensor_data["instance_segmentation"], device=self.device)[
                env_idxs, pose_idxs
            ].reshape(-1, 1)
            semantic_data = torch.tensor(sensor_data["semantic_segmentation"], device=self.device)[
                env_idxs, pose_idxs
            ].reshape(-1, 1)

            permissible_mask = (normals_mask & depth_mask).unsqueeze(-1)
            rgb_masked = (rgb_data * permissible_mask).reshape(-1, sensor_height, sensor_width, 3)
            instance_masked = (instance_data * permissible_mask).reshape(-1, sensor_height, sensor_width)
            semantic_masked = (semantic_data * permissible_mask).reshape(-1, sensor_height, sensor_width)
            valid_mask = (normals_mask & depth_mask).reshape(-1, sensor_height, sensor_width)

            for i in selected_envs:
                env_key = f"env_{i}"
                env_mask = env_idxs == i
                masked_dict[env_key][sensor_name] = {
                    "rgb": rgb_masked[env_mask],
                    "instance_segmentation": instance_masked[env_mask],
                    "semantic_segmentation": semantic_masked[env_mask],
                    "valid_mask": valid_mask[env_mask],
                }

        return masked_dict

    def _accumulate_pose_delta_from_sensor(
        self, pose_delta: torch.Tensor, sensor_data: dict, observation_idx: int, env_instance_map: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Accumulate bin counts from one sensor for one candidate pose into pose_delta.

        :param pose_delta: Dense count updates for one environment, shape (n_instances, n_length_bins, n_theta_bins).
        :param sensor_data: Sensor observation dictionary with keys "rgb", "instance_segmentation", and "valid_mask".
        :param observation_idx: Observation index inside the provided sensor batch.
        :param env_instance_map: Mapping from raw instance labels to contiguous instance indices.
        :return: Tuple of (instance_indices, length_bin_indices, theta_bin_indices) for touched bins.
        """
        rgb_pose = sensor_data["rgb"][observation_idx]
        instance_pose = sensor_data["instance_segmentation"][observation_idx].to(dtype=torch.int64)
        valid_pose = sensor_data["valid_mask"][observation_idx]

        bin_pose = self.binned_tree.rgb_to_bin(rgb_pose)

        flat_valid = valid_pose.reshape(-1)
        flat_bin_l = bin_pose[..., 0].reshape(-1).to(dtype=torch.int64)
        flat_bin_t = bin_pose[..., 1].reshape(-1).to(dtype=torch.int64)
        flat_instance = instance_pose.reshape(-1)

        valid = flat_valid & (flat_bin_l >= 0) & (flat_bin_t >= 0)
        if not torch.any(valid):
            empty = torch.empty((0,), dtype=torch.int64, device=self.device)
            return empty, empty, empty

        valid_instances = flat_instance[valid]
        valid_bin_l = flat_bin_l[valid]
        valid_bin_t = flat_bin_t[valid]

        mapped_instance_idx = torch.full_like(valid_instances, -1)
        num_instances = pose_delta.shape[0]

        direct_mask = (valid_instances >= 0) & (valid_instances < num_instances)
        mapped_instance_idx[direct_mask] = valid_instances[direct_mask]

        unresolved_mask = mapped_instance_idx < 0
        if torch.any(unresolved_mask):
            unresolved_vals = valid_instances[unresolved_mask].tolist()
            remapped = [env_instance_map[int(v)] for v in unresolved_vals] # should throw KeyError if instance not found, meaning we didn't correctly save the unique instances for this env
            mapped_instance_idx[unresolved_mask] = torch.tensor(remapped, dtype=torch.int64, device=self.device)

        mapped_valid = mapped_instance_idx >= 0
        if not torch.any(mapped_valid):
            empty = torch.empty((0,), dtype=torch.int64, device=self.device)
            return empty, empty, empty

        observed_instance_idxs = mapped_instance_idx[mapped_valid]
        observed_length_bins = valid_bin_l[mapped_valid]
        observed_theta_bins = valid_bin_t[mapped_valid]

        pose_delta.index_put_(
            (
                observed_instance_idxs,
                observed_length_bins,
                observed_theta_bins,
            ),
            torch.ones(int(mapped_valid.sum().item()), dtype=pose_delta.dtype, device=self.device),
            accumulate=True,
        )
        return observed_instance_idxs, observed_length_bins, observed_theta_bins

    def _select_best_pose_for_env(
        self,
        env_key: str,
        env_pose_candidates: torch.Tensor,
        env_sensor_data: dict,
        env_instance_map: dict,
        sensor_names: list[str],
    ) -> int | None:
        """Return the local candidate index that maximizes new occupied bins for one environment."""
        if env_pose_candidates.numel() == 0:
            return None

        num_pose_candidates = int(env_pose_candidates.shape[0])
        live_counts = self.binned_tree.bin_counts[env_key]

        best_local_idx = 0
        best_score = torch.tensor(float("-inf"), device=self.device)

        for pose_local_idx in range(num_pose_candidates):
            pose_delta = torch.zeros_like(self.binned_tree.bin_counts[env_key], device=self.device)
            semantic_views = []

            for sensor_name in sensor_names:
                sensor_data = env_sensor_data.get(sensor_name)
                if sensor_data is None:
                    continue

                self._accumulate_pose_delta_from_sensor(
                    pose_delta=pose_delta,
                    sensor_data=sensor_data,
                    observation_idx=pose_local_idx,
                    env_instance_map=env_instance_map,
                )
                semantic_views.append(sensor_data["semantic_segmentation"][pose_local_idx])

            semantic_data = torch.cat([view.reshape(-1) for view in semantic_views], dim=0) if semantic_views else None
            score = self.score_pose_delta(pose_delta=pose_delta, live_counts=live_counts, semantic_data=semantic_data)
            # print(f"Score for pose {pose_local_idx}: {score}")


            if score > best_score:
                best_score = score
                best_local_idx = pose_local_idx

        return best_local_idx

    def _semantic_weight_from_id(self, semantic_id: int) -> float:
        """
        Map a semantic class ID to a gain weight.

        :param semantic_id: Semantic class ID from the sensor output.
        :return: Gain weight for that class.
        """
        # print(semantic_id)
        semantic_label = None
        if isinstance(self.unique_semantics, dict):
            semantic_label = self.unique_semantics.get(semantic_id, self.unique_semantics.get(str(semantic_id)))
        elif isinstance(self.unique_semantics, (list, tuple)) and 0 <= semantic_id < len(self.unique_semantics):
            semantic_label = self.unique_semantics[semantic_id]

        if semantic_label is None:
            return 1.0

        semantic_name = str(semantic_label).lower()
        if "spur" in semantic_name:
            return float(self.gain_config.branch_gains["spur"])
        if "branch" in semantic_name:
            return float(self.gain_config.branch_gains["branch"])
        if "trunk" in semantic_name:
            return float(self.gain_config.branch_gains["trunk"])
        return 1.0

    def score_pose_delta(self, pose_delta: torch.Tensor, live_counts: torch.Tensor, semantic_data: torch.Tensor | None = None) -> float:
        """
        Compute a score for a candidate pose delta without mutating live bin state.

        :param pose_delta: Candidate count delta with shape (n_instances, n_length_bins, n_theta_bins).
        :param semantic_data: Optional semantic class IDs from the candidate views.
        :return: A float score for the pose delta.
        """
        if pose_delta.numel() == 0:
            return 0.0

        observed_mask = pose_delta > 0
        if not torch.any(observed_mask):
            return 0.0
        
        projected_counts = (live_counts[observed_mask] + pose_delta[observed_mask]).to(dtype=torch.float32)
        gains = self.gain_config.saturation_lambda / (
            1 + torch.exp(self.gain_config.steepness * (projected_counts - self.gain_config.n_midpoint))
        )
        score = float(gains.sum().item())
        # print(f"semantic shape: {semantic_data.shape if semantic_data is not None else None}")
        # print(f"pose_delta shape: {pose_delta.shape}")
        if semantic_data is None or semantic_data.numel() == 0:
            return score

        semantic_ids = semantic_data.reshape(-1).to(dtype=torch.int64)
        semantic_ids = semantic_ids[semantic_ids >= 0]
        if semantic_ids.numel() == 0:
            return score

        unique_semantic_ids, semantic_counts = torch.unique(semantic_ids, return_counts=True)
        semantic_weight = 0.0
        total_semantic_count = float(semantic_ids.numel())
        for semantic_id, count in zip(unique_semantic_ids.tolist(), semantic_counts.tolist()):
            semantic_weight += self._semantic_weight_from_id(int(semantic_id)) * float(count)

        return score * (semantic_weight / total_semantic_count)

    def get_closest_pose_idxs(self, curr_poses: torch.Tensor):
        """Get closes poses to the current pose within a specified radius"""
        dists = (self.poses[:, :, :3] - curr_poses[:, None, :3]).norm(dim=-1)
        mask = dists <= self.observation_config.pose_search_radius

        search_count = 1
        while not torch.any(mask):
            mask = dists <= self.observation_config.pose_search_radius * 1.1 * search_count
            search_count += 1

        idxs = torch.nonzero(mask, as_tuple=False)

        return idxs

    def compute_env_info_gain_from_pose_idxs(
        self, env_ids: torch.Tensor, closest_idxs: torch.Tensor, trial_data_dict: dict
    ) -> torch.Tensor:
        """Evaluate candidate poses per env without mutating global bin state and return best (env, pose) pairs."""
        env_ids, closest_idxs_env = self._filter_pose_candidates(env_ids, closest_idxs, trial_data_dict)
        if closest_idxs_env.numel() == 0:
            return torch.empty((0, 2), dtype=torch.long, device=self.device)

        selected_envs = env_ids.tolist()
        pose_candidates_by_env = {f"env_{i}": closest_idxs_env[closest_idxs_env[:, 0] == i, 1] for i in selected_envs}
        instance_to_idx_by_env = self._build_instance_to_idx_by_env(selected_envs)
        masked_dict = self._build_masked_dict(selected_envs, closest_idxs_env, trial_data_dict)
        sensor_names = list(trial_data_dict["sensors"].keys())

        best_pose_by_env = []
        for i in selected_envs:
            env_key = f"env_{i}"
            best_local_idx = self._select_best_pose_for_env(
                env_key=env_key,
                env_pose_candidates=pose_candidates_by_env[env_key],
                env_sensor_data=masked_dict[env_key],
                env_instance_map=instance_to_idx_by_env[env_key],
                sensor_names=sensor_names,
            )

            if best_local_idx is None:
                continue

            best_pose_by_env.append(
                torch.stack(
                    [
                        torch.tensor(i, dtype=torch.long, device=self.device),
                        pose_candidates_by_env[env_key][best_local_idx].to(dtype=torch.long),
                    ]
                )
            )

        if len(best_pose_by_env) == 0:
            return torch.empty((0, 2), dtype=torch.long, device=self.device)

        return torch.stack(best_pose_by_env, dim=0)

    def _compress_bin_updates(
        self, instance_idxs: torch.Tensor, length_bins: torch.Tensor, theta_bins: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compress sparse bin hits into unique bin triplets with counts.

        :param instance_idxs: Instance indices for observed pixels.
        :param length_bins: Length-bin indices for observed pixels.
        :param theta_bins: Theta-bin indices for observed pixels.
        :return: Tuple of (unique_triplets, counts) where unique_triplets has shape (N, 3).
        """
        if instance_idxs.numel() == 0:
            empty_triplets = torch.empty((0, 3), dtype=torch.int64, device=self.device)
            empty_counts = torch.empty((0,), dtype=torch.int64, device=self.device)
            return empty_triplets, empty_counts

        triplets = torch.stack((instance_idxs, length_bins, theta_bins), dim=1)
        return torch.unique(triplets, dim=0, return_counts=True)

    def _to_tensor_on_device(self, value) -> torch.Tensor:
        """
        Convert an observation value to a tensor on the evaluator device.

        :param value: Observation value which may be a torch tensor or an object with a .torch tensor.
        :return: Tensor on self.device.
        """
        if hasattr(value, "torch"):
            value = value.torch
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        return value.to(device=self.device)

    def _build_sensor_data_for_env(self, sensor_data: object, env_idx: int) -> dict | None:
        """
        Build a normalized per-env sensor_data dictionary expected by the accumulator.

        :param sensor_data: Sensor observation payload (live sensor object or already-normalized dict).
        :param env_idx: Environment index to extract.
        :return: Dict with keys rgb, instance_segmentation, valid_mask or None if required fields are missing.
        """
        if isinstance(sensor_data, dict) and "rgb" in sensor_data and "instance_segmentation" in sensor_data:
            rgb = self._to_tensor_on_device(sensor_data["rgb"])
            instance = self._to_tensor_on_device(sensor_data["instance_segmentation"])
            valid = sensor_data.get("valid_mask")
            valid = self._to_tensor_on_device(valid) if valid is not None else None
        else:
            output = getattr(sensor_data, "output", None)
            if output is None:
                return None
            if "rgb" not in output or "instance_segmentation_fast" not in output:
                return None

            rgb = self._to_tensor_on_device(output["rgb"])
            instance = self._to_tensor_on_device(output["instance_segmentation_fast"])
            valid = None
            if "depth" in output:
                depth = self._to_tensor_on_device(output["depth"]).squeeze(-1)
                valid = torch.isfinite(depth) & (depth > 0)

        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)
        if instance.dim() == 2:
            instance = instance.unsqueeze(0)

        if valid is None:
            valid = torch.ones(instance.shape, dtype=torch.bool, device=self.device)
        else:
            if valid.dim() == 2:
                valid = valid.unsqueeze(0)
            valid = valid.to(device=self.device, dtype=torch.bool)

        if env_idx >= rgb.shape[0] or env_idx >= instance.shape[0] or env_idx >= valid.shape[0]:
            return None

        rgb_env = rgb[env_idx : env_idx + 1]
        instance_env = instance[env_idx : env_idx + 1]
        valid_env = valid[env_idx : env_idx + 1]

        return {
            "rgb": rgb_env,
            "instance_segmentation": instance_env,
            "valid_mask": valid_env,
        }

    def update_bin_count_scores_from_observation(
        self, env_ids: torch.Tensor, observations: dict
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
        """
        Update live bin counts and bin scores from a batch of observations.

        :param env_ids: Environment IDs for each observation row.
        :param observations: Per-sensor observation batches keyed by sensor name.
        :return: Tuple of (info_gains, dense_bin_updates_by_env, sparse_bin_updates_by_env).
        """
        if env_ids.numel() == 0:
            return torch.empty((0,), dtype=torch.float32, device=self.device), {}, {}

        env_ids = env_ids.to(device=self.device).reshape(-1)
        selected_envs = env_ids.unique().tolist()
        instance_to_idx_by_env = self._build_instance_to_idx_by_env(selected_envs)

        info_gains = torch.zeros(env_ids.shape[0], dtype=torch.float32, device=self.device)
        dense_bin_updates_by_env = {}
        sparse_bin_updates_by_env = {}

        for i in selected_envs:
            env_key = f"env_{i}"
            env_mask = env_ids == i

            pose_delta = torch.zeros_like(self.binned_tree.bin_counts[env_key], device=self.device)
            base_nonzero = (self.binned_tree.bin_counts[env_key] > 0).sum().to(dtype=torch.float32)

            observed_instance_list = []
            observed_length_list = []
            observed_theta_list = []

            for sensor_name, sensor_data in observations.items():
                env_sensor_data = self._build_sensor_data_for_env(sensor_data=sensor_data, env_idx=i)
                if env_sensor_data is None:
                    continue

                observed_instance, observed_length, observed_theta = self._accumulate_pose_delta_from_sensor(
                    pose_delta=pose_delta,
                    sensor_data=env_sensor_data,
                    observation_idx=0,
                    env_instance_map=instance_to_idx_by_env[env_key],
                )
                if observed_instance.numel() > 0:
                    observed_instance_list.append(observed_instance)
                    observed_length_list.append(observed_length)
                    observed_theta_list.append(observed_theta)

            self.binned_tree.bin_counts[env_key] += pose_delta # element-wise addition to update the live bin counts
            
            updated_nonzero = (self.binned_tree.bin_counts[env_key] > 0).sum().to(dtype=torch.float32)
            info_gains[env_mask] = updated_nonzero - base_nonzero

            dense_bin_updates_by_env[env_key] = pose_delta

            if len(observed_instance_list) == 0:
                sparse_bin_updates_by_env[env_key] = {
                    "instance_idx": torch.empty((0,), dtype=torch.int64, device=self.device),
                    "length_bin": torch.empty((0,), dtype=torch.int64, device=self.device),
                    "theta_bin": torch.empty((0,), dtype=torch.int64, device=self.device),
                    "count_delta": torch.empty((0,), dtype=torch.int64, device=self.device),
                }
            else:
                all_instance = torch.cat(observed_instance_list)
                all_length = torch.cat(observed_length_list)
                all_theta = torch.cat(observed_theta_list)
                unique_triplets, counts = self._compress_bin_updates(all_instance, all_length, all_theta)
                sparse_bin_updates_by_env[env_key] = {
                    "instance_idx": unique_triplets[:, 0],
                    "length_bin": unique_triplets[:, 1],
                    "theta_bin": unique_triplets[:, 2],
                    "count_delta": counts,
                }

        return info_gains, dense_bin_updates_by_env, sparse_bin_updates_by_env


if __name__ == "__main__":

    def h5_to_dict(node) -> dict:
        """
        Convert an h5py File/Group/Dataset object to a dictionary, preserving attrs.
        """
        if hasattr(node, "keys"):
            # Group node — recurse into children, then attach this group's own attrs
            result = {key: h5_to_dict(node[key]) for key in node.keys()}
            attrs = {attr: node.attrs[attr] for attr in node.attrs}
            if attrs:
                result["attrs"] = attrs
            return result
        elif hasattr(node, "read_direct"):
            # Dataset node — read into numpy array
            value = node[()]  # use [()] instead of [:] — [:] fails on scalar datasets
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            elif hasattr(value, "dtype") and value.dtype.kind == "S":
                value = value.astype(str)
            attrs = {attr: node.attrs[attr] for attr in node.attrs}
            return {"data": value, "attrs": attrs} if attrs else value
        else:
            return node

    # from isaaclab_sensor_learning.utils import h5py_utils

    # test_h5_file = "/home/luke/IsaacLab/source/isaaclab-sensor-learning/data/SL-PDC_20260730_133953.h5"
    # test_h5_file = "/home/luke/IsaacLab/source/isaaclab-sensor-learning/data/SL-PDC_20260806_165340.h5"
    test_h5_file = "/home/luke/IsaacLab/source/isaaclab-sensor-learning/data/data_collection/SL-PDC_20260810_145937.h5"

    with h5.File(test_h5_file, "r") as f:
        trial_dict = h5_to_dict(f)

    trial_name = trial_dict["attrs"]["trial_name"]
    num_envs = trial_dict["attrs"]["num_envs"]

    start_pose_w = torch.tensor(
        [
            [2.0977, 1.0163, 2.0615, -0.7144, -0.0093, 0.0095, 0.6996],
            [-2.9023, 1.0163, 2.0615, -0.7144, -0.0093, 0.0095, 0.6996],
        ],
        dtype=torch.float32,
        device="cuda:0",
    )

    coverage_evaluator = CoverageEvaluator(
        n_envs=num_envs,
        trial_data=trial_dict,
        observation_config=ObservationConfig(),
        gain_config=GainConfig(),
    )

    closest_idxs = coverage_evaluator.get_closest_pose_idxs(curr_poses=start_pose_w)
    # print(f"Closest pose indices: {closest_idxs.shape}, {closest_idxs}")

    compute_info_gain = coverage_evaluator.compute_env_info_gain_from_pose_idxs(
        env_ids=torch.tensor([0, 1], device=coverage_evaluator.device),
        closest_idxs=closest_idxs,
        trial_data_dict=trial_dict,
    )
    print(compute_info_gain)

    # info_gains = coverage_evaluator.compute_info_gain_from_pose_idxs(
    #     closest_idxs=closest_idxs, trial_data_dict=trial_dict
    # )
