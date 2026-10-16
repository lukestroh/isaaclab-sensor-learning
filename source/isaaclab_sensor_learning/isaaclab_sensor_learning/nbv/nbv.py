#!/usr/bin/env python3
from dataclasses import dataclass, field

import h5py as h5
import json
import numpy as np
import os
import time
import cv2
import torch

from isaaclab.utils import math as math_utils

_SKIP_INSTANCE_LABELS = {"BACKGROUND", "UNLABELLED"}


@dataclass
class ObservationConfig:
    max_incidence_angle: float = 45.0  # degrees
    pose_search_radius: float = 0.5  # radius for finding neighboring poses


@dataclass
class GainConfig:
    saturation_lambda: float = 5.0  # controls how fast gain saturates
    n_midpoint: int = 5  # observations beyond n_midpoint add little value (depending on steepness)
    steepness: float = 1.3  # controls steepness of gain curve around n_midpoint
    branch_gains: dict[str, float] = field(
        default_factory=lambda: {
            "trunk": 1.0,
            "branch": 5.0,
            "spur": 10.0,
        }
    )


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
        self.rgb_to_bin_lut = self._create_rgb_to_bin_lut()
        self.unique_instances = unique_instances
        self.bin_counts = {
            f"env_{i}": torch.zeros(
                (_n_custom_ids(unique_instances[f"env_{i}"]), self.n_length_bins, self.n_theta_bins),
                dtype=torch.int32,
                device=self.device,
            )
            for i in range(n_envs)
        }
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
        Create a lookup table (LUT) that maps RGB values to bin indices based on the provided UV image.

        NOTE: cv2.imread returns BGR; this converts to RGB so the LUT matches sensor RGB order.
        """
        if self.uv_bin_img is None:
            raise ValueError("UV bin img is None.")

        img = self.uv_img[..., :3][..., ::-1].astype(np.int32)
        lut = np.full((256, 256, 256, 2), -1, dtype=np.int16)
        lut[img[..., 0], img[..., 1], img[..., 2]] = self.rgb_bins
        # print(lut)
        return lut

    def rgb_to_bin(self, rgb_batch: torch.Tensor) -> torch.Tensor:
        """Convert RGB to (length_bin, theta_bin) using a CPU LUT. Avoids GPU advanced indexing."""
        rgb_np = rgb_batch.detach().cpu().numpy().astype(np.float32)
        if rgb_np.size == 0:
            return torch.empty(rgb_batch.shape[:-1] + (2,), dtype=torch.int64, device=self.device)
        rgb_np = np.nan_to_num(rgb_np, nan=0.0, posinf=0.0, neginf=0.0)
        if float(np.max(rgb_np)) <= 1.01:
            rgb_np = rgb_np * 255.0
        rgb_i = np.clip(rgb_np[..., :3], 0, 255).astype(np.int32)
        bins = self.rgb_to_bin_lut[rgb_i[..., 0], rgb_i[..., 1], rgb_i[..., 2]]
        return torch.from_numpy(np.ascontiguousarray(bins, dtype=np.int64)).to(device=self.device)


def _n_custom_ids(prim_path_to_custom_id: dict) -> int:
    """Return dense bin-row count from a {prim_path: custom_id} map."""
    if not prim_path_to_custom_id:
        return 0
    return max(int(v) for v in prim_path_to_custom_id.values()) + 1


def _as_int(value) -> int:
    if isinstance(value, np.ndarray):
        return int(value.item())
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return int(value)


def _is_custom_id_value(value) -> bool:
    if isinstance(value, (int, np.integer)):
        return True
    if isinstance(value, np.ndarray) and value.size == 1:
        return True
    return False


def _extract_id_to_labels(info: dict | None) -> dict:
    """Return instance idToLabels from a camera info dict, if present."""
    if not isinstance(info, dict):
        return {}
    instance_info = info.get("instance_segmentation_fast", {})
    if not isinstance(instance_info, dict):
        return {}
    return instance_info.get("idToLabels") or instance_info.get("idsToLabels") or {}


class NBVEvaluator:
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
        self.observation_config = observation_config
        self.gain_config = gain_config

        self.prim_path_to_custom_id = {f"env_{i}": {} for i in range(self.n_envs)}
        self.render_id_to_prim_path = {f"env_{i}": {} for i in range(self.n_envs)}
        self.h5_render_id_lut = {}
        self.instance_semantic_weight = {}

        for i in range(self.n_envs):
            env_key = f"env_{i}"
            tree_group = trial_data["trees"][env_key]
            for inst_key, inst_id in tree_group.get("attrs", {}).items():
                if not str(inst_key).startswith("/World"):
                    continue
                if not _is_custom_id_value(inst_id):
                    continue
                self.prim_path_to_custom_id[env_key][str(inst_key)] = _as_int(inst_id)

            instance_labels = tree_group.get("instance_labels", {})
            instance_label_attrs = (
                instance_labels.get("attrs", instance_labels) if isinstance(instance_labels, dict) else {}
            )
            for render_id, prim_path in instance_label_attrs.items():
                try:
                    self.render_id_to_prim_path[env_key][_as_int(render_id)] = str(prim_path)
                except (TypeError, ValueError):
                    continue

            self.h5_render_id_lut[env_key] = self._build_render_id_lut(
                env_idx=i, render_id_to_prim_path=self.render_id_to_prim_path[env_key]
            )
            self.instance_semantic_weight[env_key] = self._build_instance_semantic_weights(env_key)

        self.unique_instances = self.prim_path_to_custom_id
        self.binned_tree = BinnedTree(
            unique_instances=self.prim_path_to_custom_id, n_envs=self.n_envs, device=self.device
        )
        self.skipped_poses = torch.from_numpy(
            np.ascontiguousarray(trial_data["poses"]["actual_poses"]["skipped_poses"])
        ).to(device=self.device)
        self.sensor_tensors = {}
        for sensor_name, sensor_data in trial_data["sensors"].items():
            self.sensor_tensors[sensor_name] = {
                "pose": self._h5_to_device(sensor_data["pose_w"]),
                "rgb": self._h5_to_device(sensor_data["rgb"]),
                "depth": self._h5_to_device(sensor_data["depth"]),
                "normals": self._h5_to_device(sensor_data["normals"]),
                "instance_segmentation": self._h5_to_device(sensor_data["instance_segmentation"]),
                "attrs": sensor_data["attrs"],
            }
        self._dbg_to_tensor_calls = 0
        self._dbg_accumulate_logs = 0
        return

    def _h5_to_device(self, array) -> torch.Tensor:
        """Move an H5 numpy array to the evaluator device once. Does not wrap live observations."""
        if isinstance(array, torch.Tensor):
            return array
        return torch.from_numpy(np.ascontiguousarray(array)).to(device=self.device)

    def _semantic_weight_from_prim_path(self, prim_path: str) -> float:
        """Map a tree prim path to a class gain. Prefer leaf type (spur > branch > trunk)."""
        name = prim_path.lower()
        if "spur" in name:
            return float(self.gain_config.branch_gains["spur"])
        if "branch" in name:
            return float(self.gain_config.branch_gains["branch"])
        if "trunk" in name:
            return float(self.gain_config.branch_gains["trunk"])
        return 1.0

    def _build_instance_semantic_weights(self, env_key: str) -> torch.Tensor:
        """Build a (n_instances,) tensor of class weights indexed by custom id."""
        prim_map = self.prim_path_to_custom_id[env_key]
        n_instances = _n_custom_ids(prim_map)
        weights = torch.ones(n_instances, dtype=torch.float32, device=self.device)
        for prim_path, custom_id in prim_map.items():
            weights[int(custom_id)] = self._semantic_weight_from_prim_path(prim_path)
        return weights

    def _build_render_id_lut(self, env_idx: int, render_id_to_prim_path: dict) -> torch.Tensor:
        """
        Build a 1D LUT mapping Isaac Sim render instance IDs to custom bin-row ids.

        Entries are -1 for BACKGROUND, UNLABELLED, other-env prims, and unknown paths.
        """
        env_prefix = f"/World/envs/env_{env_idx}/"
        prim_to_custom = self.prim_path_to_custom_id[f"env_{env_idx}"]
        render_ids = []
        custom_ids = []
        for raw_id, prim_path in render_id_to_prim_path.items():
            label = str(prim_path)
            if label.upper() in _SKIP_INSTANCE_LABELS:
                continue
            if env_prefix not in label:
                continue
            custom_id = prim_to_custom.get(label)
            if custom_id is None:
                continue
            render_ids.append(_as_int(raw_id))
            custom_ids.append(int(custom_id))

        if len(render_ids) == 0:
            return torch.full((1,), -1, dtype=torch.int64, device=self.device)

        lut = torch.full((max(render_ids) + 1,), -1, dtype=torch.int64, device=self.device)
        lut[torch.tensor(render_ids, dtype=torch.int64, device=self.device)] = torch.tensor(
            custom_ids, dtype=torch.int64, device=self.device
        )
        return lut

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
        filtered = closest_idxs[~torch.isin(closest_idxs[:, 1], self.skipped_poses)]
        filtered = filtered[torch.isin(filtered[:, 0], env_ids)]
        return env_ids, filtered

    def _build_masked_dict(
        self, selected_envs: list[int], closest_idxs_env: torch.Tensor, trial_data_dict: dict
    ) -> dict:
        """
        Build per-env/per-sensor masked RGB, instance segmentation, and validity tensors.

        :param selected_envs: Environment IDs to keep.
        :param closest_idxs_env: Candidate (env_id, pose_id) pairs filtered to selected_envs.
        :param trial_data_dict: Trial metadata dictionary.
        :return: Nested dict keyed by env and sensor name.
        """
        masked_dict = {f"env_{i}": {} for i in selected_envs}
        env_idxs = closest_idxs_env[:, 0]
        pose_idxs = closest_idxs_env[:, 1]

        for sensor_name, sensor_t in self.sensor_tensors.items():
            sensor_height = sensor_t["attrs"]["height"]
            sensor_width = sensor_t["attrs"]["width"]

            sensor_pose_data = sensor_t["pose"][env_idxs, pose_idxs].flatten(start_dim=0, end_dim=-2)

            sensor_orientations = sensor_pose_data[:, 3:7]
            sensor_orientations_matrix = math_utils.matrix_from_quat(sensor_orientations)
            sensor_orientation_v = sensor_orientations_matrix @ torch.tensor([[0.0, 0.0, 1.0]], device=self.device).T
            sensor_orientation_v = (
                sensor_orientation_v.transpose(1, 2)
                .reshape(-1, 3)
                .repeat_interleave(sensor_height * sensor_width, dim=0)
            )

            normals_data = sensor_t["normals"][env_idxs, pose_idxs] * -1
            flattened_normals = normals_data.flatten(start_dim=0, end_dim=-2)
            cosang = (flattened_normals * sensor_orientation_v).sum(dim=-1).clamp(-1.0, 1.0)
            angles = torch.acos(cosang)
            normals_mask = angles <= np.radians(self.observation_config.max_incidence_angle)

            depth_data = sensor_t["depth"][env_idxs, pose_idxs].flatten()
            depth_mask = (depth_data >= sensor_t["attrs"]["z_near"]) & (depth_data < sensor_t["attrs"]["z_far"])

            rgb_data = sensor_t["rgb"][env_idxs, pose_idxs].flatten(start_dim=0, end_dim=-2)
            instance_data = sensor_t["instance_segmentation"][env_idxs, pose_idxs].reshape(-1, 1)

            permissible_mask = (normals_mask & depth_mask).unsqueeze(-1)
            rgb_masked = (rgb_data * permissible_mask).reshape(-1, sensor_height, sensor_width, 3)
            instance_masked = (instance_data * permissible_mask).reshape(-1, sensor_height, sensor_width)
            valid_mask = (normals_mask & depth_mask).reshape(-1, sensor_height, sensor_width)

            for i in selected_envs:
                env_key = f"env_{i}"
                env_mask = env_idxs == i
                masked_dict[env_key][sensor_name] = {
                    "rgb": rgb_masked[env_mask],
                    "instance_segmentation": instance_masked[env_mask],
                    "valid_mask": valid_mask[env_mask],
                }

        return masked_dict

    def _map_render_ids_to_custom_ids(self, render_ids: torch.Tensor, render_id_lut: torch.Tensor) -> torch.Tensor:
        """Vectorized map from Isaac Sim instance IDs to custom bin-row ids. Unmapped values are -1.
        TODO: make sure this is working correctly."""
        mapped = torch.full_like(render_ids, -1)
        in_range = (render_ids >= 0) & (render_ids < render_id_lut.numel())
        if torch.any(in_range):
            mapped[in_range] = render_id_lut[render_ids[in_range]]
        return mapped

    def _accumulate_pose_delta_from_sensor(
        self, pose_delta: torch.Tensor, sensor_data: dict, observation_idx: int, render_id_lut: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Accumulate bin counts from one sensor for one candidate pose into pose_delta.

        Does not mutate live ``bin_counts``. Render instance IDs are mapped through
        ``render_id_lut`` (render id -> custom id) before indexing bins.

        :param pose_delta: Dense count updates for one environment, shape (n_instances, n_length_bins, n_theta_bins).
        :param sensor_data: Sensor observation dictionary with keys "rgb", "instance_segmentation", and "valid_mask".
        :param observation_idx: Observation index inside the provided sensor batch.
        :param render_id_lut: 1D tensor mapping render instance IDs to custom ids.
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

        mapped_instance_idx = self._map_render_ids_to_custom_ids(flat_instance[valid], render_id_lut)
        mapped_valid = mapped_instance_idx >= 0
        if not torch.any(mapped_valid):
            empty = torch.empty((0,), dtype=torch.int64, device=self.device)
            return empty, empty, empty

        observed_instance_idxs = mapped_instance_idx[mapped_valid]
        observed_length_bins = flat_bin_l[valid][mapped_valid]
        observed_theta_bins = flat_bin_t[valid][mapped_valid]

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
        render_id_lut: torch.Tensor,
        sensor_names: list[str],
    ) -> int | None:
        """Return the local candidate index that maximizes predicted coverage gain for one environment."""
        if env_pose_candidates.numel() == 0:
            return None

        num_pose_candidates = int(env_pose_candidates.shape[0])
        live_counts = self.binned_tree.bin_counts[env_key]
        instance_weights = self.instance_semantic_weight[env_key]

        best_local_idx = 0
        best_score = torch.tensor(float("-inf"), device=self.device)

        for pose_local_idx in range(num_pose_candidates):
            pose_delta = torch.zeros_like(live_counts, device=self.device)

            for sensor_name in sensor_names:
                sensor_data = env_sensor_data.get(sensor_name)
                if sensor_data is None:
                    continue
                self._accumulate_pose_delta_from_sensor(
                    pose_delta=pose_delta,
                    sensor_data=sensor_data,
                    observation_idx=pose_local_idx,
                    render_id_lut=render_id_lut,
                )

            score = self.score_pose_delta(
                pose_delta=pose_delta, live_counts=live_counts, instance_weights=instance_weights
            )
            # print(score)
            if score > best_score:
                best_score = score
                best_local_idx = pose_local_idx

        return best_local_idx

    def score_pose_delta(
        self, pose_delta: torch.Tensor, live_counts: torch.Tensor, instance_weights: torch.Tensor
    ) -> float:
        """
        Compute a score for a candidate pose delta without mutating live bin state.

        Predicted bin gains from all sensors (already summed into ``pose_delta``) are
        weighted by each instance's semantic class.

        :param pose_delta: Candidate count delta with shape (n_instances, n_length_bins, n_theta_bins).
        :param live_counts: Current bin counts with the same shape as ``pose_delta``.
        :param instance_weights: Per-instance class weights, shape (n_instances,).
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
        # print(gains)
        # print("\nprojected_counts\n", projected_counts)
        # print("\nlive_counts:\n", live_counts[observed_mask])
        # print("pose_delta\n", pose_delta[observed_mask])
        # print(gains)
        instance_idxs = torch.nonzero(observed_mask, as_tuple=False)[:, 0]
        weighted_gains = gains * instance_weights[instance_idxs]
        # print(float(weighted_gains.sum().item()))
        return float(weighted_gains.sum().item())

    def get_closest_pose_idxs(self, curr_poses: torch.Tensor):
        """Get closest poses to the current pose within a specified radius."""
        dists = (self.poses[:, :, :3] - curr_poses[:, None, :3]).norm(dim=-1)
        mask = dists <= self.observation_config.pose_search_radius

        search_count = 1
        while not torch.any(mask):
            mask = dists <= self.observation_config.pose_search_radius * 1.1 * search_count
            search_count += 1

        return torch.nonzero(mask, as_tuple=False)

    def compute_env_info_gain_from_pose_idxs(
        self, env_ids: torch.Tensor, closest_idxs: torch.Tensor, trial_data_dict: dict
    ) -> torch.Tensor:
        """Evaluate candidate poses per env without mutating global bin state and return best (env, pose) pairs."""
        env_ids, closest_idxs_env = self._filter_pose_candidates(env_ids, closest_idxs, trial_data_dict)
        if closest_idxs_env.numel() == 0:
            return torch.empty((0, 2), dtype=torch.long, device=self.device)

        selected_envs = env_ids.tolist()
        # print(f"selected envs: {selected_envs}")
        pose_candidates_by_env = {f"env_{i}": closest_idxs_env[closest_idxs_env[:, 0] == i, 1] for i in selected_envs}

        # print(pose_candidates_by_env)
        # print(f"pose candidataes by env: {pose_candidates_by_env}")
        masked_dict = self._build_masked_dict(selected_envs, closest_idxs_env, trial_data_dict)
        sensor_names = list(trial_data_dict["sensors"].keys())

        best_pose_by_env = []
        for i in selected_envs:
            env_key = f"env_{i}"
            best_local_idx = self._select_best_pose_for_env(
                env_key=env_key,
                env_pose_candidates=pose_candidates_by_env[env_key],
                env_sensor_data=masked_dict[env_key],
                render_id_lut=self.h5_render_id_lut[env_key],
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

    def _observation_torch(self, value) -> torch.Tensor:
        """Return a live observation as a torch tensor without copying Warp storage."""
        if isinstance(value, torch.Tensor):
            return value
        torch_view = getattr(value, "torch", None)
        if isinstance(torch_view, torch.Tensor):
            return torch_view
        raise TypeError(f"Expected a torch.Tensor or ProxyArray, got {type(value).__name__}")

    def _prepare_sensor_batch(self, sensor_data: object) -> dict | None:
        """Slice live CameraData tensors; do not recast Warp storage."""
        id_to_labels = {}
        if isinstance(sensor_data, dict) and "rgb" in sensor_data and "instance_segmentation" in sensor_data:
            rgb = self._observation_torch(sensor_data["rgb"])
            instance = self._observation_torch(sensor_data["instance_segmentation"])
            valid = sensor_data.get("valid_mask")
            valid = self._observation_torch(valid) if valid is not None else None
            id_to_labels = sensor_data.get("id_to_labels") or {}
        else:
            output = getattr(sensor_data, "output", None)
            info = getattr(sensor_data, "info", None)
            if output is None or info is None:
                return None
            if "rgb" not in output or "instance_segmentation_fast" not in output:
                return None

            rgb = self._observation_torch(output["rgb"])
            instance = self._observation_torch(output["instance_segmentation_fast"])
            id_to_labels = _extract_id_to_labels(info)
            valid = None
            if "depth" in output:
                depth = self._observation_torch(output["depth"]).squeeze(-1)
                valid = torch.isfinite(depth) & (depth > 0)

        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)
        if instance.dim() == 4 and instance.shape[-1] == 1:
            instance = instance.squeeze(-1)
        if instance.dim() == 2:
            instance = instance.unsqueeze(0)

        if valid is None:
            valid = torch.ones(instance.shape, dtype=torch.bool, device=self.device)
        else:
            if valid.dim() == 2:
                valid = valid.unsqueeze(0)
            valid = valid.to(device=self.device, dtype=torch.bool)

        return {
            "rgb": rgb,
            "instance_segmentation": instance,
            "valid_mask": valid,
            "id_to_labels": id_to_labels,
        }

    def _slice_sensor_batch(self, sensor_batch: dict, env_idx: int) -> dict | None:
        """Slice a prepared sensor batch to one environment. Does not convert Warp arrays."""
        rgb = sensor_batch["rgb"]
        instance = sensor_batch["instance_segmentation"]
        valid = sensor_batch["valid_mask"]
        if env_idx >= rgb.shape[0] or env_idx >= instance.shape[0] or env_idx >= valid.shape[0]:
            return None
        return {
            "rgb": rgb[env_idx : env_idx + 1],
            "instance_segmentation": instance[env_idx : env_idx + 1],
            "valid_mask": valid[env_idx : env_idx + 1],
            "id_to_labels": sensor_batch["id_to_labels"],
        }

    def _build_sensor_data_for_env(self, sensor_data: object, env_idx: int) -> dict | None:
        """Build a normalized per-env sensor_data dictionary expected by the accumulator."""
        sensor_batch = self._prepare_sensor_batch(sensor_data)
        if sensor_batch is None:
            return None
        return self._slice_sensor_batch(sensor_batch, env_idx)

    def update_bin_count_scores_from_observation(
        self, env_ids: torch.Tensor, observations: dict
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
        """
        Update live bin counts from a batch of observations.

        :param env_ids: Environment IDs for each observation row.
        :param observations: Per-sensor observation batches keyed by sensor name.
        :return: Tuple of (info_gains, dense_bin_updates_by_env, sparse_bin_updates_by_env).
        """
        if env_ids.numel() == 0:
            return torch.empty((0,), dtype=torch.float32, device=self.device), {}, {}

        env_ids = env_ids.to(device=self.device).reshape(-1)
        selected_envs = env_ids.unique().tolist()

        info_gains = torch.zeros(env_ids.shape[0], dtype=torch.float32, device=self.device)
        dense_bin_updates_by_env = {}
        sparse_bin_updates_by_env = {}

        prepared_sensors = {}
        for sensor_name, sensor_data in observations.items():
            prepared_sensors[sensor_name] = self._prepare_sensor_batch(sensor_data)

        for i in selected_envs:
            env_key = f"env_{i}"
            env_mask = env_ids == i

            pose_delta = torch.zeros_like(self.binned_tree.bin_counts[env_key], device=self.device)
            base_nonzero = (self.binned_tree.bin_counts[env_key] > 0).sum().to(dtype=torch.float32)

            observed_instance_list = []
            observed_length_list = []
            observed_theta_list = []

            for sensor_name, sensor_batch in prepared_sensors.items():
                if sensor_batch is None:
                    continue
                env_sensor_data = self._slice_sensor_batch(sensor_batch, i)
                if env_sensor_data is None:
                    continue

                render_id_lut = self._build_render_id_lut(
                    env_idx=i, render_id_to_prim_path=env_sensor_data["id_to_labels"]
                )
                observed_instance, observed_length, observed_theta = self._accumulate_pose_delta_from_sensor(
                    pose_delta=pose_delta,
                    sensor_data=env_sensor_data,
                    observation_idx=0,
                    render_id_lut=render_id_lut,
                )
                if observed_instance.numel() > 0:
                    observed_instance_list.append(observed_instance)
                    observed_length_list.append(observed_length)
                    observed_theta_list.append(observed_theta)

            self.binned_tree.bin_counts[env_key] += pose_delta

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
            result = {key: h5_to_dict(node[key]) for key in node.keys()}
            attrs = {attr: node.attrs[attr] for attr in node.attrs}
            if attrs:
                result["attrs"] = attrs
            return result
        elif hasattr(node, "read_direct"):
            value = node[()]
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            elif hasattr(value, "dtype") and value.dtype.kind == "S":
                value = value.astype(str)
            attrs = {attr: node.attrs[attr] for attr in node.attrs}
            return {"data": value, "attrs": attrs} if attrs else value
        else:
            return node

    test_h5_file = "/home/luke/IsaacLab/source/isaaclab-sensor-learning/data/data_collection/SL-PDC_20260810_145937.h5"

    with h5.File(test_h5_file, "r") as f:
        trial_dict = h5_to_dict(f)

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

    compute_info_gain = coverage_evaluator.compute_env_info_gain_from_pose_idxs(
        env_ids=torch.tensor([0, 1], device=coverage_evaluator.device),
        closest_idxs=closest_idxs,
        trial_data_dict=trial_dict,
    )
    print(compute_info_gain)
