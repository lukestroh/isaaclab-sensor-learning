#!/usr/bin/env python3
from dataclasses import dataclass, field

import h5py as h5
import numpy as np
import torch

from isaaclab.utils import math as math_utils
from isaaclab_sensor_learning.utils import h5py_utils
from isaaclab_sensor_learning.nbv import coverage_metrics
from isaaclab_sensor_learning.nbv.coverage_metrics import INSTANCE_CLASSES, CoverageConfig, instance_class  # noqa: F401

_SKIP_INSTANCE_LABELS = {"BACKGROUND", "UNLABELLED"}

# Optical axis of a stored sensor pose_w quaternion, by camera convention (the sensor group's "pose_w_convention"
# attr): (rotation-matrix column, sign).
_OPTICAL_AXIS = {"ros": (2, 1.0), "opengl": (2, -1.0), "world": (0, 1.0)}

# Camera output carrying the tree's UV texture colors. Unlit, so colors decode to the right bins (the lit "rgb" output
# is tonemapped and tinted, which pushes pixels into wrong, too-low bins).
_UV_COLOR_KEY = "albedo"


def valid_pixel_mask(
    depth: torch.Tensor,
    normals: torch.Tensor,
    sensor_quat: torch.Tensor,
    z_near: float,
    z_far: float,
    max_incidence_deg: float,
    convention: str = "ros",
) -> torch.Tensor:
    """
    Pixels that count as observations: depth in [z_near, z_far) and the surface faces the sensor within
    ``max_incidence_deg`` of its optical axis. Shared by predicted (stored) and live binning so the two agree.

    :param depth: (B, H, W) depth.
    :param normals: (B, H, W, 3) world-frame surface normals.
    :param sensor_quat: (B, 4) sensor orientation (x, y, z, w) in ``convention``.
    :param convention: camera convention of ``sensor_quat``; selects the optical axis (see ``_OPTICAL_AXIS``).
    :return: (B, H, W) bool mask.
    """
    axis, sign = _OPTICAL_AXIS[convention]
    view_dir = sign * math_utils.matrix_from_quat(sensor_quat)[:, :, axis]  # (B, 3)
    cosang = (-normals * view_dir[:, None, None, :]).sum(dim=-1).clamp(-1.0, 1.0)
    normals_mask = torch.acos(cosang) <= np.radians(max_incidence_deg)  # NaN normals compare False
    depth_mask = (depth >= z_near) & (depth < z_far)
    return normals_mask & depth_mask


@dataclass
class ObservationConfig:
    max_incidence_angle: float = 45.0  # degrees
    pose_search_radius: float = 0.5  # radius for finding neighboring poses
    candidate_batch_size: int = 16  # max candidate poses read from disk / held in VRAM at once
    cache_pose_deltas: bool = True  # cache each candidate's sparse bin delta so it is only read from disk once


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
    """
    Per-env bin counts of shape (n_instances, n_length_bins, n_theta_bins).

    Tree parts are textured with ``uv.png``: R encodes the theta (around) coordinate and G the length (along)
    coordinate, both 0-255 across the full UV range, with B = 0. A pixel's bin is read straight from its color.
    """

    def __init__(
        self,
        n_envs: int,
        unique_instances: dict,
        n_length_bins: int = 20,
        n_theta_bins: int = 36,
        device: str = "cuda:0",
    ):
        self.device = device
        self.n_length_bins = n_length_bins
        self.n_theta_bins = n_theta_bins
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

    def uv_color_to_bin(self, uv_colors: torch.Tensor) -> torch.Tensor:
        """
        Convert UV texture colors to (length_bin, theta_bin).

        :param uv_colors: (..., C >= 3) colors in 0-255 (uint8 from the camera, or float32 as stored in h5).
            Each 8-bit value is read at its center, (c + 0.5) / 256, as a normalized UV coordinate.
        :return: (..., 2) int64 bins on ``uv_colors``' device.
        """
        rgb = torch.nan_to_num(uv_colors[..., :3].to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        rgb = rgb.clamp(0.0, 255.0)
        theta = ((rgb[..., 0] + 0.5) * (self.n_theta_bins / 256.0)).long().clamp(0, self.n_theta_bins - 1)
        length = ((rgb[..., 1] + 0.5) * (self.n_length_bins / 256.0)).long().clamp(0, self.n_length_bins - 1)
        return torch.stack((length, theta), dim=-1)


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
    instance_info = info.get("instance_segmentation", {})
    if not isinstance(instance_info, dict):
        return {}
    return instance_info.get("idToLabels") or instance_info.get("idsToLabels") or {}


class NBVEvaluator:
    """
    Greedy next-best-view evaluator over pre-captured poses.

    Pose/tree metadata is loaded eagerly. Sensor images stay on disk and are only read for the candidate
    poses being scored, ``ObservationConfig.candidate_batch_size`` poses at a time.
    """

    def __init__(
        self,
        trial_data_path: str,
        observation_config: ObservationConfig,
        gain_config: GainConfig,
        device: str = "cuda:0",
        n_envs: int = 1,
        coverage_config: CoverageConfig | None = None,
    ) -> None:
        self.device = device
        self.n_envs = n_envs
        self.observation_config = observation_config
        self.gain_config = gain_config
        self.coverage_config = coverage_config if coverage_config is not None else CoverageConfig()

        # Kept open for the lifetime of the evaluator; call close() when done.
        self.h5_file = h5.File(trial_data_path, "r")
        trial_data = h5py_utils.h5_to_dict(self.h5_file, attrs_only=("sensors",))
        self.sensor_attrs = {name: group["attrs"] for name, group in trial_data["sensors"].items()}
        self.poses = torch.tensor(
            trial_data["poses"]["actual_poses"]["eef_poses"], dtype=torch.float32, device=self.device
        )
        self.skipped_poses = torch.from_numpy(
            np.ascontiguousarray(trial_data["poses"]["actual_poses"].get("skipped_poses", []))
        ).to(device=self.device)

        self.prim_path_to_custom_id = {f"env_{i}": {} for i in range(self.n_envs)}
        self.render_id_to_prim_path = {f"env_{i}": {} for i in range(self.n_envs)}
        self.h5_render_id_lut = {}
        self.instance_semantic_weight = {}
        self.instance_class_ids = {}

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
            self.instance_class_ids[env_key] = coverage_metrics.class_ids_from_prim_map(
                self.prim_path_to_custom_id[env_key], device=self.device
            )

        self.binned_tree = BinnedTree(
            unique_instances=self.prim_path_to_custom_id, n_envs=self.n_envs, device=self.device
        )

        # (env_id, pose_id) -> sparse bin-count delta summed over all sensors. Captured data is static,
        # so a candidate's delta never changes; only its score (which depends on live counts) does.
        self._pose_delta_cache: dict[tuple[int, int], torch.Tensor] = {}
        # last seen Camera.frame per sensor, so live binning only counts freshly rendered frames
        self._last_frames: dict[str, torch.Tensor] = {}
        return

    def close(self) -> None:
        self.h5_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _semantic_weight_from_prim_path(self, prim_path: str) -> float:
        """Map a tree prim path to its class gain (see ``instance_class``); unknown classes get 1.0."""
        cls = instance_class(prim_path)
        return float(self.gain_config.branch_gains[cls]) if cls in self.gain_config.branch_gains else 1.0

    def instance_coverage(self, min_hits: tuple[int, ...] | None = None) -> dict[str, dict[str, dict]]:
        """
        Instance detection from the live bin counts (``coverage_metrics.instance_coverage``), per env.

        :param min_hits: hit thresholds; defaults to ``coverage_config.instance_min_hits``.
        :return: ``{env_key: {cls: {"n_instances": int, "seen": {k: fraction}}}}``.
        """
        min_hits = tuple(self.coverage_config.instance_min_hits if min_hits is None else min_hits)
        return {
            env_key: coverage_metrics.instance_coverage(counts, self.instance_class_ids[env_key], min_hits)
            for env_key, counts in self.binned_tree.bin_counts.items()
        }

    def instance_extent_per_instance(self, env_key: str, min_hits: int) -> dict[str, torch.Tensor]:
        """Per-instance extent of one env at threshold ``min_hits`` (``coverage_metrics.instance_extent_per_instance``)."""
        return coverage_metrics.instance_extent_per_instance(
            self.binned_tree.bin_counts[env_key], self.instance_class_ids[env_key], min_hits, self.coverage_config
        )

    def instance_extent(self, min_hits: tuple[int, ...] | None = None) -> dict[str, dict[str, dict]]:
        """
        Per-class extent from the live bin counts (``coverage_metrics.instance_extent``), per env.

        :return: ``{env_key: {cls: {k: {"length_coverage", "theta_coverage", "continuous"}}}}``.
        """
        min_hits = tuple(self.coverage_config.instance_min_hits if min_hits is None else min_hits)
        return {
            env_key: coverage_metrics.instance_extent(
                counts, self.instance_class_ids[env_key], min_hits, self.coverage_config
            )
            for env_key, counts in self.binned_tree.bin_counts.items()
        }

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
        self, env_ids: torch.Tensor, closest_idxs: torch.Tensor
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

    def _load_masked_sensor_batch(self, sensor_name: str, env_id: int, pose_ids: np.ndarray) -> dict:
        """
        Read one sensor's captured data for a batch of poses in one env and mask out invalid pixels.

        Pixel validity is ``valid_pixel_mask`` (the same mask live binning uses).

        :param pose_ids: Sorted, unique pose indices (required by h5py fancy indexing).
        :return: Dict with "uv_color" (B, H, W, 3), "instance_segmentation" (B, H, W), and "valid_mask" (B, H, W).
        """
        group = self.h5_file[f"sensors/{sensor_name}"]
        attrs = self.sensor_attrs[sensor_name]
        data = {
            key: torch.from_numpy(group[key][env_id, pose_ids]).to(device=self.device)
            for key in ("pose_w", _UV_COLOR_KEY, "depth", "normals", "instance_segmentation")
        }
        valid_mask = valid_pixel_mask(
            depth=data["depth"],
            normals=data["normals"],
            sensor_quat=data["pose_w"][:, 3:7],
            z_near=attrs["z_near"],
            z_far=attrs["z_far"],
            max_incidence_deg=self.observation_config.max_incidence_angle,
            convention=str(attrs["pose_w_convention"]),
        )
        return {
            "uv_color": data[_UV_COLOR_KEY][..., :3],
            "instance_segmentation": data["instance_segmentation"],
            "valid_mask": valid_mask,
        }

    def _compute_pose_deltas(self, env_id: int, pose_ids: np.ndarray) -> list[torch.Tensor]:
        """
        Compute the predicted bin-count delta (summed over sensors) for each pose, reading from disk in batches.

        :param pose_ids: Sorted, unique pose indices for ``env_id``.
        :return: One sparse (n_instances, n_length_bins, n_theta_bins) delta per pose, in ``pose_ids`` order.
        """
        env_key = f"env_{env_id}"
        render_id_lut = self.h5_render_id_lut[env_key]
        live_counts = self.binned_tree.bin_counts[env_key]
        batch_size = self.observation_config.candidate_batch_size

        pose_deltas = []
        for start in range(0, len(pose_ids), batch_size):
            batch_pose_ids = pose_ids[start : start + batch_size]
            sensor_batches = [
                self._load_masked_sensor_batch(sensor_name, env_id, batch_pose_ids)
                for sensor_name in self.sensor_attrs
            ]
            for local_idx in range(len(batch_pose_ids)):
                pose_delta = torch.zeros_like(live_counts)
                for sensor_batch in sensor_batches:
                    self._accumulate_pose_delta_from_sensor(
                        pose_delta=pose_delta,
                        sensor_data=sensor_batch,
                        observation_idx=local_idx,
                        render_id_lut=render_id_lut,
                    )
                pose_deltas.append(pose_delta.to_sparse().coalesce())
        return pose_deltas

    def _get_pose_deltas(self, env_id: int, pose_ids: np.ndarray) -> list[torch.Tensor]:
        """Return pose deltas for ``pose_ids``, only reading poses that are not already cached."""
        if not self.observation_config.cache_pose_deltas:
            return self._compute_pose_deltas(env_id, pose_ids)

        missing = np.array([p for p in pose_ids if (env_id, int(p)) not in self._pose_delta_cache], dtype=np.int64)
        if missing.size > 0:
            for pose_id, pose_delta in zip(missing, self._compute_pose_deltas(env_id, missing)):
                self._pose_delta_cache[(env_id, int(pose_id))] = pose_delta
        return [self._pose_delta_cache[(env_id, int(p))] for p in pose_ids]

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
        :param sensor_data: Dictionary with "uv_color", "instance_segmentation", and "valid_mask" batches.
        :param observation_idx: Observation index inside the provided sensor batch.
        :param render_id_lut: 1D tensor mapping render instance IDs to custom ids.
        :return: Tuple of (instance_indices, length_bin_indices, theta_bin_indices) for touched bins.
        """
        instance_pose = sensor_data["instance_segmentation"][observation_idx].to(dtype=torch.int64)
        valid_pose = sensor_data["valid_mask"][observation_idx]
        bin_pose = self.binned_tree.uv_color_to_bin(sensor_data["uv_color"][observation_idx])

        valid = valid_pose.reshape(-1)
        flat_bin_l = bin_pose[..., 0].reshape(-1)
        flat_bin_t = bin_pose[..., 1].reshape(-1)
        flat_instance = instance_pose.reshape(-1)

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

    def score_pose_delta(
        self, pose_delta: torch.Tensor, live_counts: torch.Tensor, instance_weights: torch.Tensor
    ) -> float:
        """
        Compute a score for a candidate pose delta without mutating live bin state.

        Predicted bin gains from all sensors (already summed into ``pose_delta``) are
        weighted by each instance's semantic class.

        :param pose_delta: Coalesced sparse count delta with shape (n_instances, n_length_bins, n_theta_bins).
        :param live_counts: Current bin counts with the same shape as ``pose_delta``.
        :param instance_weights: Per-instance class weights, shape (n_instances,).
        :return: A float score for the pose delta.
        """
        observed = pose_delta.values() > 0
        if not torch.any(observed):
            return 0.0

        instance_idxs, length_idxs, theta_idxs = pose_delta.indices()[:, observed]
        projected_counts = (
            live_counts[instance_idxs, length_idxs, theta_idxs] + pose_delta.values()[observed]
        ).to(dtype=torch.float32)
        gains = self.gain_config.saturation_lambda / (
            1 + torch.exp(self.gain_config.steepness * (projected_counts - self.gain_config.n_midpoint))
        )
        weighted_gains = gains * instance_weights[instance_idxs]
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

    def get_candidate_pose_idxs(
        self,
        curr_poses: torch.Tensor,
        env_ids: list[int],
        visited: torch.Tensor | None = None,
        max_radius: float = float("inf"),
    ) -> tuple[torch.Tensor, list[int]]:
        """
        Per env: unvisited, non-skipped captured poses near the current EEF pose.

        The search radius starts at ``pose_search_radius`` and grows (x1.1, x2.2, ... of it) until it reaches the
        nearest eligible pose. An env whose nearest eligible pose is farther than ``max_radius`` (or that has none
        left) is exhausted.

        :param curr_poses: (n_envs, >=3) current EEF poses in world frame.
        :param visited: (n_envs, n_poses) bool, poses already used as goals.
        :return: ((N, 2) candidate (env_id, pose_id) rows, list of exhausted env ids).
        """
        base_radius = self.observation_config.pose_search_radius
        eligible = torch.ones(self.poses.shape[:2], dtype=torch.bool, device=self.device)
        if self.skipped_poses.numel() > 0:
            eligible[:, self.skipped_poses.long()] = False
        if visited is not None:
            eligible &= ~visited.to(self.device)

        rows, exhausted = [], []
        for env_id in env_ids:
            dists = (self.poses[env_id, :, :3] - curr_poses[env_id, :3].to(self.device)).norm(dim=-1)
            if not bool(eligible[env_id].any()):
                exhausted.append(env_id)
                continue
            nearest = float(dists[eligible[env_id]].min())
            if nearest > max_radius:
                exhausted.append(env_id)
                continue
            radius, k = base_radius, 1
            while radius < nearest:  # same growth schedule as get_closest_pose_idxs
                radius, k = base_radius * 1.1 * k, k + 1
            pose_ids = torch.nonzero(eligible[env_id] & (dists <= radius), as_tuple=False)[:, 0]
            rows.append(torch.stack([torch.full_like(pose_ids, env_id), pose_ids], dim=1))
        candidates = torch.cat(rows) if rows else torch.empty((0, 2), dtype=torch.long, device=self.device)
        return candidates, exhausted

    def compute_env_info_gain_from_pose_idxs(
        self, env_ids: torch.Tensor, closest_idxs: torch.Tensor, return_scores: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Score each env's candidate poses against its live bin counts and return the best (env_id, pose_id) pairs.

        Only the candidate poses are read from disk (or the delta cache). Does not mutate live bin state.
        Envs with no remaining candidates are omitted from the result.

        :param return_scores: also return each returned pair's predicted score, shape (N,).
        """
        env_ids, candidates = self._filter_pose_candidates(env_ids, closest_idxs)

        best_pose_by_env, best_scores = [], []
        for i in env_ids.tolist():
            pose_ids = torch.unique(candidates[candidates[:, 0] == i, 1]).cpu().numpy()
            if pose_ids.size == 0:
                continue

            env_key = f"env_{i}"
            live_counts = self.binned_tree.bin_counts[env_key]
            instance_weights = self.instance_semantic_weight[env_key]
            scores = [
                self.score_pose_delta(pose_delta, live_counts, instance_weights)
                for pose_delta in self._get_pose_deltas(i, pose_ids)
            ]
            # argmax returns the first maximum, so ties go to the lowest pose id
            best_local = int(np.argmax(scores))
            best_pose_by_env.append((i, int(pose_ids[best_local])))
            best_scores.append(scores[best_local])

        best = (
            torch.tensor(best_pose_by_env, dtype=torch.long, device=self.device)
            if best_pose_by_env
            else torch.empty((0, 2), dtype=torch.long, device=self.device)
        )
        if return_scores:
            return best, torch.tensor(best_scores, dtype=torch.float32, device=self.device)
        return best

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

    def _prepare_live_sensor_batch(self, sensor_name: str, sensor_data: object) -> dict | None:
        """
        Read one sensor's live CameraData for all envs and apply ``valid_pixel_mask``.

        Depth limits come from the capture file's sensor attrs (same rig); the orientation is ``quat_w_ros``.
        Returns None if the sensor lacks the albedo output, instance segmentation, depth, or normals.
        """
        output = getattr(sensor_data, "output", None)
        info = getattr(sensor_data, "info", None)
        if output is None or info is None:
            return None
        if any(k not in output for k in (_UV_COLOR_KEY, "instance_segmentation", "depth", "normals")):
            return None

        instance = self._observation_torch(output["instance_segmentation"])
        depth = self._observation_torch(output["depth"])
        instance = instance.squeeze(-1) if instance.dim() == 4 else instance
        depth = depth.squeeze(-1) if depth.dim() == 4 else depth
        attrs = self.sensor_attrs[sensor_name]
        valid = valid_pixel_mask(
            depth=depth,
            normals=self._observation_torch(output["normals"])[..., :3],
            sensor_quat=self._observation_torch(sensor_data.quat_w_ros),
            z_near=attrs["z_near"],
            z_far=attrs["z_far"],
            max_incidence_deg=self.observation_config.max_incidence_angle,
            convention="ros",
        )
        return {
            "uv_color": self._observation_torch(output[_UV_COLOR_KEY])[..., :3],
            "instance_segmentation": instance,
            "valid_mask": valid,
            "id_to_labels": _extract_id_to_labels(info),
        }

    def _fresh_envs(self, sensor_name: str, frame: torch.Tensor | None) -> torch.Tensor | None:
        """
        Per-env bool mask of envs whose camera rendered a new frame since the last call; None means "all".

        Between renders (``update_period`` = 1 / data_rate_hz) ``sensor.data`` keeps returning the previous frame,
        so binning every physics step would count each real frame several times.
        """
        if frame is None:
            return None
        frame = self._observation_torch(frame).to(device=self.device).reshape(-1)
        last = self._last_frames.get(sensor_name)
        fresh = torch.ones_like(frame, dtype=torch.bool) if last is None else frame != last
        self._last_frames[sensor_name] = frame.clone()
        return fresh

    def update_bin_count_scores_from_observation(
        self, env_ids: torch.Tensor, observations: dict, frames: dict[str, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
        """
        Update live bin counts from a batch of observations.

        :param env_ids: Environment IDs for each observation row.
        :param observations: Per-sensor CameraData keyed by sensor name.
        :param frames: Per-sensor ``Camera.frame`` (per-env render counters). Sensors/envs whose counter did not
            advance since the previous call are skipped. None bins every observation (no rate limiting).
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
        fresh_by_sensor = {}
        for sensor_name, sensor_data in observations.items():
            prepared_sensors[sensor_name] = self._prepare_live_sensor_batch(sensor_name, sensor_data)
            fresh_by_sensor[sensor_name] = self._fresh_envs(sensor_name, None if frames is None else frames[sensor_name])

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
                fresh = fresh_by_sensor[sensor_name]
                if fresh is not None and not bool(fresh[i]):
                    continue
                env_sensor_data = {
                    k: (v if k == "id_to_labels" else v[i : i + 1]) for k, v in sensor_batch.items()
                }

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
    import sys

    start_pose_w = torch.tensor(
        [
            [2.0977, 1.0163, 2.0615, -0.7144, -0.0093, 0.0095, 0.6996],
            [-2.9023, 1.0163, 2.0615, -0.7144, -0.0093, 0.0095, 0.6996],
        ],
        dtype=torch.float32,
        device="cuda:0",
    )

    with NBVEvaluator(
        trial_data_path=sys.argv[1],
        n_envs=start_pose_w.shape[0],
        observation_config=ObservationConfig(),
        gain_config=GainConfig(),
    ) as nbv_evaluator:
        closest_idxs = nbv_evaluator.get_closest_pose_idxs(curr_poses=start_pose_w)
        best_poses = nbv_evaluator.compute_env_info_gain_from_pose_idxs(
            env_ids=torch.arange(start_pose_w.shape[0], device=nbv_evaluator.device), closest_idxs=closest_idxs
        )
        print(best_poses)
