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
        self.rgb_to_bin_lut = self._create_rgb_to_bin_lut()
        # print(self.rgb_to_bin_lut)

        self.instance_ids = {}        # env_id -> 1D int64 tensor of instance ids
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
                (len(unique_instances[f"env_{i}"]), self.n_length_bins, self.n_theta_bins), dtype=torch.int32
            )
            for i in range(n_envs)
        }
        self.bin_scores = {
            f"env_{i}": torch.zeros(
                (len(unique_instances[f"env_{i}"]), self.n_length_bins, self.n_theta_bins), dtype=torch.float32
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
        Create a lookup table (LUT) that maps RGB values to bin indices based on the provided UV image. The LUT is a 4D tensor where the first three dimensions correspond to the RGB values (0-255) and the last dimension contains the corresponding bin indices (length_bin, theta_bin). The LUT is initialized with -1 for all entries, indicating that no bin is assigned

        NOTE: keeps whatever channel order cv2 gave you (BGR if 3-channel, BGRA if 4).
        Whatever order you use here MUST match the order of the batch tensor you query with later.
        """
        img = self.uv_img[..., :3].astype(np.int64)  # drop alpha if present

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
        :rtype: torch.Tensorself.unique_instances = unique_instances
        for env_id, ids in self.unique_instances.items():
            ids_t = torch.tensor([ast.literal_eval(x) for x in ids], dtype=torch.int64, device=self.device)
            self.instance_ids[env_id] = ids_t
            print(self.instance_ids[env_id])
            
            map_t = -torch.ones(, dtype=torch.int64, device=self.device)
            map_t[ids_t] = torch.arange(ids_t.shape[0], dtype=torch.int64, device=self.device)
            self.instance_idx_map[env_id] = map_t
        # self.instance_to_idx = {
        """
        # Ensure the input tensor is of type long for indexing
        rgb_batch = rgb_batch.long()
        # Use advanced indexing to get the bin indices from the LUT
        bin_indices = self.rgb_to_bin_lut[rgb_batch[..., 0], rgb_batch[..., 1], rgb_batch[..., 2]]
        return bin_indices


class CoverageEvaluator:
    def __init__(
        self,
        pose_data: torch.Tensor,
        trial_data: dict,
        observation_config: ObservationConfig,
        gain_config: GainConfig,
        device: str = "cuda:0",
        n_envs: int = 1,
    ) -> None:
        self.device = device
        self.n_envs = n_envs
        self.poses = pose_data
        self.trial_data = trial_data

        # print(self.kd_tree)
        self.observation_config = observation_config
        self.gain_config = gain_config

        # Unique instances (by sensor)
        for i in range(self.n_envs):
            self.unique_instances = {
                f"env_{i}": trial_data["trees"][f"env_{i}"]["instance_labels"]["attrs"] for i in range(self.n_envs)
            }

        # Unique semantic classes (by sensor)
        self.unique_semantics = trial_data["trees"]["semantic_labels"]["attrs"]

        # Binned tree for mapping RGB values to bins
        self.binned_tree = BinnedTree(unique_instances=self.unique_instances, n_envs=self.n_envs, device=self.device)
        return

    def get_closest_pose_idxs(self, curr_poses: torch.Tensor):
        """Get closes poses to the current pose within a specified rad        [0, 0],
        [0, 0],
        [0, 0]], device='cuda:0', dtype=torch.int16)
closest_idxs shape: torch.Size([252, 2])
bin_vals shape: torch.Size([252, 8, 8, 2])
self.bin_counts shape: {'env_0': torch.Size([1601, 20, 36]), 'env_1': torch.Size([1264, 20, 36])}
instance_masked shape: torch.Size([252, 8, 8, 4])
ius using a kd-tree for each environment."""
        dists = (self.poses[:, :, :3] - curr_poses[:, None, :3]).norm(dim=-1)
        mask = dists <= self.observation_config.pose_search_radius

        search_count = 1
        while not torch.any(mask):
            mask = dists <= self.observation_config.pose_search_radius * 1.1 * search_count
            search_count += 1

        idxs = torch.nonzero(mask, as_tuple=False)

        return idxs

    def compute_info_gain_from_pose_idxs(self, closest_idxs: torch.Tensor, trial_data_dict: dict) -> torch.Tensor:
        """
        Compute information gain from the closest pose indices.
        1. Get an image frame from the pose index, and extract the depth, rgb, normals, instance segmentation, and semantic segmentation data.
        2. Check the normals against the sensor orientation. If greater than 45 degrees, mask as 0
        3. Check the depths. If less than 0.03 or greater than 3.4, mask as 0
        4. Bin counts using rgb value and instance segmentation.

        """
        # print("HELLOW ORLD")
        # print({torch.any(self.binned_tree.bin_counts[f'env_{i}']) for i in range(self.n_envs)})
        # print(closest_idxs.shape)
        sensor_data_dict = trial_data_dict["sensors"]
        for sensor_name, sensor_data in sensor_data_dict.items():
            # print(sensor_name, sensor_data.keys())
            # print(sensor_data["attrs"])
            # print()
            sensor_width = sensor_data["attrs"]["width"]
            sensor_height = sensor_data["attrs"]["height"]
            # print(sensor_name, sensor_data.keys())
            env_idxs = closest_idxs[:, 0]
            pose_idxs = closest_idxs[:, 1]

            # SENSOR POSES            # mask normals that are greater than the max incidence angle
            sensor_pose_data = torch.tensor(sensor_data["pose"], device=self.device)[env_idxs, pose_idxs].flatten(
                start_dim=0, end_dim=-2
            )

            normals_data_inv = sensor_data["normals"] # invert normals to point relative to sensor frame
            normals_data = torch.tensor(normals_data_inv, device=self.device)[env_idxs, pose_idxs]

            # NORMALS MASK
            # Convert normals to quaternions
            flattened_normals = normals_data.flatten(start_dim=0, end_dim=-2)
            world_z = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).expand_as(flattened_normals)
            # axis that rotates world_z to the normal (cross(world_z, normal))
            rot_axes = torch.cross(world_z, flattened_normals, dim=-1)
            sin_theta = torch.linalg.norm(rot_axes, dim=-1)
            cos_theta = (flattened_normals * world_z).sum(dim=-1)
            angles = torch.atan2(sin_theta, cos_theta)
            normals_quat = math_utils.quat_from_angle_axis(angle=angles, axis=rot_axes)
            sensor_orientations = sensor_pose_data[:, 3:7]
            sensor_orientations_repeat_interleave = sensor_orientations.repeat_interleave(
                sensor_height * sensor_width, dim=0
            )
            quat_diff_err = math_utils.quat_error_magnitude(q1=sensor_orientations_repeat_interleave, q2=normals_quat)
            # mask normals that are greater than the max incidence angle
            normals_mask = quat_diff_err <= np.radians(self.observation_config.max_incidence_angle)

            # DEPTH MASK
            depth_data = torch.tensor(sensor_data["depth"], device=self.device)[env_idxs, pose_idxs].flatten()
            depth_mask = (depth_data >= sensor_data["attrs"]["z_near"]) & (depth_data < sensor_data["attrs"]["z_far"])

            # RGB
            rgb_data = torch.tensor(sensor_data["rgb"], device=self.device)[env_idxs, pose_idxs].flatten(
                start_dim=0, end_dim=-2
            )
            # INSTANCE SEGMENTATION
            instance_data = torch.tensor(sensor_data["instance_segmentation"], device=self.device)[
                env_idxs, pose_idxs
            ].flatten(start_dim=0, end_dim=-2)
            # # SEMANTIC SEGMENTATION
            semantic_data = torch.tensor(sensor_data["semantic_segmentation"], device=self.device)[
                env_idxs, pose_idxs
            ].flatten(start_dim=0, end_dim=-2)

            permissible_mask = (normals_mask & depth_mask).unsqueeze(-1)
            rgb_masked = rgb_data * permissible_mask
            instance_masked = (instance_data * permissible_mask).reshape(-1, sensor_height, sensor_width, 4)
            semantic_masked = (semantic_data * permissible_mask).reshape(-1, sensor_height, sensor_width, 4)
            # print(instance_masked.shape)
            import time
            quat_diff_err = quat_diff_err.reshape(-1, sensor_height, sensor_width)
            rgb_data = rgb_data.reshape(-1, sensor_height, sensor_width, 3)

            _idx = 112
            print(rgb_data.reshape(-1, sensor_height, sensor_width, 3)[_idx])
            print(torch.rad2deg(quat_diff_err).reshape(-1, sensor_height, sensor_width)[_idx])

            print(self.observation_config.max_incidence_angle)
            print(np.radians(self.observation_config.max_incidence_angle))
            print(torch.where(normals_mask))
            
            # print([0])
            # print(f"any _masked: {torch.any(quat_diff_err)}")
            import sys; sys.exit()

            # BINNING
            # 1. add bin_vals to copy of self.binned_tree.bin_counts for each instance in the environment.
            bin_vals = self.binned_tree.rgb_to_bin(rgb_masked).reshape(-1, 8, 8, 2)
            bin_vals_dict = {f"env_{i}": bin_vals[env_idxs == i] for i in range(self.n_envs)}
            # print(f"bin_vals_dict: {dict((k, v.shape) for k, v in bin_vals_dict.items())}")
            # bin_vals_by_env = {f"env_{i}": bin_vals[env_idxs == i]. for i in range(self.n_envs)}

            # Vectorized update: for each environment, compute per-instance bin counts
            for env_id, env_bin_vals in bin_vals_dict.items():
                env_idx = int(env_id.split("_")[1])

                # Flatten bin indices for this env's pixels
                bins_flat = env_bin_vals.reshape(-1, 2).long()

                # Flatten instance RGBA values for pixels in this env: shape (P,4)
                inst_pixels = instance_masked[env_idxs == env_idx].reshape(-1, 4).to(dtype=torch.int64)
                print(torch.any(inst_pixels))

                # Filter out invalid bin entries (lut initialized with -1 for unknowns)
                valid_mask = bins_flat[:, 0] >= 0
                if not torch.any(valid_mask):
                    print(f"No valid bin indices for env {env_id}. Skipping bin count update.")
                    continue

                idxs = bins_flat[:, 0] * self.binned_tree.n_theta_bins + bins_flat[:, 1]
                idxs_valid = idxs[valid_mask]
                inst_valid = inst_pixels[valid_mask]

                if inst_valid.numel() == 0:
                    print(f"No valid instance pixels for env {env_id}. Skipping bin count update.")
                    continue

                # Pack RGBA tuples into 64-bit scalars so we can compare/match them efficiently
                packed = (
                    inst_valid[:, 0]
                    | (inst_valid[:, 1] << 8)
                    | (inst_valid[:, 2] << 16)
                    | (inst_valid[:, 3] << 24)
                ).to(dtype=torch.int64)

                print(packed)

                # Get unique packed values observed in these pixels and inverse indices per pixel
                unique_packed, inv = torch.unique(packed, return_inverse=True)

                # Build packed tensor for the environment's full instance list (global ordering)
                env_inst_list = [ast.literal_eval(x) for x in self.unique_instances[env_id]]
                env_inst_arr = torch.tensor(env_inst_list, dtype=torch.int64, device=self.device)
                env_inst_packed = (
                    env_inst_arr[:, 0]
                    | (env_inst_arr[:, 1] << 8)
                    | (env_inst_arr[:, 2] << 16)
                    | (env_inst_arr[:, 3] << 24)
                ).to(dtype=torch.int64)

                # Map each unique_packed (observed) to the index in the global env instance ordering
                # matches shape: (num_env_instances, num_unique_observed)
                matches = env_inst_packed.unsqueeze(1) == unique_packed.unsqueeze(0)
                positions = torch.nonzero(matches, as_tuple=False)
                if positions.numel() == 0:
                    print(f"No matches found for env {env_id}. Skipping bin count update.")
                    continue

                # positions[:,1] -> index in unique_packed; positions[:,0] -> index in env_inst_packed
                unique_to_env_idx = -torch.ones(unique_packed.shape[0], dtype=torch.int64, device=self.device)
                unique_to_env_idx[positions[:, 1]] = positions[:, 0]

                # Translate per-pixel inverse indices to global instance indices
                inst_idx_per_pixel = unique_to_env_idx[inv]

                # Now compute linear positions for bincount: inst_idx * B + bin_idx
                num_instances = env_inst_packed.shape[0]
                B = self.binned_tree.n_length_bins * self.binned_tree.n_theta_bins
                linear_pos = inst_idx_per_pixel * B + idxs_valid.to(dtype=torch.int64)

                # Remove any pixels that did not map to a known instance (inst_idx == -1)
                valid_pixels = linear_pos >= 0
                if not torch.any(valid_pixels):
                    print(f"No valid pixels for env {env_id}. Skipping bin count update.")
                    continue
                linear_pos = linear_pos[valid_pixels]

                # Accumulate counts via bincount into shape (num_instances, B)
                counts_1d = torch.bincount(linear_pos, minlength=(num_instances * B)).to(dtype=torch.int32)
                counts_update = counts_1d.view(num_instances, B).view(
                    num_instances, self.binned_tree.n_length_bins, self.binned_tree.n_theta_bins
                )

                # Add to global bin counts for this environment
                self.binned_tree.bin_counts[env_id] += counts_update

            # bin_sums = {
            #     key: torch.zeros_like(value, dtype=torch.float32) for key, value in self.binned_tree.bin_counts.items()
            # }

            print({torch.any(self.binned_tree.bin_counts[f'env_{i}']) for i in range(self.n_envs)})

                        

            # print(self.binned_tree.bin_counts)

            # 2. compute the gain for all instances per pose image per environment.
            # 3. find the highest gain instance and select as next desired pose.
            # 4. update self.binned_tree.bin_counts with the new bin_vals for the selected instance.
            # 5. update self.binned_tree.bin_scores with the new gain for the selected instance.
            # bin_sums = {key: torch.zeros_like(value, dtype=torch.float32) for key, value in self.binned_tree.bin_counts.items()}
            # for i in range(self.n_envs):
            #     bin_sums[f"env_{i}"] = bin_vals

            # print(bin_vals)
            print(f"closest_idxs shape: {closest_idxs.shape}")
            print(f"bin_vals shape: {bin_vals.shape}")
            print(f"self.bin_counts shape: {dict((k, v.shape) for k, v in self.binned_tree.bin_counts.items())}")
            print(f"instance_masked shape: {instance_masked.reshape(-1, 8, 8, 4).shape}")
            # print(f"semantic_masked shape: {semantic_masked.reshape(-1, 8, 8, 4).shape}")

            # print(self.binned_tree.bins.shape)
            # possible_bin_update = self.binned_tree.bins + bin_vals

            # SCORING

            break

        return


#     def bin_views_from_pose(self, pose_idx: int, pose_data: dict, sensor_data: dict) -> np.ndarray | None:

#         return

#     def compute_bin_gains(self, bins: np.ndarray) -> np.ndarray:
#         """logistic decay function for minimal return on repeated observations, with saturation at n_max observations"""
#         n = self.observation_counts[bins]
#         # print(f"observation counts for faces {faces}: {n}")
#         # gains = self.gain_config.saturation_lambda * (1 - np.exp(-n / self.gain_config.n_max))
#         gains = self.gain_config.saturation_lambda / (
#             1 + np.exp(self.gain_config.steepness * (n - self.gain_config.n_midpoint))
#         )
#         return gains

#     def compute_pose_gain(self, bin_gains: np.ndarray) -> float:
#         if len(bin_gains) == 0:
#             return 0.0
#         pose_gain = np.sum(bin_gains)
#         return pose_gain


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


if __name__ == "__main__":
    # from isaaclab_sensor_learning.utils import h5py_utils

    # test_h5_file = "/home/luke/IsaacLab/source/isaaclab-sensor-learning/data/SL-PDC_20260730_133953.h5"
    test_h5_file = "/home/luke/IsaacLab/source/isaaclab-sensor-learning/data/SL-PDC_20260806_165340.h5"

    with h5.File(test_h5_file, "r") as f:
        trial_dict = h5_to_dict(f)

    actual_poses = trial_dict["poses"]["actual_poses"]["eef_poses"]
    actual_poses = torch.tensor(actual_poses, dtype=torch.float32, device="cuda:0")

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
        pose_data=actual_poses,
        trial_data=trial_dict,
        observation_config=ObservationConfig(),
        gain_config=GainConfig(),
    )

    closest_idxs = coverage_evaluator.get_closest_pose_idxs(curr_poses=start_pose_w)
    # print(f"Closest pose indices: {closest_idxs.shape}, {closest_idxs}")

    info_gains = coverage_evaluator.compute_info_gain_from_pose_idxs(
        closest_idxs=closest_idxs, trial_data_dict=trial_dict
    )
