#!/usr/bin/env python3
from dataclasses import dataclass

import h5py as h5
import numpy as np
import plotly.graph_objects as go
import scipy.spatial as sps
from plotly import data
from pybullet_tree_sim.sensors.sensor import st
from pybullet_tree_sim.tree_metadata import Face
from scipy.constants import c
import os
from sensor_learning.plot import evaluation_plots as eval_plots
from sensor_learning.utils import pose_generator


@dataclass
class ObservationConfig:
    min_distance: float  # meters, get as input from z_near
    max_distance: float  # meters, get as input from z_far
    max_incidence_angle: float = 45.0  # degrees
    kd_radius: float = 0.1  # radius for finding neighboring poses in kd-tree search
    bin_size: float = 0.05  # meters, size of bins for distance histogram


@dataclass
class GainConfig:
    saturation_lambda: float = 1.0  # controls how fast gain saturates
    n_midpoint: int = 5  # observations beyond n_max add little value
    steepness: float = 2.0  # controls steepness of gain curve around n_midpoint


class BinnedTree:
    def __init__(self):
        self.bins = {}
        self.bin_size = 0.05  # meters
        return



class CoverageEvaluator:
    def __init__(self, data_dict: dict, observation_config: ObservationConfig, gain_config: GainConfig) -> None:
        self.data = data_dict
        self.poses = self.data["eef_poses"]["actual_poses"]
        self.kd_tree = sps.KDTree(self.poses[:, :3])
        # print(self.kd_tree)
        self.observation_config = observation_config
        self.gain_config = gain_config
        self.observation_counts = np.zeros(shape=, dtype=np.int32)
        self.pose_gains = np.zeros(shape=self.data["eef_poses"]["actual_poses"].shape[0], dtype=np.float32)
        self.total_gain = 0.0
        return
    
    def get_closest_poses(self, curr_pose):
        idxs = self.kd_tree.query_ball_point(x=curr_pose[:3], r=self.observation_config.kd_radius)
        return idxs
    
    def bin_views_from_pose(self, pose_idx: int, pose_data: dict, sensor_data: dict) -> np.ndarray | None:
        
        return
    
    def compute_bin_gains(self, bins: np.ndarray) -> np.ndarray:
        """logistic decay function for minimal return on repeated observations, with saturation at n_max observations"""
        n = self.observation_counts[bins]
        # print(f"observation counts for faces {faces}: {n}")
        # gains = self.gain_config.saturation_lambda * (1 - np.exp(-n / self.gain_config.n_max))
        gains = self.gain_config.saturation_lambda / (
            1 + np.exp(self.gain_config.steepness * (n - self.gain_config.n_midpoint))
        )
        return gains
    
    def compute_pose_gain(self, bin_gains: np.ndarray) -> float:
        if len(bin_gains) == 0:
            return 0.0
        pose_gain = np.sum(bin_gains)
        return pose_gain