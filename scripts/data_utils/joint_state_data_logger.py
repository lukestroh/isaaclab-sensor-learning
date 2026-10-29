#!/usr/bin/env python3
import numpy as np
import torch
import h5py as h5
import os


class JointStateDataLogger:
    def __init__(self, compression_level: int = 9):
        pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(pkg_dir, "data", "joint_states")
        self.filepath = os.path.join(data_dir, "joint_states.h5")

        self.compression_level = compression_level
        os.makedirs(data_dir, exist_ok=True)
        return

    def save_joint_states(self, joint_states: list[torch.Tensor]):
        """Save joint states to an HDF5 file.

        Args:
            joint_states (list[torch.Tensor]): List of joint states, each of shape (N,), where N is the number of joints.
            timestamp (float): Timestamp for the data.
        """
        _joint_states = torch.stack(joint_states, dim=0)  # Shape: (B, N)
        with h5.File(self.filepath, "w") as f:
            f.create_dataset("joint_states", data=_joint_states.cpu().numpy(), compression="gzip", compression_opts=self.compression_level)
        print(f"[INFO]: Saved joint states to {self.filepath}")
        return

    