#!/usr/bin/env python3

import numpy as np


def xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion from (x, y, z, w) to (w, x, y, z) format."""
    if quat.shape[-1] != 4:
        raise ValueError("Input quaternion must have shape (..., 4)")
    return np.stack([quat[..., 3], quat[..., 0], quat[..., 1], quat[..., 2]], axis=-1)


def wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion from (w, x, y, z) to (x, y, z, w) format."""
    if quat.shape[-1] != 4:
        raise ValueError("Input quaternion must have shape (..., 4)")
    return np.stack([quat[..., 1], quat[..., 2], quat[..., 3], quat[..., 0]], axis=-1)


if __name__ == "__main__":
    # Example usage
    quat_xyzw = np.array([0.7071, 0.0, 0.7071, 0.0])  # (x, y, z, w)
    quat_wxyz = xyzw_to_wxyz(quat_xyzw)
    print("WXYZ format:", quat_wxyz)

    quat_converted_back = wxyz_to_xyzw(quat_wxyz)
    print("Converted back to XYZW format:", quat_converted_back)

    # examples with batch of quaternions
    batch_quats_xyzw = np.array([[0.7071, 0.0, 0.7071, 0.0], [0.0, 0.7071, 0.0, 0.7071]])  # shape (2, 4)
    batch_quats_wxyz = xyzw_to_wxyz(batch_quats_xyzw)
    print("Batch WXYZ format:", batch_quats_wxyz)

    batch_quats_converted_back = wxyz_to_xyzw(batch_quats_wxyz)
    print("Batch converted back to XYZW format:", batch_quats_converted_back)
