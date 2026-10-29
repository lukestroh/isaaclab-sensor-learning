"""Shared fixtures. Tests run without Isaac Sim: Omniverse UI modules are stubbed, everything else is real."""

import os
import sys
from dataclasses import dataclass
from unittest import mock

import cv2
import h5py
import numpy as np
import pytest

# The package __init__ registers a UI extension that imports omni.ext / omni.ui, which only exist inside Kit.
for _module in ("omni", "omni.ext", "omni.ui"):
    sys.modules.setdefault(_module, mock.MagicMock())
sys.modules["omni.ext"].IExt = type("IExt", (), {})

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source", "isaaclab_sensor_learning"))

from isaaclab_sensor_learning.nbv import nbv  # noqa: E402

UV_IMG_PATH = os.path.join(os.path.dirname(nbv.__file__), "uv.png")
N_LENGTH_BINS, N_THETA_BINS = 20, 36
IMG_H = IMG_W = 4
Z_NEAR, Z_FAR = 0.1, 2.0
IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)  # (x, y, z, w), the order used by isaaclab.utils.math in Isaac Lab 3

# render id -> class. Render id 0 is background; anything not listed here has no label.
RENDER_IDS = {"trunk": 10, "branch": 11, "spur": 12}


def prim_paths(env_id: int) -> dict[str, str]:
    root = f"/World/envs/env_{env_id}/tree/trunkId_0"
    return {"trunk": root, "branch": f"{root}/branchL_0_Id_000", "spur": f"{root}/spurId_0000"}


# custom (bin-row) id per class, as written by trial_setup.get_tree_metadata
CUSTOM_IDS = {"trunk": 0, "branch": 1, "spur": 2}


@pytest.fixture(scope="session")
def uv_rgb() -> np.ndarray:
    """uv.png as RGB uint8 (cv2 loads BGR)."""
    return cv2.imread(UV_IMG_PATH, cv2.IMREAD_UNCHANGED)[..., :3][..., ::-1].copy()


def bin_color(uv_rgb: np.ndarray, length_bin: int, theta_bin: int) -> np.ndarray:
    """RGB color of the center pixel of a (length, theta) bin in uv.png. Length runs along rows."""
    h, w = uv_rgb.shape[:2]
    row = int((length_bin + 0.5) * h / N_LENGTH_BINS)
    col = int((theta_bin + 0.5) * w / N_THETA_BINS)
    return uv_rgb[row, col].astype(np.float32)


@dataclass
class Pixel:
    """One synthetic pixel. Every pixel not given is background (render id 0, depth 0)."""

    cls: str | None  # "trunk" / "branch" / "spur", or None for an unlabelled render id
    length_bin: int = 0
    theta_bin: int = 0
    depth: float = 1.0
    incidence_deg: float = 0.0  # angle between the surface normal (flipped) and the sensor's forward axis


def render_pixels(
    uv_rgb: np.ndarray,
    images: list[list[Pixel]],
    forward_axis: np.ndarray = np.array([0.0, 0.0, 1.0]),
    tilt_axis: np.ndarray = np.array([1.0, 0.0, 0.0]),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (rgb, depth, normals, instance) arrays of shape (n_images, H, W[, 3]); see ``write_capture``."""
    n = len(images)
    rgb = np.zeros((n, IMG_H, IMG_W, 3), np.float32)
    depth = np.zeros((n, IMG_H, IMG_W), np.float32)
    normals = np.zeros((n, IMG_H, IMG_W, 3), np.float32)
    instance = np.zeros((n, IMG_H, IMG_W), np.float32)
    for i, image in enumerate(images):
        for k, px in enumerate(image):
            r, c = divmod(k, IMG_W)
            angle = np.radians(px.incidence_deg)
            # surface normal pointing back at the sensor, tilted by the incidence angle
            normals[i, r, c] = -(np.cos(angle) * forward_axis + np.sin(angle) * tilt_axis)
            depth[i, r, c] = px.depth
            rgb[i, r, c] = bin_color(uv_rgb, px.length_bin, px.theta_bin)
            instance[i, r, c] = RENDER_IDS[px.cls] if px.cls else 99
    return rgb, depth, normals, instance


def write_capture(
    path: str,
    uv_rgb: np.ndarray,
    pixels: list[list[list[Pixel]]],
    eef_positions: np.ndarray | None = None,
    sensor_quat: tuple = IDENTITY_QUAT,
    forward_axis: np.ndarray = np.array([0.0, 0.0, 1.0]),
    tilt_axis: np.ndarray = np.array([1.0, 0.0, 0.0]),
    convention: str = "ros",
) -> str:
    """
    Write a capture h5 with the same layout as CaptureDataLogger.

    :param pixels: pixels[env][pose] is the list of non-background pixels for that capture (at most IMG_H * IMG_W).
    :param eef_positions: (n_envs, n_poses, 3) EEF positions; defaults to poses spaced 1 cm apart along x.
    :param forward_axis: the sensor's optical axis in world frame for ``sensor_quat`` (used to build normals).
        Defaults to +Z: the optical axis of an identity quaternion in the ROS convention.
    :param tilt_axis: a world axis perpendicular to ``forward_axis``, used to tilt normals by ``incidence_deg``.
    :param convention: value of the sensor group's ``pose_w_convention`` attr.
    """
    n_envs, n_poses = len(pixels), len(pixels[0])
    rgb, depth, normals, instance = render_pixels(
        uv_rgb, [px for env in pixels for px in env], forward_axis, tilt_axis
    )
    shape = (n_envs, n_poses, IMG_H, IMG_W)
    rgb, depth, normals, instance = (
        rgb.reshape(shape + (3,)), depth.reshape(shape), normals.reshape(shape + (3,)), instance.reshape(shape)
    )

    if eef_positions is None:
        eef_positions = np.zeros((n_envs, n_poses, 3), np.float32)
        eef_positions[..., 0] = np.arange(n_poses) * 0.01
    eef_poses = np.concatenate([eef_positions, np.tile(IDENTITY_QUAT, (n_envs, n_poses, 1))], axis=-1)
    sensor_pose = np.concatenate([eef_positions, np.tile(sensor_quat, (n_envs, n_poses, 1))], axis=-1)

    with h5py.File(path, "w") as f:
        f.attrs["num_envs"] = n_envs
        f.create_dataset("poses/actual_poses/eef_poses", data=eef_poses.astype(np.float32))
        for e in range(n_envs):
            tree = f.require_group(f"trees/env_{e}")
            labels = tree.require_group("instance_labels")
            labels.attrs["0"] = "BACKGROUND"
            for cls, path_ in prim_paths(e).items():
                tree.attrs[path_] = CUSTOM_IDS[cls]
                labels.attrs[str(RENDER_IDS[cls])] = path_
        g = f.require_group("sensors/cam0")
        g.attrs.update({"height": IMG_H, "width": IMG_W, "z_near": Z_NEAR, "z_far": Z_FAR})
        g.attrs["pose_w_convention"] = convention
        g.create_dataset("pose_w", data=sensor_pose.astype(np.float32))
        g.create_dataset("albedo", data=rgb.astype(np.uint8))
        g.create_dataset("depth", data=depth)
        g.create_dataset("normals", data=normals)
        g.create_dataset("instance_segmentation", data=instance)
    return path


def make_evaluator(path: str, n_envs: int, **obs_kwargs) -> "nbv.NBVEvaluator":
    return nbv.NBVEvaluator(
        trial_data_path=path,
        observation_config=nbv.ObservationConfig(**obs_kwargs),
        gain_config=nbv.GainConfig(),
        device="cpu",
        n_envs=n_envs,
    )
