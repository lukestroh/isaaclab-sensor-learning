#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as R


def ori_vec_to_quat(ori_vecs: np.ndarray, base_vec: np.ndarray = np.array([0, 0, 1])) -> np.ndarray:
    """Convert orientation vectors to quaternions

    :param ori_vecs: Orientation vectors
    :type ori_vecs: np.ndarray
    :param base_vec: Base vector to align from, defaults to np.array([0, 0, 1])
    :type base_vec: np.ndarray, optional
    :return: Quaternions
    :rtype: np.ndarray
    """
    norms = np.linalg.norm(ori_vecs, axis=1, keepdims=True)
    ori_vecs = ori_vecs / norms
    angles = np.arccos(np.dot(ori_vecs, base_vec) / (np.linalg.norm(ori_vecs, axis=1) * np.linalg.norm(base_vec)))
    rot_vecs = np.cross(base_vec, ori_vecs)
    rot_vec_norms = np.linalg.norm(rot_vecs, axis=1, keepdims=True)
    rot_vec_norms = np.divide(rot_vecs, rot_vec_norms, where=(rot_vec_norms > 1e-10), out=np.zeros_like(rot_vecs))
    rot_vecs = rot_vec_norms * angles[:, np.newaxis]
    quats = R.from_rotvec(rot_vecs).as_quat()
    return quats


def generate_discrete_pts(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    x_size=5,
    y_size=5,
    z_size=5,
) -> np.ndarray:
    """Generate discrete points in a 3D space defined by the given ranges and sizes.

    :param x_range: Range for x-axis (min, max)
    :type x_range: tuple
    :param y_range: Range for y-axis (min, max)
    :type y_range: tuple
    :param z_range: Range for z-axis (min, max)
    :type z_range: tuple
    :param x_size: Number of points along x-axis, defaults to 5
    :type x_size: int, optional
    :param y_size: Number of points along y-axis, defaults to 5
    :type y_size: int, optional
    :param z_size: Number of points along z-axis, defaults to 5
    :type z_size: int, optional
    :return: Discrete points in the 3D space
    :rtype: np.ndarray
    """
    x = np.linspace(x_range[0], x_range[1], x_size)
    y = np.linspace(y_range[0], y_range[1], y_size)
    z = np.linspace(z_range[0], z_range[1], z_size)

    grid = np.meshgrid(x, y, z, indexing="ij")
    discrete_pts = np.vstack(list(map(np.ravel, grid))).T
    return discrete_pts


def generate_uniform_spherical_pts(
    theta_range: tuple[float, float], phi_range: tuple[float, float], size: int, start_orientation=np.array([0, 0, 1])
) -> np.ndarray:
    """Generate uniform points on a sphere defined by the given theta and phi ranges and size.

    :param theta_range: Range for polar angle theta (min, max) in radians
    :type theta_range: tuple
    :param phi_range: Range for azimuthal angle phi (min, max) in radians
    :type phi_range: tuple
    :param size: Number of points to generate per axis
    :type size: int
    :param start_orientation: Starting orientation vector to align with, defaults to np.array([0, 0, 1])
    :type start_orientation: np.ndarray, optional
    :return: Uniform points on the sphere
    :rtype: np.ndarray
    """
    theta = np.linspace(theta_range[0], theta_range[1], size)
    phi = np.linspace(phi_range[0], phi_range[1], size)

    grid = np.meshgrid(theta, phi, indexing="ij")
    spherical_pts = np.vstack(list(map(np.ravel, grid))).T

    vectors = np.empty((spherical_pts.shape[0], 3), dtype=np.float64)

    vectors[:, 0] = np.sin(spherical_pts[:, 0]) * np.cos(spherical_pts[:, 1])
    vectors[:, 1] = np.sin(spherical_pts[:, 0]) * np.sin(spherical_pts[:, 1])
    vectors[:, 2] = np.cos(spherical_pts[:, 0])

    # Align generated vectors with start_orientation vector
    rotation_axis = np.cross(np.array([0, 0, 1]), start_orientation)
    rotation_angle = np.arccos(np.dot(np.array([0, 0, 1]), start_orientation) / (np.linalg.norm(start_orientation)))
    rot_vecs = rotation_axis * rotation_angle
    rot_mats = R.from_rotvec(rot_vecs).as_matrix()
    vectors = (rot_mats @ vectors.T).T

    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def generate_discrete_poses(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    theta_range: tuple[float, float],
    phi_range: tuple[float, float],
    x_size: int,
    y_size: int,
    z_size: int,
    angles_size: int,
    start_orientation=np.array([0, 0, 1]),
) -> np.ndarray:
    """Generate discrete poses by combining discrete points in space with uniform orientations on a sphere.

    :param x_range: Range for x-axis (min, max)
    :type x_range: tuple
    :param y_range: Range for y-axis (min, max)
    :type y_range: tuple
    :param z_range: Range for z-axis (min, max)
    :type z_range: tuple
    :param theta_range: Range for polar angle theta (min, max) in radians
    :type theta_range: tuple
    :param phi_range: Range for azimuthal angle phi (min, max) in radians
    :type phi_range: tuple
    :param x_size: Number of points along x-axis
    :type x_size: int
    :param y_size: Number of points along y-axis
    :type y_size: int
    :param z_size: Number of points along z-axis
    :type z_size: int
    :param angles_size: Number of points to generate per angle axis
    :type angles_size: int
    :param start_orientation: Starting orientation vector to align with, defaults to np.array([0, 0, 1])
    :type start_orientation: np.ndarray, optional
    :return: Discrete poses combining positions and orientations
    :rtype: np.ndarray
    """
    points = generate_discrete_pts(x_range, y_range, z_range, x_size, y_size, z_size)
    orientations = generate_uniform_spherical_pts(
        theta_range=theta_range, phi_range=phi_range, size=angles_size, start_orientation=start_orientation
    )
    quats = ori_vec_to_quat(orientations)

    points_repeated = np.repeat(points, quats.shape[0], axis=0)
    quats_tiled = np.tile(quats, (points.shape[0], 1))
    poses = np.hstack((points_repeated, quats_tiled))

    # Return only unique poses
    poses = np.unique(poses, axis=0)
    return poses
