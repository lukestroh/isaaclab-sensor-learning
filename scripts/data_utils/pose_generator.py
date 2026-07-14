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


def generate_uniform_rectangular_pts(
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


def generate_uniform_spherical_vectors(
    theta_range: tuple[float, float],
    phi_range: tuple[float, float],
    theta_size: int,
    phi_size: int,
    start_orientation=np.array([0, 0, 1]),
) -> np.ndarray:
    """Generate uniform vectors on a sphere defined by the given theta and phi ranges and size.

    :param theta_range: Range for polar angle theta (min, max) in radians
    :type theta_range: tuple
    :param phi_range: Range for azimuthal angle phi (min, max) in radians
    :type phi_range: tuple
    :param theta_size: Number of points along the polar angle axis
    :type theta_size: int
    :param phi_size: Number of points along the azimuthal angle axis
    :type phi_size: int
    :param start_orientation: Starting orientation vector to align with, defaults to np.array([0, 0, 1])
    :type start_orientation: np.ndarray, optional
    :return: Uniform vectors on the sphere
    :rtype: np.ndarray
    """
    theta = np.linspace(theta_range[0], theta_range[1], theta_size)
    phi = np.linspace(phi_range[0], phi_range[1], phi_size)

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


def rotate_points_about_axis(
    points: np.ndarray, axis_point: np.ndarray, axis_direction: np.ndarray, psi: float
) -> np.ndarray:
    """
    Rotate `points` (N,3) by angle psi (radians) about the line:
        L(t) = axis_point + t * axis_direction
    Right-hand rule about axis_direction.
    """
    axis_point = np.asarray(axis_point, dtype=float)
    axis_dir = np.asarray(axis_direction, dtype=float)
    axis_dir = axis_dir / np.linalg.norm(axis_dir)

    # Rodrigues' rotation formula: R = I + sin(psi) K + (1 - cos(psi)) K^2
    K = np.array(
        [
            [0, -axis_dir[2], axis_dir[1]],
            [axis_dir[2], 0, -axis_dir[0]],
            [-axis_dir[1], axis_dir[0], 0],
        ]
    )
    R = np.eye(3) + np.sin(psi) * K + (1 - np.cos(psi)) * (K @ K)

    shifted = points - axis_point  # move axis to origin
    rotated = shifted @ R.T
    return rotated + axis_point  # move back


def generate_uniform_cylindrical_pts(
    r_range: tuple[float, float],
    theta_range: tuple[float, float],
    z_range: tuple[float, float],
    r_size=5,
    theta_size=5,
    z_size=5,
    start_orientation=np.array([[-1, 0, 0], [0, 0, -1], [-1, 0, 0]]),
) -> np.ndarray:
    r = np.linspace(r_range[0], r_range[1], r_size)
    z = np.linspace(z_range[0], z_range[1], z_size)

    r_max = r[-1]
    positions = []

    for ri in r:
        # scale angular sample count with r; guarantees at least 1 point (handles r=0)
        n_theta_i = max(1, round(theta_size * ri / r_max)) if r_max > 0 else theta_size
        theta_i = np.linspace(theta_range[0], theta_range[1], n_theta_i)

        x_i = ri * np.cos(theta_i)
        y_i = ri * np.sin(theta_i)
        ring_xy = np.stack([x_i, y_i], axis=-1)  # (n_theta_i, 2)

        # cross this ring's (x,y) points with all z levels
        ring_xy_rep = np.repeat(ring_xy, z_size, axis=0)  # (n_theta_i * z_size, 2)
        z_rep = np.tile(z, n_theta_i)  # (n_theta_i * z_size,)

        ring_pts = np.concatenate([ring_xy_rep, z_rep[:, None]], axis=-1)
        positions.append(ring_pts)

    cylindrical_pts = np.vstack(positions)

    axis_point = np.array([0,0,np.mean(z_range)])

    cylindrical_pts = rotate_points_about_axis(
        cylindrical_pts, axis_point=axis_point, axis_direction=np.array([-1, 0, 0]), psi=np.pi / 2
    )
    cylindrical_pts = rotate_points_about_axis(
        cylindrical_pts, axis_point=axis_point, axis_direction=np.array([0, 0, 1]), psi=np.pi / 2
    )

    return cylindrical_pts


def generate_uniform_cylindrical_poses(
    r_range: tuple[float, float],
    r_size: int,
    theta_cyl_range: tuple[float, float],
    theta_cyl_size: int,
    z_range: tuple[float, float],
    z_size: int,
    theta_ori_range: tuple[float, float],
    theta_ori_size: int,
    phi_ori_range: tuple[float, float],
    phi_ori_size: int,
    start_orientation=np.array([0, 0, 1]),
):
    """Generate uniform poses in a cylindrical shell defined by the given ranges and sizes."""
    points = generate_uniform_cylindrical_pts(r_range, theta_cyl_range, z_range, r_size, theta_cyl_size, z_size)
    orientations = generate_uniform_spherical_vectors(
        theta_range=theta_ori_range,
        phi_range=phi_ori_range,
        theta_size=theta_ori_size,
        phi_size=phi_ori_size,
        start_orientation=start_orientation,
    )
    quats = ori_vec_to_quat(orientations)

    points_repeated = np.repeat(points, quats.shape[0], axis=0)
    quats_tiled = np.tile(quats, (points.shape[0], 1))
    poses = np.hstack((points_repeated, quats_tiled))
    poses = np.unique(poses, axis=0)  # Return only unique poses
    return poses


def generate_uniform_rectangular_poses(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    theta_range: tuple[float, float],
    phi_range: tuple[float, float],
    x_size: int,
    y_size: int,
    z_size: int,
    theta_size: int,
    phi_size: int,
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
    :param theta_size: Number of points along the polar angle axis
    :type theta_size: int
    :param phi_size: Number of points along the azimuthal angle axis
    :type phi_size: int
    :param start_orientation: Starting orientation vector to align with, defaults to np.array([0, 0, 1])
    :type start_orientation: np.ndarray, optional
    :return: Discrete poses combining positions and orientations
    :rtype: np.ndarray
    """
    points = generate_uniform_rectangular_pts(x_range, y_range, z_range, x_size, y_size, z_size)
    orientations = generate_uniform_spherical_vectors(
        theta_range=theta_range,
        phi_range=phi_range,
        theta_size=theta_size,
        phi_size=phi_size,
        start_orientation=start_orientation,
    )
    quats = ori_vec_to_quat(orientations)

    points_repeated = np.repeat(points, quats.shape[0], axis=0)
    quats_tiled = np.tile(quats, (points.shape[0], 1))
    poses = np.hstack((points_repeated, quats_tiled))

    # Return only unique poses
    poses = np.unique(poses, axis=0)
    return poses


def main():
    import plotly.graph_objects as go
    from plot.plot_generated_poses import plot_generated_poses

    # # rectangular pose generation
    # x_range = (-0.5, 0.5)
    # y_range = (1.0, 1.6)
    # z_range = (1.2, 2.0)
    # theta_range = (-np.pi / 4, np.pi / 4)
    # phi_range = (-np.pi / 2, np.pi / 2)
    # x_size = 5
    # y_size = 5
    # z_size = 5
    # theta_size = 5
    # phi_size = 5
    # start_orientation = np.array([0, 1, 0])

    # poses = generate_uniform_rectangular_poses(
    #     x_range=x_range,
    #     y_range=y_range,
    #     z_range=z_range,
    #     theta_range=theta_range,
    #     phi_range=phi_range,
    #     x_size=x_size,
    #     y_size=y_size,
    #     z_size=z_size,
    #     theta_size=theta_size,
    #     phi_size=phi_size,
    #     start_orientation=start_orientation,
    # )

    # cylindrical pose generation
    r_range = (0.5, 1.5)
    r_size = 5
    theta_cyl_range = (-np.pi / 8, np.pi / 8)
    theta_cyl_size = 10
    z_range = (1.0, 1.8)
    z_size = 5
    theta_ori_range = (-np.pi / 4, np.pi / 4)
    theta_ori_size = 5
    phi_ori_range = (-np.pi / 2, np.pi / 2)
    phi_ori_size = 5
    start_orientation = np.array([0, 1, 0])
    poses = generate_uniform_cylindrical_poses(
        r_range=r_range,
        r_size=r_size,
        theta_cyl_range=theta_cyl_range,
        theta_cyl_size=theta_cyl_size,
        z_range=z_range,
        z_size=z_size,
        theta_ori_range=theta_ori_range,
        theta_ori_size=theta_ori_size,
        phi_ori_range=phi_ori_range,
        phi_ori_size=phi_ori_size,
        start_orientation=start_orientation,
    )

    fig = plot_generated_poses(poses, title="Generated Poses")

    fig.show()
    return


if __name__ == "__main__":
    main()
