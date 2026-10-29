#!/usr/bin/env python3
from dataclasses import dataclass
import plotly.graph_objects as go

import numpy as np
from typing import Any


def _quaternion_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the quaternion that rotates unit vector `a` onto unit vector `b`.

    Returns quaternion as [x, y, z, w].
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = np.clip(np.dot(a, b), -1.0, 1.0)

    if dot > 1.0 - 1e-9:
        # Already aligned
        return np.array([0.0, 0.0, 0.0, 1.0])
    if dot < -1.0 + 1e-9:
        # Anti-parallel: rotate 180 degrees around any perpendicular axis
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0])

    axis = np.cross(a, b)
    s = np.sqrt((1.0 + dot) * 2.0)
    return np.array([axis[0] / s, axis[1] / s, axis[2] / s, s / 2.0])


def _spherical_centroid(points: np.ndarray, radius: float) -> np.ndarray:
    """Compute the spherical centroid of a set of surface points.

        The centroid is the mean of the Cartesian coordinates re-normalised
        to lie on the sphere surface at the given radius.
    ls

        Args:
            points: (K, 3) array of points on the sphere surface.
            radius: Sphere radius in metres.

        Returns:
            (3,) centroid point on the sphere surface.
    """
    mean = points.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm < 1e-12:
        # Degenerate case: points cancel out — return the first point unchanged
        return points[0]
    return mean / norm * radius


def _sample_cap_mesh(colatitude_max: float, radius: float, m_points: int) -> np.ndarray:
    """Sample M points uniformly over a spherical cap surface using Fibonacci.

    Used as the discretized surface for Lloyd's Voronoi assignment step.

    Args:
        colatitude_max: Half-angle of the cap in radians.
        radius:         Sphere radius in metres.
        m_points:       Number of mesh points to sample.

    Returns:
        (M, 3) array of unit-radius points scaled to `radius`.
    """
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z_min = np.cos(colatitude_max)
    k = np.arange(m_points)
    z = 1.0 - (k + 0.5) / m_points * (1.0 - z_min)
    phi = golden_angle * k
    rho = np.sqrt(1.0 - z**2)
    pts = np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=1)
    return pts * radius  # (M, 3)


@dataclass
class LloydSphereSensorLayout:
    """Generates a sensor layout over a spherical cap using Lloyd's algorithm
    (centroidal Voronoi tessellation) with a fixed pole sensor.

    The pole sensor is pinned at (0, 0, radius) throughout relaxation.
    The remaining N-1 free sensors are initialised from a Fibonacci spiral
    and iteratively relaxed toward the centroids of their Voronoi cells on
    a discretized cap mesh.

    Convergence is declared when the maximum movement of any free sensor
    in a single iteration drops below `tol * radius`.

    Note: the result is reproducible given the same inputs (deterministic
    initialisation and mesh), but is a local CVT minimum, not guaranteed
    globally optimal.

    Attributes:
        radius:      Radius of the sphere in metres.
        colatitude:  Half-angle of the spherical cap in degrees.
        n_sensors:   Total number of sensors to place.
        max_iter:    Maximum number of Lloyd iterations (default 200).
        tol:         Convergence tolerance as a fraction of radius (default 1e-6).
        mesh_points: Number of mesh points for Voronoi discretisation (default 8000).
    """

    sensors: list[dict]
    n_sensors: int
    radius: float
    colatitude: float  # degrees
    max_iter: int = 200
    tol: float = 1e-6
    mesh_points: int = 8000

    def generate_sensor_layout(self) -> dict[str, Any]:
        """Generate a Lloyd-relaxed sensor layout over a spherical cap.

        Returns a dict with:
            poses            : Nx7 ndarray, columns [x, y, z, qx, qy, qz, qw]
            n_sensors        : int, actual number of sensors placed
            radius           : float, sphere radius (m)
            colatitude_deg   : float, cap half-angle (degrees)
            sensor_type      : str
            sensor_dims      : tuple
            method           : str, 'lloyd'
            iterations       : int, number of Lloyd iterations run
            converged        : bool, whether convergence threshold was met
            final_max_move   : float, max sensor movement in last iteration (m)
        """
        n = self.n_sensors
        r = self.radius
        colatitude_rad = np.deg2rad(self.colatitude)

        if n <= 0:
            raise ValueError("n_sensors must be >= 1")

        sensor_z_axis = np.array([0.0, 0.0, 1.0])

        # --- Pole sensor: always fixed at (0, 0, r) ---
        pole = np.array([0.0, 0.0, r])
        n_free = n - 1

        if n_free == 0:
            pose = np.array([0.0, 0.0, r, 0.0, 0.0, 0.0, 1.0])
            return self._build_result(pose[np.newaxis, :], iterations=0, converged=True, final_max_move=0.0)

        # --- Initialise free sensors from Fibonacci on the cap ---
        # Exclude z=1 (pole) since that position is already taken.
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        z_min = np.cos(colatitude_rad)
        k = np.arange(n_free)
        # Offset by 1 so k=0 doesn't land at z=1 (pole already reserved)
        z_init = 1.0 - (k + 1.0) / (n_free + 1.0) * (1.0 - z_min)
        phi_init = golden_angle * k
        rho_init = np.sqrt(1.0 - z_init**2)
        free = (
            np.stack(
                [rho_init * np.cos(phi_init), rho_init * np.sin(phi_init), z_init],
                axis=1,
            )
            * r
        )  # (n_free, 3)

        # --- Cap mesh for Voronoi assignment ---
        mesh = _sample_cap_mesh(colatitude_rad, r, self.mesh_points)  # (M, 3)

        # --- Lloyd iterations ---
        converged = False
        iterations = 0
        final_max_move = 0.0
        abs_tol = self.tol * r

        # All sensor positions including fixed pole: shape (n, 3)
        # Pole is index 0, free sensors are indices 1..n_free
        for it in range(self.max_iter):
            iterations = it + 1
            all_sensors = np.vstack([pole[np.newaxis, :], free])  # (n, 3)

            # Voronoi assignment: for each mesh point, find nearest sensor
            # using dot product (equivalent to geodesic distance on a sphere)
            # mesh: (M, 3), all_sensors: (n, 3)
            # dots[i, j] = mesh[i] . all_sensors[j]  (proportional to cos(angle))
            dots = mesh @ all_sensors.T  # (M, n)
            assignments = np.argmax(dots, axis=1)  # (M,) index into all_sensors

            # Update free sensors only (skip index 0 = pole)
            new_free = np.empty_like(free)
            max_move = 0.0
            for j in range(n_free):
                sensor_idx = j + 1  # offset because pole is index 0
                mask = assignments == sensor_idx
                if mask.sum() == 0:
                    # Sensor has no assigned mesh points — leave it in place
                    new_free[j] = free[j]
                    continue
                centroid = _spherical_centroid(mesh[mask], r)
                move = np.linalg.norm(centroid - free[j])
                max_move = max(max_move, move)
                new_free[j] = centroid

            final_max_move = max_move
            free = new_free

            if max_move < abs_tol:
                converged = True
                break

        # --- Build final poses ---
        all_sensors = np.vstack([pole[np.newaxis, :], free])  # (n, 3)
        poses = []
        for pt in all_sensors:
            outward = pt / np.linalg.norm(pt)
            quat = _quaternion_from_two_vectors(sensor_z_axis, outward)
            poses.append(np.concatenate([pt, quat]))

        poses_array = np.array(poses)  # Nx7
        return self._build_result(
            poses_array,
            iterations=iterations,
            converged=converged,
            final_max_move=float(final_max_move),
        )

    def _build_result(
        self,
        poses: np.ndarray,
        iterations: int,
        converged: bool,
        final_max_move: float,
    ) -> dict[str, Any]:
        return {
            "poses": poses,
            "n_sensors": int(poses.shape[0]),
            "radius": self.radius,
            "colatitude_deg": self.colatitude,
            "sensors": self.sensors,
            # "sensor_model": self.sensor_model,
            # "sensor_dims": self.sensor_dims,
            "method": "lloyd",
            "iterations": iterations,
            "converged": converged,
            "final_max_move_m": final_max_move,
        }


def plot_spherical_sector(radius, colatitude_deg, n_theta=30, n_phi=60, fig: go.Figure = None):
    if fig is None:
        fig = go.Figure()
    colatitude_rad = np.deg2rad(colatitude_deg)
    theta = np.linspace(0, colatitude_rad, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    x = radius * np.sin(theta_grid) * np.cos(phi_grid)
    y = radius * np.sin(theta_grid) * np.sin(phi_grid)
    z = radius * np.cos(theta_grid)

    # get rid of the color bar
    fig.add_trace(
        go.Surface(x=x, y=y, z=z, opacity=0.3, showscale=False, colorscale="Viridis", name="Spherical Sector")
    )
    return fig


def plot_sensor_layout(layout_dict: dict):
    radius = layout_dict["radius"]
    colatitude = layout_dict["colatitude_deg"]
    poses = layout_dict["poses"]

    fig = go.Figure(
        data=[
            go.Scatter3d(x=poses[:, 0], y=poses[:, 1], z=poses[:, 2], mode="markers", marker=dict(size=3, color="blue"))
        ]
    )

    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[radius * np.cos(np.radians(colatitude))],
            mode="markers",
            marker=dict(size=8, color="red"),
            name="Origin",
        )
    )

    fig = plot_spherical_sector(radius, colatitude, fig=fig)

    fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"))

    fig.show()
    return


def main():
    lloyd_layout = LloydSphereSensorLayout(
        sensor_model="vl53l8cx", n_sensors=13, radius=0.05, colatitude=np.degrees(np.pi / 8)
    )
    lloyd_layout_dict = lloyd_layout.generate_sensor_layout()
    print(lloyd_layout_dict)

    plot_sensor_layout(layout_dict=lloyd_layout_dict)

    return


if __name__ == "__main__":
    main()
