#!/usr/bin/env python3

import plotly.graph_objects as go
import numpy as np
from numpy.typing import ArrayLike


def plot_vector(
    fig: go.Figure,
    position: np.ndarray,
    orientation: np.ndarray,
    scale: float,
    color: str,
    anchor: str = "tail",
    name: str = "",
    showlegend: bool = False,
):

    # x=np.asarray([position[0]])
    # y=np.asarray([position[1]])
    # z=np.asarray([position[2]])
    # u=np.asarray([orientation[0]])
    # v=np.asarray([orientation[1]])
    # w=np.asarray([orientation[2]])

    # norm = np.sqrt()

    fig.add_trace(
        go.Cone(
            x=[position[0]],
            y=[position[1]],
            z=[position[2]],
            u=[orientation[0]],
            v=[orientation[1]],
            w=[orientation[2]],
            name=name,
            sizemode="scaled",
            sizeref=0.1 * scale,
            showscale=False,
            showlegend=showlegend,
            anchor=anchor,
            colorscale=[[0, color], [1, color]],
            customdata=np.column_stack((position, orientation)),
            hovertemplate=(
                "x: %{x}<br>"
                "y: %{y}<br>"
                "z: %{z}<br>"
                "u: %{u}<br>"
                "v: %{v}<br>"
                "w: %{w}<br>"
                "norm: %{norm}<br>"
                "<extra></extra>"
            ),
        )
    )

    return fig


def plot_vectors(
    fig: go.Figure,
    positions: np.ndarray,
    orientations: np.ndarray,
    scale: float,
    color: str,
    anchor: str = "tail",
    name: str = "",
    showlegend: bool = False,
):
    fig.add_trace(
        go.Cone(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            u=orientations[:, 0],
            v=orientations[:, 1],
            w=orientations[:, 2],
            name=name,
            sizemode="scaled",
            sizeref=0.1 * scale,
            showscale=False,
            showlegend=showlegend,
            anchor=anchor,
            colorscale=[[0, color], [1, color]],
            customdata=np.column_stack((positions, orientations)),
            hovertemplate=(
                "x: %{x}<br>" "y: %{y}<br>" "z: %{z}<br>" "u: %{u}<br>" "v: %{v}<br>" "w: %{w}<br>" "<extra></extra>"
            ),
        )
    )

    return fig


def plot_3d_coordinate_frame(
    position: np.ndarray,
    orientation: np.ndarray,
    axis_length: float = 0.05,
    cone_scale: float = 0.25,
    name: str = "",
    parent_frame: str = "",
    showlegend: bool = False,
    fig: go.Figure | None = None,
):
    """
    Plots a set of orthonormal vectors in a 3d space.

    :param fig: The plotly figure to add the coordinate frame to. If None, a new figure will be created.
    :param position: The position of the origin of the coordinate frame.
    :param orientation: The orientation of the coordinate frame as a 3x3 rotation matrix
    :param axis_length: The length of the axes.
    :param cone_scale: The scale of the cones at the end of the axes.
    :param name: The name of the coordinate frame.
    :param parent_frame: The name of the parent frame.
    :param showlegend: Whether to show the legend for the coordinate frame.
    :return: The plotly figure with the coordinate frame added.

    """
    if fig is None:
        fig = go.Figure()

    colors = {"x": "red", "y": "green", "z": "blue"}
    unit_vectors = {"x": np.array([1, 0, 0]), "y": np.array([0, 1, 0]), "z": np.array([0, 0, 1])}

    for axis_name, unit_vec in unit_vectors.items():
        direction = orientation @ unit_vec
        pos_end = position + axis_length * direction

        fig.add_trace(
            go.Scatter3d(
                x=[position[0], pos_end[0]],
                y=[position[1], pos_end[1]],
                z=[position[2], pos_end[2]],
                mode="lines",
                line=dict(width=3, color=colors[axis_name]),
                name=f"{name}__{axis_name}-axis",
                legendgroup=f"{name}_{axis_name}",
                legendgrouptitle=dict(text=name),
                showlegend=showlegend,
            )
        )

        fig.add_trace(
            go.Cone(
                x=[pos_end[0]],
                y=[pos_end[1]],
                z=[pos_end[2]],
                u=[cone_scale * direction[0]],
                v=[cone_scale * direction[1]],
                w=[cone_scale * direction[2]],
                anchor="tail",
                sizemode="scaled",
                sizeref=0.1 * cone_scale,
                showscale=False,
                colorscale=[[0, colors[axis_name]], [1, colors[axis_name]]],
                name=f"{name}__{axis_name}-axis",
                legendgroup=f"{name}_{axis_name}",
                legendgrouptitle=dict(text=name),
                showlegend=showlegend,
            )
        )

    return fig


def plot_circle(center: ArrayLike, radius: float, name: str = "", color: str = "black", fig: go.Figure = None):
    if fig is None:
        fig = go.Figure()

    theta = np.linspace(0, 2 * np.pi, 360)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)

    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=color), name=name))
    return fig


def plot_circle_3d(center: ArrayLike, radius: float, name: str = "", color: str = "black", fig: go.Figure = None):
    """TODO: be able to position circle in any plane"""
    if fig is None:
        fig = go.Figure()

    theta = np.linspace(0, 2 * np.pi, 360)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)

    fig.add_trace(go.Scatter3d(x=x, y=y, z=np.full(len(x), 0.0), mode="lines", line=dict(color=color), name=name))
    return fig


def plot_cylinder(
    center: ArrayLike,
    orientation: ArrayLike,
    radius,
    height,
    nt: int = 100,
    nh: int = 50,
    name: str = "",
    color: str = "#000000",
    opacity: float = 0.7,
    fig: go.Figure = None,
) -> go.Figure:
    if fig is None:
        fig = go.Figure()

    # first, create a cylinder that lies along the z-axis, then rotate and translate the points
    _theta = np.linspace(0, 2 * np.pi, nt)
    _z = np.linspace(-height / 2, height / 2, nh)

    theta, z_local = np.meshgrid(_theta, _z)

    x_local = radius * np.cos(theta)
    y_local = radius * np.sin(theta)

    z_axis = np.array([0, 0, 1])

    # ensure orientation is unit vector
    orientation = np.asarray(orientation) / np.linalg.norm(orientation)

    if np.allclose(orientation, z_axis):
        # Already aligned
        rotation_matrix = np.eye(3)
    elif np.allclose(orientation, -z_axis):
        # Opposite direction, rotate 180° around x-axis
        rotation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    else:
        v = np.cross(z_axis, orientation)
        s = np.linalg.norm(v)
        c = np.dot(z_axis, orientation)

        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

        rotation_matrix = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s**2))

    # Apply rotation and translation
    points_local = np.stack([x_local.flatten(), y_local.flatten(), z_local.flatten()])
    points_global = rotation_matrix @ points_local

    # Reshape back to grid form
    X_global = points_global[0].reshape(x_local.shape) + center[0]
    Y_global = points_global[1].reshape(y_local.shape) + center[1]
    Z_global = points_global[2].reshape(z_local.shape) + center[2]

    fig.add_trace(
        go.Surface(
            x=X_global,
            y=Y_global,
            z=Z_global,
            showscale=False,
            opacity=opacity,
            name=name,
            surfacecolor=np.ones_like(X_global),  # Constant color values
            colorscale=[[0, color], [1, color]],  # Single color
            showlegend=True,
        )
    )

    fig.update_layout(
        scene=dict(aspectmode="data"),
    )

    return fig


def plot_plane_from_point_and_normal_vec(
    point: np.ndarray, norm: np.ndarray, plane_size: int = 1, color: str = "#731c1c", fig: go.Figure = None
):
    if fig is None:
        fig = go.Figure()

    x_vals = np.linspace(-plane_size / 2, plane_size / 2, 10)
    y_vals = np.linspace(-plane_size / 2, plane_size / 2, 10)
    x, y = np.meshgrid(x_vals, y_vals)
    z = compute_z_on_plane(x=x, y=y, norm=norm, point_on_plane=point)

    # Create the surface
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            opacity=0.6,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            showlegend=True,
            name="branch_plane",
        )
    )

    return fig


def compute_z_on_plane(x, y, norm, point_on_plane):
    a, b, c = norm
    x0, y0, z0 = point_on_plane

    if c == 0:
        raise ValueError("The plane is vertical in z (normal.z = 0), z is undefined for given x, y")

    # Plane equation: a(x - x0) + b(y - y0) + c(z - z0) = 0
    z = ((-a * (x - x0)) - (b * (y - y0))) / c + z0
    return z


def get_tube_mesh_info(coefs, radius, u_vals, n_theta=16):
    # curve centers
    centers = (coefs @ np.vstack([u_vals**2, u_vals, np.ones_like(u_vals)])).T

    # tangents
    tgs = 2 * coefs[:, 0] * u_vals[:, None] + coefs[:, 1]
    tgs /= np.linalg.norm(tgs, axis=1)[:, None]

    # normals and binormals
    up = np.array([0, 0, 1.0])
    normals = np.cross(tgs, up)
    mask = np.linalg.norm(normals, axis=1) < 1e-6
    if mask.any():
        up2 = np.array([0, 1, 0])
        normals[mask] = np.cross(tgs[mask], up2)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    binormals = np.cross(tgs, normals)

    # vertices
    X, Y, Z = [], [], []
    for C, N, B in zip(centers, normals, binormals):
        for theta in np.linspace(0, 2 * np.pi, n_theta, endpoint=False):
            pt = C + radius * (np.cos(theta) * N + np.sin(theta) * B)
            X.append(pt[0])
            Y.append(pt[1])
            Z.append(pt[2])

    # triangle indices — FIXED winding order
    I, J, K = [], [], []
    n_u = len(u_vals)
    for i in range(n_u - 1):
        for j in range(n_theta):
            nj = (j + 1) % n_theta
            a = i * n_theta + j
            b = (i + 1) * n_theta + j
            c = i * n_theta + nj
            d = (i + 1) * n_theta + nj
            # correct winding order for consistent normals
            I += [a, b]
            J += [c, d]
            K += [b, c]

    return dict(x=X, y=Y, z=Z, i=I, j=J, k=K)


def plot_tube_mesh(tube_mesh_info: dict, opacity: float = 0.3, name: str = "", fig: go.Figure = None) -> go.Figure:
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Mesh3d(**tube_mesh_info, opacity=opacity, name=name))
    return fig


def main():

    fig = go.Figure()

    fig = plot_3d_coordinate_frame(fig=fig, position=np.zeros(3), orientation=None)

    fig = plot_vector(fig=fig, position=[1, 1, 1], orientation=[1, 1, 1], scale=0.5)

    fig.show()

    return


if __name__ == "__main__":
    main()
