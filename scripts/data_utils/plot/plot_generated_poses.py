#!/usr/bin/env python3
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R

from . import plotly_helpers as ph


def plot_generated_poses(poses: np.ndarray, title: str = "Generated Poses", fig: go.Figure | None = None) -> go.Figure:
    """
    Plots the generated poses in 3D space using Plotly.

    :param poses: A numpy array of shape (N, 7) where each row represents a pose with [x, y, z, qx, qy, qz, qw].
    :param title: Title of the plot.
    :param fig: Existing Plotly figure to add traces to.
    """
    if fig is None:
        fig = go.Figure()

    fig = ph.plot_3d_coordinate_frame(fig=fig, position=np.zeros(3), orientation=np.identity(3), showlegend=True)

    # Add scatter plot for positions
    fig.add_trace(
        go.Scatter3d(
            x=poses[:, 0],
            y=poses[:, 1],
            z=poses[:, 2],
            mode="markers",
            marker=dict(size=5, color="blue"),
            name="Positions",
        )
    )

    # Add vectors for orientations
    mats = R.from_quat(poses[:, 3:7]).as_matrix()
    vecs = mats @ np.array([0, 0, 1])  # Example: plot z-axes of the orientations
    fig = ph.plot_vectors(fig=fig, positions=poses[:, :3], orientations=vecs, scale=1.0, color="red")

    # Set layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )

    return fig
