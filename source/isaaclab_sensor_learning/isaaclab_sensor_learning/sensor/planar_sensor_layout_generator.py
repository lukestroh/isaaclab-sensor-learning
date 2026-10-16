#!/usr/bin/env python3

import plotly.graph_objects as go
import numpy as np
from dataclasses import dataclass
from typing import Any


@dataclass
class SensorLayoutGenerator:
    """A class to generate sensor layouts for end-effectors. The sensor layouts can be either spherical or planar.
    The class provides methods to generate the sensor layouts based on the specified parameters.
    """

    # sensor_model: str
    n_sensors: int


@dataclass
class PlanarSensorLayout(SensorLayoutGenerator):
    """Generates a sensor layout distributed over a flat rectangular plane.

    Sensors are arranged in a hexagonal packing pattern centered at the
    plane origin (0, 0, 0).

    Column count is aspect-ratio-aware: n_cols = round(sqrt(n * w / h)),
    so wide planes get more columns than rows and tall planes get more rows
    than columns.

    Coordinate convention:
        - Plane lies in the XY plane at z=0
        - Origin is at the center of the plane
        - +Z is the outward normal (sensor pointing direction)
        - X spans the width, Y spans the height

    Attributes:
        width:      Width of the plane in metres (X axis).
        height:     Height of the plane in metres (Y axis).
        n_sensors:  Total number of sensors to place.
    """

    width: float
    height: float
    sensors: list[dict]
    n_sensors: int
    max_iter: int = 200
    tol: float = 1e-6
    mesh_points: int = 8000

    def generate_sensor_layout(self) -> dict[str, Any]:
        """Generate a hexagonally packed planar sensor layout.

        Returns a dict with:
            poses           : Nx7 ndarray, columns [x, y, z, qx, qy, qz, qw]
            n_sensors       : int, actual number of sensors placed
            n_rows          : int, number of rows in the grid
            n_cols          : int, number of columns in the base grid
            sensors_per_row : list[int], sensor count per row
            width           : float, plane width (m)
            height          : float, plane height (m)
            col_spacing     : float, column spacing (m)
            row_spacing     : float, row spacing (m)
            sensor_type     : str
            sensor_dims     : tuple
            method          : str, 'hexagonal'
        """
        n = self.n_sensors
        w = self.width
        h = self.height

        if n <= 0:
            raise ValueError("n_sensors must be >= 1")
        if w <= 0 or h <= 0:
            raise ValueError("width and height must be positive")

        # --- Grid dimensions ---
        # Aspect-ratio-aware column count so spacing is isotropic.
        # Clamp to [1, n] to handle degenerate cases.
        n_cols = int(round(np.sqrt(n * w / h)))
        n_cols = max(1, min(n_cols, n))
        n_rows = int(np.ceil(n / n_cols))

        # Spacing between sensor centres
        col_spacing = w / n_cols
        row_spacing = h / n_rows

        # --- Build grid positions ---
        # Sensors are centered in the plane: offset so the grid midpoint
        # lands at (0, 0).
        sensor_z_axis = np.array([0.0, 0.0, 1.0])
        # Identity quaternion — all sensors point along +Z
        identity_quat = np.array([0.0, 0.0, 0.0, 1.0])

        poses = []
        sensors_per_row = []
        remaining = n

        for row_idx in range(n_rows):
            count = min(n_cols, remaining)
            remaining -= count
            sensors_per_row.append(count)

            # Stagger odd rows by half a column spacing (hexagonal offset)
            x_offset = (col_spacing / 2.0) if (row_idx % 2 == 1) else 0.0

            # Y position: centered in the plane
            # Row 0 is at the top (+Y), last row at bottom (-Y)
            y = (h / 2.0) - (row_idx + 0.5) * row_spacing

            # X positions: center the (possibly short) row within the plane
            # Full rows span symmetrically; short last rows are also centered.
            total_row_width = (count - 1) * col_spacing
            x_start = -total_row_width / 2.0 + x_offset

            for col_idx in range(count):
                x = x_start + col_idx * col_spacing
                position = np.array([x, y, 0.0])
                poses.append(np.concatenate([position, identity_quat]))

        poses_array = np.array(poses)  # Nx7

        return {
            "poses": poses_array,
            "n_sensors": int(poses_array.shape[0]),
            "n_rows": n_rows,
            "n_cols": n_cols,
            "sensors_per_row": sensors_per_row,
            "width": w,
            "height": h,
            "col_spacing": col_spacing,
            "row_spacing": row_spacing,
            "sensor_model": self.sensor_model,
            # "sensor_dims": self.sensor_dims,
            "method": "hexagonal",
        }


def plot_plane(pos: np.ndarray, fig: go.Figure = None):
    # Create a plane in the XY plane at z=0
    if fig is None:
        fig = go.Figure()
    x = np.array([-0.06, 0.06, 0.06, -0.06]) + pos[0]
    y = np.array([-0.06, -0.06, 0.06, 0.06]) + pos[1]
    z = np.array([0.0, 0.0, 0.0, 0.0]) + pos[2]

    fig.add_trace(go.Mesh3d(x=x, y=y, z=z, color="lightblue", opacity=0.5))

    return fig


def plot_sensor_layout(layout_dict: dict):
    poses = layout_dict["poses"]

    fig = go.Figure(
        data=[
            go.Scatter3d(x=poses[:, 0], y=poses[:, 1], z=poses[:, 2], mode="markers", marker=dict(size=3, color="blue"))
        ]
    )

    fig = plot_plane(pos=np.array([0.0, 0.0, 0.0]), fig=fig)

    fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"))

    fig.show()
    return


def main():
    lloyd_layout = PlanarSensorLayout(width=0.1, height=0.1, n_sensors=20, sensor_model="vl53l8cx")
    lloyd_layout_dict = lloyd_layout.generate_sensor_layout()
    print(lloyd_layout_dict)

    plot_sensor_layout(layout_dict=lloyd_layout_dict)

    return


if __name__ == "__main__":
    main()
