#!/usr/bin/env python3
from typing import Union
import numpy as np


def get_fov_from_dfov(
    width: int,
    height: int,
    dfov: Union[int, float],
    degrees: bool = False,
):
    """
    Returns the vertical and horizontal field of view (FoV) in degrees given the diagonal field of view (dFoV) in degrees.
    https://www.litchiutilities.com/docs/fov.php
    https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/opengl-perspective-projection-matrix.html

    :param width (int): pixel width of the camera image
    :param height (int): pixel height of the camera image
    :param dfov (int/float): diagonal field of view of the camera (degrees).
    :param degrees (bool): whether to return the FoV in degrees. Default is False.
    """
    for key, val in locals().items():
        if key == "degrees":
            pass
        if val <= 0:
            raise ValueError(f"Parameter '{key}' cannot be less than 0. Value: {val}")
    if degrees:
        _dfov = np.deg2rad(dfov)
    else:
        _dfov = dfov
    camera_diag = np.sqrt(width**2 + height**2)
    fov_h = 2 * np.arctan(np.tan(_dfov / 2) * height / camera_diag)
    fov_w = 2 * np.arctan(np.tan(_dfov / 2) * width / camera_diag)
    if degrees:
        return np.rad2deg(fov_h), np.rad2deg(fov_w)
    else:
        return fov_h, fov_w


def get_intrinsic_matrix_from_dfov(
    width: int,
    height: int,
    dfov: Union[int, float],
    degrees: bool = True,
):
    """
    Returns the focal length in pixels given the diagonal field of view (dFoV) in degrees.
    https://www.litchiutilities.com/docs/fov.php
    https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/opengl-perspective-projection-matrix.html

    :param width (int): pixel width of the camera image
    :param height (int): pixel height of the camera image
    :param dfov (int/float): diagonal field of view of the camera (degrees).
    :param degrees (bool): whether the input dFoV is in degrees. Default is False.
    """
    for key, val in locals().items():
        if key == "degrees":
            pass
        if val <= 0:
            raise ValueError(f"Parameter '{key}' cannot be less than 0. Value: {val}")
    if degrees:
        _dfov = np.deg2rad(dfov)
    else:
        _dfov = dfov
    camera_diag_px = np.sqrt(width**2 + height**2)
    focal_length_px = camera_diag_px / (2 * np.tan(_dfov / 2))

    cx, cy = width / 2, height / 2

    K = np.array([[focal_length_px, 0, cx], [0, focal_length_px, cy], [0, 0, 1]])

    return K
