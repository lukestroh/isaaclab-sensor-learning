"""UV-color -> (length_bin, theta_bin) decoding. Coverage depends entirely on this mapping being right."""

import numpy as np
import pytest
import torch
from conftest import N_LENGTH_BINS, N_THETA_BINS, bin_color

from isaaclab_sensor_learning.nbv import nbv


@pytest.fixture(scope="module")
def tree():
    return nbv.BinnedTree(n_envs=1, unique_instances={"env_0": {"/a": 0}}, device="cpu")


def test_uv_image_encoding(uv_rgb):
    """uv.png: theta in R (along columns), length in G (along rows), both spanning 0-255, B = 0."""
    assert uv_rgb[..., 2].max() == 0
    assert uv_rgb[512, :, 0].min() == 0 and uv_rgb[512, :, 0].max() == 255
    assert uv_rgb[:, 256, 1].min() == 0 and uv_rgb[:, 256, 1].max() == 255
    assert np.all(np.diff(uv_rgb[512, :, 0].astype(int)) >= 0)  # R increases left -> right
    assert np.all(np.diff(uv_rgb[:, 256, 1].astype(int)) >= 0)  # G increases top -> bottom


def test_bin_center_colors_round_trip(tree, uv_rgb):
    bins = [(l, t) for l in range(N_LENGTH_BINS) for t in range(N_THETA_BINS)]
    colors = torch.tensor(np.stack([bin_color(uv_rgb, l, t) for l, t in bins]))
    assert tree.uv_color_to_bin(colors).tolist() == [list(b) for b in bins]


def test_decoding_every_pixel_of_uv_png_matches_its_region(tree, uv_rgb):
    """Every texel decodes to the bin of the region it sits in, up to one bin at region edges (8-bit quantization)."""
    h, w = uv_rgb.shape[:2]
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    expected = np.stack([rows * N_LENGTH_BINS // h, cols * N_THETA_BINS // w], -1)
    decoded = tree.uv_color_to_bin(torch.tensor(uv_rgb)).numpy()
    assert np.abs(decoded - expected).max() <= 1
    assert (decoded == expected).all(-1).mean() > 0.8


def test_full_range_reachable(tree):
    """Regression: the lit rgb render capped colors at ~0.6 x 255, making the top 40% of bins unreachable."""
    ramp = torch.arange(256, dtype=torch.uint8)
    colors = torch.stack([ramp, ramp, torch.zeros_like(ramp)], -1)
    bins = tree.uv_color_to_bin(colors)
    assert set(bins[:, 0].tolist()) == set(range(N_LENGTH_BINS))
    assert set(bins[:, 1].tolist()) == set(range(N_THETA_BINS))


def test_uint8_float_and_rgba_inputs_agree(tree, uv_rgb):
    color = bin_color(uv_rgb, 13, 30)
    as_uint8 = torch.tensor(color.astype(np.uint8))
    as_float = torch.tensor(color)
    as_rgba = torch.cat([as_uint8, torch.tensor([255], dtype=torch.uint8)])
    for c in (as_uint8, as_float, as_rgba):
        assert tree.uv_color_to_bin(c[None]).tolist() == [[13, 30]]


def test_dark_float_colors_are_not_rescaled(tree):
    """Colors are always 0-255. A dark patch (all values <= 1) must not be mistaken for normalized [0, 1] input."""
    assert tree.uv_color_to_bin(torch.tensor([[1.0, 1.0, 0.0]])).tolist() == [[0, 0]]
