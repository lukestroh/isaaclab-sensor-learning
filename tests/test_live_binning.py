"""Live binning from CameraData during the NBV loop: frame-rate limiting and agreement with predicted binning."""

import numpy as np
import torch
from conftest import IDENTITY_QUAT, RENDER_IDS, Pixel, make_evaluator, prim_paths, render_pixels, write_capture


ENV_ID_OFFSET = 1000


class FakeCameraData:
    """
    The subset of isaaclab CameraData that NBVEvaluator reads, for all envs of one sensor (images[e] is env e).

    Live render ids are unique across envs (one idToLabels for the whole stage), so env e's ids are offset by
    ``ENV_ID_OFFSET * e``.
    """

    def __init__(self, uv_rgb, images: list[list[Pixel]], n_envs: int):
        rgb, depth, normals, instance = render_pixels(uv_rgb, images)
        offsets = (np.arange(n_envs) * ENV_ID_OFFSET)[:, None, None]
        instance = np.where(instance > 0, instance + offsets, instance)
        alpha = np.full(rgb.shape[:-1] + (1,), 255, np.uint8)
        self.output = {
            "albedo": torch.tensor(np.concatenate([rgb.astype(np.uint8), alpha], -1)),  # RGBA uint8, like the AOV
            "depth": torch.tensor(depth)[..., None],
            "normals": torch.tensor(normals),
            "instance_segmentation": torch.tensor(instance.astype(np.int32))[..., None],
        }
        self.quat_w_ros = torch.tensor([IDENTITY_QUAT] * n_envs, dtype=torch.float32)
        labels = {"0": "BACKGROUND"}
        for e in range(n_envs):
            labels.update({str(RENDER_IDS[cls] + ENV_ID_OFFSET * e): path for cls, path in prim_paths(e).items()})
        self.info = {"instance_segmentation": {"idToLabels": labels}}


PIXELS = [Pixel("trunk", 2, 3), Pixel("spur", 7, 30, incidence_deg=30.0), Pixel("branch", 1, 1, incidence_deg=60.0)]


def _evaluator(tmp_path, uv_rgb, n_envs=2):
    path = write_capture(str(tmp_path / "c.h5"), uv_rgb, [[PIXELS] for _ in range(n_envs)])
    return make_evaluator(path, n_envs=n_envs)


def _total(ev, env):
    return int(ev.binned_tree.bin_counts[f"env_{env}"].sum())


def test_live_and_predicted_binning_agree(tmp_path, uv_rgb):
    """The same pixels must produce the same counts whether predicted from the capture file or observed live."""
    with _evaluator(tmp_path, uv_rgb) as ev:
        (predicted,) = ev._get_pose_deltas(0, np.array([0]))
        obs = {"cam0": FakeCameraData(uv_rgb, [PIXELS, PIXELS], n_envs=2)}
        _, dense, _ = ev.update_bin_count_scores_from_observation(torch.arange(2), obs)
        np.testing.assert_array_equal(dense["env_0"].numpy(), predicted.to_dense().numpy())
        assert _total(ev, 0) == 2  # the 60-degree branch pixel is rejected by the shared incidence limit


def test_unchanged_frame_counters_are_not_binned_again(tmp_path, uv_rgb):
    with _evaluator(tmp_path, uv_rgb) as ev:
        obs = {"cam0": FakeCameraData(uv_rgb, [PIXELS, PIXELS], n_envs=2)}
        step = lambda frame: ev.update_bin_count_scores_from_observation(  # noqa: E731
            torch.arange(2), obs, frames={"cam0": torch.tensor(frame)}
        )
        step([1, 1])
        assert (_total(ev, 0), _total(ev, 1)) == (2, 2)
        step([1, 1])  # physics stepped, camera did not re-render: same frame again
        assert (_total(ev, 0), _total(ev, 1)) == (2, 2)
        step([1, 2])  # only env 1 rendered a new frame
        assert (_total(ev, 0), _total(ev, 1)) == (2, 4)


def test_without_frame_counters_every_call_is_binned(tmp_path, uv_rgb):
    with _evaluator(tmp_path, uv_rgb) as ev:
        obs = {"cam0": FakeCameraData(uv_rgb, [PIXELS, PIXELS], n_envs=2)}
        for _ in range(3):
            ev.update_bin_count_scores_from_observation(torch.arange(2), obs)
        assert _total(ev, 0) == 6
