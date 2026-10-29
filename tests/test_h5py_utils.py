import h5py
import numpy as np

from isaaclab_sensor_learning.utils import h5py_utils


def _write(path):
    with h5py.File(path, "w") as f:
        f.attrs["num_envs"] = 2
        f.create_dataset("poses/eef_poses", data=np.ones((2, 3, 7)))
        cam = f.require_group("sensors/cam0")
        cam.attrs["height"] = 4
        cam.create_dataset("rgb", data=np.ones((2, 3, 4, 4, 3)))
        cam.require_group("nested").attrs["z_far"] = 2.0


def test_h5_to_dict_loads_everything_by_default(tmp_path):
    _write(tmp_path / "f.h5")
    with h5py.File(tmp_path / "f.h5", "r") as f:
        d = h5py_utils.h5_to_dict(f)
    assert d["attrs"]["num_envs"] == 2
    assert d["sensors"]["cam0"]["rgb"].shape == (2, 3, 4, 4, 3)


def test_attrs_only_skips_datasets_but_keeps_attrs(tmp_path):
    _write(tmp_path / "f.h5")
    with h5py.File(tmp_path / "f.h5", "r") as f:
        d = h5py_utils.h5_to_dict(f, attrs_only=("sensors",))
    cam = d["sensors"]["cam0"]
    assert "rgb" not in cam  # image data must stay on disk
    assert cam["attrs"]["height"] == 4
    assert cam["nested"]["attrs"]["z_far"] == 2.0
    # groups outside attrs_only are still fully loaded
    assert d["poses"]["eef_poses"].shape == (2, 3, 7)
