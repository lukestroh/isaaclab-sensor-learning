"""Logger helpers that don't need a running sim."""

import os
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from data_utils import capture_data_logger, evaluation_data_logger  # noqa: E402

LOGGERS = [capture_data_logger.CaptureDataLogger, evaluation_data_logger.EvaluationDataLogger]


@pytest.mark.parametrize("logger_cls", LOGGERS)
def test_save_scene_frames(tmp_path, logger_cls):
    path = tmp_path / "f.h5"
    logger = SimpleNamespace(datafile_path=str(path))  # only datafile_path is used
    origins = torch.tensor([[2.5, 0.0, 0.0], [-2.5, 0.0, 0.0]])
    base = torch.tensor([[2.5, -0.5, 0.0, 0.0, 0.0, 0.0, 1.0], [-2.5, -0.5, 0.0, 0.0, 0.0, 0.0, 1.0]])
    logger_cls.save_scene_frames(logger, env_origins=origins, robot_base_pose_w=base)
    logger_cls.save_scene_frames(logger, env_origins=origins + 1, robot_base_pose_w=base)  # re-saving overwrites
    with h5py.File(path, "r") as f:
        np.testing.assert_allclose(f["scene_frames/env_origins"][()], (origins + 1).numpy())
        np.testing.assert_allclose(f["scene_frames/robot_base_pose_w"][()], base.numpy())
        assert f["scene_frames"].attrs["quaternion_order"] == "xyzw"


def test_save_termination_with_disabled_coverage_target(tmp_path):
    path = tmp_path / "f.h5"
    logger = SimpleNamespace(datafile_path=str(path))
    rows = [{"env_id": 0, "reason": 2, "reason_name": "TIME_BUDGET", "t_end": 10.0}]
    evaluation_data_logger.EvaluationDataLogger.save_termination(
        logger, rows=rows, config={"coverage_target": None, "max_sim_time_s": 180.0}
    )
    with h5py.File(path, "r") as f:
        assert f["termination"].attrs["config/coverage_target"] == "none"
        assert f["termination"].attrs["config/max_sim_time_s"] == 180.0
        assert f["termination/reason_name"][0] == b"TIME_BUDGET"
