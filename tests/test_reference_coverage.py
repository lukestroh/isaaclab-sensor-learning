"""Time / path to the reference coverage (NSGA objective 2 inputs)."""

import numpy as np
import pytest

from isaaclab_sensor_learning.nbv.reference_coverage import ReferenceCoverageTracker


def test_records_first_crossing_and_path_only_while_active():
    tr = ReferenceCoverageTracker(n_envs=2, reference_coverage=0.4)
    pos = np.zeros((2, 3))
    for step, (cov0, cov1) in enumerate([(0.1, 0.1), (0.3, 0.2), (0.45, 0.2), (0.5, 0.3)]):
        pos = pos + np.array([[1.0, 0, 0], [0, 2.0, 0]])  # env 0 moves 1 m/step, env 1 moves 2 m/step
        active = [0, 1] if step < 3 else [0]  # env 1 stops before the last step
        tr.update(t=float(step), coverage={0: cov0, 1: cov1}, eef_pos=pos, active_env_ids=active)
    rows = tr.summary()
    assert rows[0] == {"reached_reference": True, "time_to_reference_s": 2.0, "path_to_reference_m": 2.0,
                       "path_length_m": 3.0}
    # env 1 never reached 0.4; its last step (after stopping) adds no path
    assert rows[1] == {"reached_reference": False, "time_to_reference_s": None, "path_to_reference_m": None,
                       "path_length_m": pytest.approx(4.0)}
