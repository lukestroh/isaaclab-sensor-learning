#!/usr/bin/env python3
"""
Per-env time and EEF path length to reach a reference coverage, the inputs to NSGA-II objective 2.

Pure numpy, so it can be unit-tested without Isaac Sim. Fed once per NBV step by scripts/evaluate_rig.py.
"""
import numpy as np


class ReferenceCoverageTracker:
    """
    Tracks, per env: cumulative EEF path length (only while the env is active), and the NBV time / path length at
    which its coverage first reached ``reference_coverage``.
    """

    def __init__(self, n_envs: int, reference_coverage: float):
        self.reference_coverage = float(reference_coverage)
        self.path_length = np.zeros(n_envs)  # m, cumulative while active
        self.time_to_reference = np.full(n_envs, np.nan)  # s (NBV time); NaN until reached
        self.path_to_reference = np.full(n_envs, np.nan)  # m; NaN until reached
        self._last_pos = None

    @property
    def reached(self) -> np.ndarray:
        return ~np.isnan(self.time_to_reference)

    def update(self, t: float, coverage: dict[int, float], eef_pos: np.ndarray, active_env_ids: list[int]) -> None:
        """
        :param t: NBV time (s).
        :param coverage: coverage per env id (same metric as the reference).
        :param eef_pos: (n_envs, 3) EEF positions.
        :param active_env_ids: envs that have not stopped; only these accumulate path.
        """
        eef_pos = np.asarray(eef_pos, dtype=float)
        if self._last_pos is not None and active_env_ids:
            active = np.asarray(active_env_ids)
            self.path_length[active] += np.linalg.norm(eef_pos[active] - self._last_pos[active], axis=-1)
        self._last_pos = eef_pos.copy()
        for env_id, value in coverage.items():
            if np.isnan(self.time_to_reference[env_id]) and value >= self.reference_coverage:
                self.time_to_reference[env_id] = t
                self.path_to_reference[env_id] = self.path_length[env_id]

    def summary(self) -> list[dict]:
        """One row per env (NaN -> None so it serializes to JSON null)."""

        def _f(x):
            return None if np.isnan(x) else float(x)

        return [
            {
                "reached_reference": bool(self.reached[i]),
                "time_to_reference_s": _f(self.time_to_reference[i]),
                "path_to_reference_m": _f(self.path_to_reference[i]),
                "path_length_m": float(self.path_length[i]),
            }
            for i in range(len(self.path_length))
        ]
