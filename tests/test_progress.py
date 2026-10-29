"""Per-generation progress bar: within-rig progress from evaluate_rig.py's progress.json."""

import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from moo import progress as pg  # noqa: E402


def test_rig_fraction_and_description():
    assert pg.rig_fraction(None) == 0.0 and pg.describe(None) == "starting"
    half_capture = {"phase": "capture", "done": 100, "total": 200}
    assert pg.rig_fraction(half_capture) == pytest.approx(pg.CAPTURE_SHARE / 2)
    assert pg.describe(half_capture) == "capture 100/200"
    half_nbv = {"phase": "nbv", "t": 90.0, "budget": 180.0, "active_envs": 3}
    assert pg.rig_fraction(half_nbv) == pytest.approx(pg.CAPTURE_SHARE + (1 - pg.CAPTURE_SHARE) / 2)
    assert pg.describe(half_nbv) == "NBV 90/180 s (3 envs running)"
    assert pg.rig_fraction({"phase": "nbv", "t": 500.0, "budget": 180.0, "active_envs": 1}) < 1.0  # never "done" early


def test_generation_progress_counts_and_best(tmp_path):
    out = io.StringIO()
    progress = pg.GenerationProgress(poll_s=60, file=out)
    progress.start(generation=3, n_rigs=4)
    running = tmp_path / "rig_b"
    running.mkdir()
    (running / "progress.json").write_text(json.dumps({"phase": "capture", "done": 50, "total": 100}))
    progress.rig_done("rig_a", "ok", coverage=0.3)
    progress.rig_running("rig_b", str(running))
    progress.rig_done("rig_c", "infeasible")
    progress.rig_done("rig_d", "ok", coverage=0.5, cached=True)
    assert progress._bar.n == pytest.approx(3 + pg.CAPTURE_SHARE / 2)
    text = out.getvalue()
    assert "gen 3" in text and "rig_b: capture 50/100" in text and "best 0.500 (rig_d)" in text
    assert "ok 1 infeasible 1 cached 1" in text
    progress.close()
