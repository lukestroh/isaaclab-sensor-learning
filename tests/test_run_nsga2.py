"""run_nsga2.py end to end with --dry-run (synthetic evaluator): run, checkpoint, resume."""

import csv
import os
import runpy
import sys

import yaml

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, SCRIPTS)


def run_cli(*argv):
    old = sys.argv
    sys.argv = ["run_nsga2.py", *argv]
    try:
        runpy.run_path(os.path.join(SCRIPTS, "run_nsga2.py"), run_name="__main__")
    except SystemExit as e:
        assert not e.code
    finally:
        sys.argv = old


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_dry_run_checkpoint_and_resume(tmp_path):
    runs = str(tmp_path / "runs")
    run_cli("--dry-run", "--pop-size", "6", "--n-gen", "2", "--tag", "t", "--runs-dir", runs)
    (run_id,) = os.listdir(runs)
    root = os.path.join(runs, run_id)
    assert [r["generation"] for r in rows(os.path.join(root, "progress.csv"))] == ["1", "2"]
    n_after_first = len(rows(os.path.join(root, "index.csv")))

    run_cli("--resume", run_id, "--n-gen", "4", "--runs-dir", runs)
    # resumed run continues with generation 3 (not a repeat of 2)
    assert [r["generation"] for r in rows(os.path.join(root, "progress.csv"))] == ["1", "2", "3", "4"]
    assert sorted(os.listdir(os.path.join(root, "generations"))) == [f"gen_{g:04d}" for g in (1, 2, 3, 4)]
    index = rows(os.path.join(root, "index.csv"))
    assert len(index) > n_after_first and len({r["rig_id"] for r in index}) == len(index)  # nothing simulated twice
    for r in index:
        assert os.path.exists(os.path.join(root, "rigs", f"{r['rig_id']}.yaml"))
    with open(os.path.join(root, "run.yaml")) as f:
        run = yaml.safe_load(f)
    assert run["settings"]["pop_size"] == 6 and len(run["resumed"]) == 1
    assert rows(os.path.join(root, "pareto.csv"))  # non-empty front
