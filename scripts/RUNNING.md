# Running the sensor-learning pipeline

Run notes maintained by Claude; personal notes live in `RUN_HPC.md`. Commands assume the project root
(`source/isaaclab-sensor-learning`) and the IsaacLab venv (`../../venv/bin/python`). The system `python3` lacks
h5py / plotly / torch.

## Evaluate one rig (one Isaac Sim process: capture, then NBV)

```bash
../../venv/bin/python scripts/evaluate_rig.py --rig-yaml-path data/rigs/test_rig0.yaml
```

Runs are headless unless `--visualizer` is given (Isaac Lab 3 has no `--headless`).

| flag | default | meaning |
|---|---|---|
| `--rig-yaml-path` | `data/rigs/test_rig0.yaml` | rig to evaluate |
| `--num-envs` | task config (2) | trees evaluated in parallel |
| `--coverage-target` | `0.5` | stop an env once this fraction of instances has ≥ 5 hits; `none` = run to the time budget (NSGA) |
| `--reference-coverage` | `0.4` | records time / path to first reach this coverage (NSGA objective 2) |
| `--max-nbv-time-s` | `180` (TerminationConfig) | NBV sim-time budget |
| `--out-dir` | none | write `capture.h5`, `nbv.h5`, `metrics.json` there instead of `data/capture/`, `data/evaluation/` |
| `--quiet` | off | skip the per-step coverage prints (faster) |
| `--max-capture-poses` | all | debug: capture only the first N grid poses (use ≥ ~200 so candidates exist near the start pose) |
| `--seed` | `0` | seeds trees and poses; keep fixed across rigs |

Every run prints a `[TIMER]` line (setup / capture / NBV / total) that is also saved in the evaluation h5 under
`timing`.

Quick debug run:

```bash
../../venv/bin/python scripts/evaluate_rig.py --max-capture-poses 200 --max-nbv-time-s 20 --quiet
```

## Plot a run

```bash
../../venv/bin/python scripts/plot_nbv_run.py --show                 # newest data/evaluation/*.h5
../../venv/bin/python scripts/plot_nbv_run.py --eval 20261028_132532 # by datetime id
../../venv/bin/python scripts/plot_nbv_run.py --eval <out-dir>/nbv.h5  # an --out-dir run (finds capture.h5 next to it)
```

Writes `data/plots/<run>.html`: coverage grid, movement over time, and the 3D EEF path with tree markers. Only
runs recorded with the current data format plot (termination table, scene frames, coverage config).

## Tests

```bash
../../venv/bin/python -m pytest
```

They need no Isaac Sim (Omniverse UI modules are stubbed in `tests/conftest.py`).

## NSGA-II

`scripts/run_nsga2.py` is the top-level search. It never imports Isaac Sim: it evaluates each rig with its own
`evaluate_rig.py` subprocess (`--coverage-target none --out-dir data/runs/<run_id>/evals/<rig_id>`), pinned to a GPU
slot.

```bash
../../venv/bin/python scripts/run_nsga2.py --pop-size 20 --n-gen 30 --num-envs 20
../../venv/bin/python scripts/run_nsga2.py --resume <run_id> --n-gen 40           # continue / extend a run
../../venv/bin/python scripts/run_nsga2.py --dry-run --pop-size 8 --n-gen 3 --runs-dir /tmp/runs  # no simulation
```

| flag | default | meaning |
|---|---|---|
| `--pop-size`, `--n-gen` | 20, 30 | population; total generations (on `--resume`: run until this many) |
| `--search-space` | `scripts/moo/config/spherical_search_space.yaml` | radii, colatitudes, allowed models, max sensors |
| `--seed-rigs` | `data/rigs/test_rig0.yaml` | rigs placed in the initial population; pass nothing to disable |
| `--time-budget-s`, `--reference-coverage` | 180, 0.4 | NBV budget per rig; coverage for objective 2 |
| `--num-envs`, `--max-capture-poses` | task default, all | passed to `evaluate_rig.py` |
| `--gpus`, `--slots-per-gpu` | `0`, 1 | concurrent simulations (one subprocess per slot) |
| `--timeout-s` | none | wall-clock limit per rig (counted as failed) |
| `--seed` | 0 | GA seed |
| `--tag`, `--runs-dir` | none, `data/runs` | run id suffix; where runs are written |
| `--resume RUN_ID` | | continue from `checkpoint.pkl` (settings come from `run.yaml`) |
| `--dry-run` | off | synthetic results instead of simulating (checks the pipeline in seconds) |
| `--no-progress` | off | disable the per-generation progress bar |
| `--max-consecutive-failures` | 5 | stop after this many simulations fail in a row (fix, then `--resume`) |

Progress: each generation shows a bar that also advances within each rig. Each running evaluation's
`evals/<rig_id>/progress.json` reports capture pose i/N, then NBV time t/budget. After every generation a summary
line gives:
- rigs evaluated, feasible, and on the Pareto front;
- **HV**: hypervolume, the fraction of the objective box (ideal = full coverage, 0 s, $0; worst = no coverage,
  2 × budget, the most expensive rig) dominated by the feasible front. Higher is better, and it rises as the front
  improves;
- the best-coverage rig and its eval directory. Plot it with
  `plot_nbv_run.py --eval data/runs/<run_id>/evals/<rig_id>/nbv.h5`.

Run directory `data/runs/<run_id>/`:

- `run.yaml`: settings, git hash, search space, catalog snapshot.
- `index.csv`: one row per evaluated rig; this is the cache, so finished rigs are reused after an interruption.
- `progress.csv`: per generation: hypervolume, front size, best coverage.
- `generations/gen_XXXX/population.csv`: the population, with Pareto flags.
- `pareto.csv`: the final feasible front.
- `checkpoint.pkl`: the pymoo state, written after every generation.
- `rigs/<rig_id>.yaml`: the decoded rigs.
- `evals/<rig_id>/`: `capture.h5`, `nbv.h5`, `metrics.json`, `sim.log`. Plot one with
  `plot_nbv_run.py --eval data/runs/<run_id>/evals/<rig_id>/nbv.h5`.

Objectives (minimized):
1. −coverage: fraction of instances with ≥ 5 hits, mean over envs.
2. Time to the reference coverage. An env that never reaches it gets T·(1 + shortfall/c_ref).
3. Sensor cost.

Rigs whose Lloyd sites are closer than the sensors' footprints are infeasible and never simulated. A failed
simulation (crash, CUDA OOM, timeout) is infeasible too, and is logged in `index.csv` and `evals/<rig_id>/sim.log`.
