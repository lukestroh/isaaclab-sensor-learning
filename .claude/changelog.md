# Changelog

Code changes made while working through `.claude/plan.md`. Newest first.

## 2026-10-29: NSGA progress bar, normalized hypervolume, best rig in the summary

- `evaluate_rig.py`: with `--out-dir`, writes `progress.json` atomically:
  - `{"phase": "capture", "done", "total"}` every 10 poses;
  - `{"phase": "nbv", "t", "budget", "active_envs"}` every ~1 s of NBV sim time.
- New `scripts/moo/progress.py` `GenerationProgress`: a tqdm bar per generation (one unit per rig).
  - A running rig counts by its `progress.json`: capture fills its first 40% (`CAPTURE_SHARE`), NBV the rest.
  - Polled every 2 s, so the bar and ETA move within long rigs.
  - The text after the bar shows counts (ok / failed / infeasible / cached), each running rig's phase, and the best
    coverage so far in the generation, with its rig_id.
- `SensorRigProblem(progress=...)`: `evaluate()` wraps each generation's batch in the bar, and rigs report
  running / done. `progress` is excluded from pickling and reattached via `attach()`.
- `CheckpointCallback`:
  - prints HV **normalized** to [0, 1]: HV divided by the volume of the box between the ideal point (−1, 0, 0) and
    the reference point (0, 2T, max cost). That is a constant per run, so normalized values stay comparable across
    generations.
  - prints the number of feasible rigs, and the best-coverage rig's id and eval directory.
  - `progress.csv` gains `hv_normalized` and `best_rig` (the raw `hypervolume` is kept).
- `run_nsga2.py --no-progress` disables the bar.
- Verified on a real 2-rig run: the bar advanced through `capture 30/200 → 50/200`, then `NBV 0 → 10/10 s`, with an
  ETA.
- Tests: new `tests/test_progress.py` (94 total).

## 2026-10-28: `--num-envs` fix; NSGA failure handling

- **Bug fix, `--num-envs` other than 2 always failed.** `SensorLearningEnvCfg` had its own `num_envs = 2` field,
  separate from `scene.num_envs`. `parse_env_cfg(num_envs=...)` only sets the scene's count.
  - `trial_setup.get_tree_metadata` built per-env maps from the stale value: `KeyError: 'env_2'`. This failed all
    45 simulated rigs of the user's run `nsga_20261028_154845` (20 envs).
  - `SensorLearningEnv` sized the DiffIK controller from it too.
  - The duplicate field is removed; both places now use `env.num_envs` (= `scene.num_envs`).
- Verified: `evaluate_rig.py --num-envs 20` (200 poses, 10 s, test_rig0) completes on the 8 GB laptop GPU. Coverage
  0.052–0.086 across the 20 trees.
  - Timing vs 2 envs: capture 0.609 vs 0.089 s/pose; NBV real-time factor 0.046 vs 0.30.
  - Extrapolated full evaluation (4200 poses, 180 s): ~43 + 65 min ≈ 1.8 h per rig.
- `RunDir` no longer reuses `failed` rows after a restart (the cause may be fixed); `ok` and `infeasible` rows are
  still cached.
- `SensorRigProblem(max_consecutive_failures=5)` / `run_nsga2.py --max-consecutive-failures`: stops the run with the
  last error and a `--resume` hint after N simulation failures in a row (infeasible rigs don't count).
- Tests: consecutive-failure stop; failed rigs retried after restart (92 total).

## 2026-10-28: NSGA-II steps 3–5 (problem, operators, checkpointing, CLI)

- `scripts/moo/objectives.py`:
  - `objectives_from_metrics`: F = [−mean coverage, mean time objective, cost], plus summary stats.
  - `time_objective`: time to reference, or T·(1 + (c_ref − c_final)/c_ref) if not reached.
  - `placeholder_objectives` = [0, 2T, cost] for rigs that are not simulated or that fail.
- `scripts/moo/evaluators.py`:
  - `SubprocessEvaluator` runs one `evaluate_rig.py` per rig with `--coverage-target none --out-dir ... --quiet`,
    under `CUDA_VISIBLE_DEVICES` from a GPU-slot queue, with an optional timeout. Its log goes to
    `evals/<rig_id>/sim.log`.
  - `SyntheticEvaluator` (for `--dry-run`).
- `scripts/moo/run_dir.py`: `RunDir` (run layout, thread-safe `index.csv` cache) and `new_run_id`.
- `scripts/moo/operators.py`:
  - `RigSampling`: seeds first, then uniform random.
  - `RigCrossover`: uniform per gene. This replaces the planned SBX, since there are only 5–20 values per gene.
  - `RigMutation`: platform genes ±1 (count ±2 at times); model genes random-reset with p = 1/n; always changes
    the rig.
  - `RigRepair`: round and clip.
  - `RigDuplicateElimination`: canonical genome.
- `scripts/moo/sensor_cfg_problem.py` rewritten as `SensorRigProblem` (`ElementwiseProblem`, int genome, 3
  objectives, 2 constraints: `G_spacing`, `G_failed`):
  - infeasible rigs are never simulated;
  - concurrent duplicate rigs wait for one simulation;
  - runtime parts are excluded from pickling and reattached with `attach()`.
- `scripts/moo/callbacks.py`: `CheckpointCallback`. Per generation it writes `population.csv` (rank, crowding,
  Pareto flag), appends `progress.csv` (hypervolume with ref point [0, 2T, max cost]), sets
  `problem.generation`, and writes `checkpoint.pkl` atomically. `load_checkpoint` adds the `n_iter` increment
  pymoo performs after the callback, so resumed runs continue with the next generation (a bug found in the dry
  run).
- `scripts/moo/rig_codec.py`: `genome_from_rig_yaml` seeds hand-written rigs (`test_rig0` → `rig_c33d114b43`: r 0.20,
  colatitude 20°, 8 sensors).
- `scripts/run_nsga2.py` rewritten as the CLI:
  - `--pop-size`, `--n-gen`, `--seed`, `--seed-rigs`, `--time-budget-s`, `--reference-coverage`, `--num-envs`,
    `--max-capture-poses`, `--gpus`, `--slots-per-gpu`, `--timeout-s`, `--tag`, `--runs-dir`, `--resume`,
    `--dry-run`.
  - Writes `run.yaml` (settings, git hash, search space, catalog snapshot, resume log) and `pareto.csv`.
- Tests: `test_nsga_problem.py` (13: objectives, cache, infeasible/failed rigs, persistence, pickling, operators,
  NSGA2 serial and threaded) and `test_run_nsga2.py` (dry run, checkpoint, resume) (90 total).
- A dry run during development accidentally wrote `data/runs/nsga_20261028_153450_dryrun`; it was deleted, which
  is why `--runs-dir` exists.
- `scripts/RUNNING.md`: NSGA-II section.
- **Real smoke run** (pop 4, 1 generation, 200 capture poses, 10 s budget, 2 envs, laptop GPU), all 4 rigs
  through real `evaluate_rig.py` subprocesses:
  - seeded test_rig0 (8 sensors): coverage 0.050, 64 s wall;
  - 1 sensor: coverage 0.005, 34 s;
  - 19 sensors: coverage 0.108, 111 s;
  - 1 rig infeasible (4 sensors on a 10° cap): not simulated.
  - Per-rig cost scales with sensor count: capture 0.037 → 0.172 s/pose and NBV real-time factor 0.64 → 0.17 from
    1 → 19 sensors.

## 2026-10-28: NSGA-II steps 1–2 (rig codec, single-rig evaluation script)

- pymoo 0.6.2 and dill 0.4.1 installed into the IsaacLab venv.
- **Step 1:** new `scripts/moo/rig_codec.py`, standalone (no Isaac Sim):
  - `SearchSpace` (from yaml; the digest goes into rig ids);
  - `SensorCatalog` (`cost`, `footprint`, which raises if `dimensions` x/y are missing or the yaml is empty);
  - `Genome` `[radius_idx, colat_idx, n_sensors, model_0..model_{max_total-1}]`;
  - `rig_id` (`rig_` + 10 hex of the SHA-1 of the canonical genome and the search-space digest);
  - `decode` (runs the package's Lloyd generator, loaded by path with RigGenerator's arguments; spacing
    feasibility: centers ≥ the mean of the footprint diagonals);
  - `rig_yaml_dict` / `write_rig_yaml` (sensors in Lloyd site order, named `<model>_s<site:02d>`, plus
    genome / derived metadata).
  - `scripts/moo/config/spherical_search_space.yaml` rewritten:
    - `radii_m` [0.10 … 0.30], `colatitudes_deg` [10 … 60];
    - allowed `vl53l8cx`, `max_total` 20;
    - `tolerance: 1.0e-6` (PyYAML read `1e-6` as a string).
  - Tests: `tests/test_rig_codec.py` (9).
- **Step 2:**
  - `scripts/run_sensor_learning.py` renamed to **`scripts/evaluate_rig.py`** (`git mv`): it evaluates one rig in
    one Isaac Sim process.
  - Flags now use hyphens: `--rig-yaml-path`, `--num-envs`, `--disable-fabric`, `--max-capture-poses`,
    `--max-nbv-time-s`.
  - New flags:
    - `--coverage-target <float|none>` (default 0.5; `none` disables only that termination condition);
    - `--reference-coverage` (0.4);
    - `--out-dir` (writes `capture.h5`, `nbv.h5` and `metrics.json` there).
  - `metrics.json` holds raw per-env results for the optimizer: termination row, time / path to reference, total
    path, total and per-class coverage (≥1/≥5), extent, rig summary, config and timing. On an exception it is
    written with `status: failed`, the error, an `out_of_memory` flag and the traceback.
  - `scripts/sim_utils/sim_worker.py` (empty scaffold) removed; `evaluate_rig.py` is the subprocess entry point.
- `generate_rig.RigGenerator.generate_eef_layout` keeps the rig yaml's order for optical sensors (it grouped
  cameras before ToFs), so sensor i sits on layout site i. It raises if the layout site count ≠ sensor count.
- `nbv/termination.py`: `coverage_target` may be `None`.
- New `nbv/reference_coverage.py`: `ReferenceCoverageTracker`, which records per env the first time and path to
  the reference coverage, and the path length accumulated while active.
- Loggers: `CaptureDataLogger(file_path=...)` and `EvaluationDataLogger(file_path=...)`.
  `save_termination` stores `None` config values as `"none"` (a sim run crashed on this; caught by the new
  failure path).
- `nbv_run_plots`: finds `capture.h5` next to an `nbv.h5`; flag `--path_dt` renamed to `--path-dt`.
- Verified in sim (200 poses, 10 s, `--coverage-target none --out-dir`): both envs ran to `TIME_BUDGET`;
  `metrics.json` is complete; the plot tool renders the eval directory.
- New `scripts/RUNNING.md` (Claude's run notes; `RUN_HPC.md` is the user's and untouched).
- Tests: also `test_reference_coverage.py`, termination with the target disabled, logger `None` attrs
  (76 total).

## 2026-10-28: Tree marker in the EEF path plot; backward-compatibility code removed

- `nbv_run_plots.eef_path_figure`: each env's tree is marked at its base with a vertical line along the trunk axis
  (brown), toggled with its env in the dropdown. Its position is `scene_frames/env_origins + trees/env_i.position −
  scene_frames/robot_base_pose_w[:3]`, the EEF path's frame. `NBVRun.tree_positions_b` added.
- Backward compatibility with older runs removed (decision: no data to preserve while developing):
  - `nbv/nbv.py`: the `albedo` → lit `rgb` fallback and its warning (`_stored_uv_color_key`) are gone, for both
    stored and live data. `_UV_COLOR_KEY = "albedo"` is required. `pose_w_convention` is required (no default
    to `world`).
  - `nbv_run_plots.py`: `termination`, `scene_frames` and the `instance_extent` attrs are required. Files without
    them raise `KeyError`; runs before `SL-PDC_20261028_132532` no longer plot.
  - Tests: the legacy-rgb test is removed. The convention test uses explicit `"world"` instead of a missing attr.
    The plot fixture writes termination, coverage config, scene frames and the tree position (64 total).

## 2026-10-28: Log env origins and robot base pose

- `CaptureDataLogger.save_scene_frames()` and `EvaluationDataLogger.save_scene_frames()` write a
  `scene_frames` group:
  - `env_origins` (n_envs, 3), the world origin of each env (`scene.env_origins`);
  - `robot_base_pose_w` (n_envs, 7), the robot root pose in world, quaternion xyzw (attr `quaternion_order`).
  - Re-saving overwrites.
- `scripts/run_sensor_learning.py` saves them to the capture file after the tree metadata, and to the evaluation
  file at NBV setup.
- With these, logged poses can be related: tree world pose = env origin + `trees/env_i` (position, orientation);
  EEF world position = evaluation `pose` position + base position. Only new runs have them.
- Tests: new `tests/test_loggers.py` (both loggers; 64 total).

## 2026-10-28: 3D EEF path plot

- `scripts/data_utils/plot/nbv_run_plots.py`:
  - New `eef_path_figure(run, dt=0.05)`, added as the third figure in the report. It shows the EEF path in the
    robot base frame, colored by NBV time (Viridis, one shared scale across envs), with:
    - goal markers colored by assignment time;
    - start (diamond) and stop (x) markers, the stop labelled with the termination reason.
  - Each path is cut at its env's stop time and downsampled to one point per `dt` (`--path_dt`, default 0.05 s).
  - A dropdown selects one env or "all envs" (all share the base frame); it opens on env_0.
  - `NBVRun.goal_poses` added (goal poses from `nbv_goal`).
- The 3D *bin* plot stays paused until the tree UV bugs are resolved upstream.
- Tests: `test_eef_path_is_cut_at_stop_time_and_downsampled` (62 total).

## 2026-10-28: Plot module is runnable

- The CLI (`resolve_eval_path`, `main`) moved into `scripts/data_utils/plot/nbv_run_plots.py`, which now has an
  `if __name__ == "__main__"` guard. `scripts/plot_nbv_run.py` is a thin wrapper around it. Both
  `python scripts/plot_nbv_run.py` and `python scripts/data_utils/plot/nbv_run_plots.py` work.
- `--show` warns if no browser could be opened.
- Line styling: per-env lines unchanged (opacity 0.75, width 1.5); the mean is semi-transparent (opacity 0.5,
  width 3); stop-time lines are opaque. The env palette is now `Dark24` (24 colors), so ~20 envs don't repeat.
  These are module constants `ENV_LINE`, `MEAN_LINE`, `STOP_LINE` and `ENV_COLORS`.

## 2026-10-28: Post-run plots, run timer, `--quiet`

- New `nbv/coverage_metrics.py`, depending only on torch and dataclasses.
  - It holds `INSTANCE_CLASSES`, `instance_class`, `CoverageConfig`, `class_ids_from_prim_map`,
    `per_class_values`, `instance_coverage`, `instance_extent_per_instance` and `instance_extent`, all operating on
    a counts tensor plus class ids.
  - `NBVEvaluator` delegates to it, so behaviour is unchanged. `nbv.py` re-exports `INSTANCE_CLASSES`,
    `CoverageConfig` and `instance_class`.
  - Code outside Isaac Sim (plots, later the optimizer) loads the file by path, because the package `__init__`
    needs Kit.
- New `scripts/plot_nbv_run.py`: `--eval <path | datetime id>` (default: newest), plus `--capture`, `--dt`,
  `--out` and `--show`. It writes one standalone HTML report to `data/plots/<eval name>.html`.
- New `scripts/data_utils/plot/nbv_run_plots.py`:
  - `load_run` reads the evaluation h5 plus the instance → class map from the matching capture h5.
  - `coverage_over_time` replays `sparse_bin_updates` every `dt` seconds through `coverage_metrics`.
  - `movement_over_time` gives cumulative path length, rotation and goals assigned.
  - Two figures:
    - a coverage grid: rows are instances seen, length, theta and continuous; columns are trunk, branch, spur and
      total;
    - a movement row.
  - Both show per-env lines (thin), the mean over envs (thick black), solid / dashed for ≥1 / ≥5 hits, and dotted
    lines at each env's stop time, labelled with its termination reason.
  - The coverage config is read back from the file's `instance_extent` attrs.
  - Verified on `SL-PDC_20261028_114336`: the replayed final values match the saved `termination.final_coverage`,
    `instance_coverage` and `instance_extent` exactly.
- `scripts/run_sensor_learning.py`:
  - Wall-clock timer: setup / capture (and s/pose) / NBV (and real-time factor) / total, printed as `[TIMER]`.
    It is saved to the evaluation h5 `timing` attrs (with `nbv_sim_time_s`, `n_capture_poses`, `quiet`).
  - `--quiet` skips the per-step bin / instance / extent printouts and their computation. The termination
    coverage is still computed.
- `scripts/data_utils/evaluation_data_logger.py`: `save_run_timing()`.
- `.claude/launch.json`: a `nbv-plots` static server (`python3 -m http.server 8765 --directory data/plots`), used
  to preview reports in the desktop app's browser pane.
- Tests: new `tests/test_nbv_run_plots.py`, covering the standalone import with no Omniverse modules, the replay
  over time with out-of-order updates, and movement (61 total).

## 2026-10-28: Per-env termination

- New `nbv/termination.py`:
  - `TerminationReason` (IntEnum): `RUNNING`, `COVERAGE_TARGET`, `TIME_BUDGET`, `GOAL_BUDGET`,
    `PREDICTED_GAIN_FLOOR`, `PLATEAU`, `CANDIDATES_EXHAUSTED`, `UNREACHABLE_GOALS`.
  - `TerminationConfig` with the agreed values:
    - coverage target: total instances with ≥5 hits ≥ 0.5;
    - 180 s of NBV sim time; 300 goals;
    - predicted-gain floor at 1% of the first goal's best score;
    - plateau: < 0.005 gain over 20 goals;
    - 1.0 m maximum search radius;
    - 5 s goal timeout, ending the env after 3 consecutive or 10 total timeouts.
  - `TerminationTracker`: per-env state. Events: `update` (coverage target, time budget), `on_goal_assigned`
    (gain floor, goal budget), `on_goal_reached` (plateau), `goal_timed_out` / `on_goal_timeout`. Also `summary()`.
    The first reason is kept.
- `nbv/nbv.py`:
  - `get_candidate_pose_idxs(curr_poses, env_ids, visited, max_radius)`: a per-env candidate search that excludes
    visited and skipped poses and grows the radius per env. It returns the exhausted envs instead of raising.
    It fixes `get_closest_pose_idxs` growing the radius until *any* env had a pose, which led to
    `ValueError("no unseen_closes_idxs")`.
  - `compute_env_info_gain_from_pose_idxs(..., return_scores=True)` also returns the best predicted score per env.
- `scripts/run_sensor_learning.py`:
  - One `assign_goals()` for the initial and later goals. This fixes the initial-goal double indexing; the old
    `seen_pose_idxs` / `current_poses` bookkeeping is replaced by a `visited` (n_envs, n_poses) mask.
  - Finished envs hold their current EEF pose (`hold_pose`) and are excluded from binning. Goal-reached checks
    run only for active envs. The run ends once every env has stopped.
  - The initial goals are logged at the current sim time (they were logged at t = 0, which showed up as a ~35 s gap).
  - `--max_nbv_steps` is replaced by `--max_nbv_time_s`, which overrides `TerminationConfig.max_sim_time_s`.
  - Prints each env's stop reason at the end.
- `scripts/data_utils/evaluation_data_logger.py`: `save_termination()` writes a `termination` table, one dataset
  per column (`env_id, reason, reason_name, t_end, step_end, n_goals, n_goals_reached, n_timeouts,
  final_coverage`). Attrs hold `reason_<code>` names and `config/<field>` values. Times are NBV-relative.
- Tests: new `tests/test_termination.py`, covering every reason, per-env independence, the candidate search and
  score return (58 total).
- **Bug fix, capture file naming race:**
  - `CaptureDataLogger` named its file from its own `datetime.now()`, while the script looked the file up by
    `trial_metadata["datetime_id"]`, created moments earlier. When the two fell in different seconds, the NBV
    phase raised `FileNotFoundError` (seen 2026-10-28: SL-PDC_..._113504 vs ..._113505).
  - `CaptureDataLogger(datetime_id=...)` now takes the trial's id, and the script reads the path from
    `cdlog.datafile_path`.
- Verified in sim (200 capture poses, `--max_nbv_time_s 20`): both envs stopped with `TIME_BUDGET` at 20 s
  (26/27 and 31/32 goals, coverage 0.092 / 0.089). The `termination` table, instance coverage, extent and bin
  counts are all saved. With 40 capture poses, both envs stopped at t = 0 with `CANDIDATES_EXHAUSTED`, which is
  correct: the first 40 grid poses are all > 1 m from the start pose. On the full grid the nearest is 0.23 m away.

## 2026-10-27: Per-class gap defaults

- `CoverageConfig.continuity_max_gap_bins` now defaults to `{"trunk": 2, "branch": 2, "spur": 4}`, chosen as
  physical break sizes:
  - spur ~10 cm → 0.5 cm/bin, a 2 cm break → 4 bins;
  - branch ~50 cm → 2.5 cm/bin, a 5 cm break → 2 bins.
  - L stays 0.5 for all classes.
- On `SL-PDC_20261026_213705` (env_0), spur continuity rises from 0.155 to 0.221 (k = 1) and from 0.013 to 0.023
  (k = 5); branch and trunk are unchanged.
- Tests: the definition-example test now uses a 5-bin spur gap; added boundary tests for each class default
  (47 total).

## 2026-10-27: Instance extent: length/theta coverage and continuity

- `nbv/nbv.py`:
  - `CoverageConfig` gains `continuity_min_length_frac` ("L", 0.5) and `continuity_max_gap_bins` ("g", 2).
  - `NBVEvaluator.instance_extent_per_instance(env, k)` returns, per instance at hit threshold k:
    - `length_coverage`: fraction of the 20 length bins with ≥ k hits summed over theta.
    - `theta_coverage`: fraction of the 36 theta bins with ≥ k hits summed over length.
    - `longest_gap`: the longest run of consecutive unhit length bins, including runs at either end (= 20 if
      unseen).
    - `continuous`: `length_coverage ≥ L` and `longest_gap ≤ g`.
  - `NBVEvaluator.instance_extent(min_hits=None)` returns per class plus total, at each k: those values averaged
    over **all** instances (unseen ones count as 0); `continuous` becomes a fraction.
- `scripts/data_utils/evaluation_data_logger.py`: `save_instance_extent()` writes:
  - `instance_extent/<env>/<class>/ge_<k>` attrs (class means);
  - `instance_extent/<env>/per_instance/ge_<k>/{length_coverage, theta_coverage, longest_gap, continuous}`
    datasets, plus `per_instance/class_id`;
  - the config and class names as root attrs.
- `scripts/run_sensor_learning.py`: prints `extent (len, theta, continuous)` per env each step, and saves the
  extent with the final bin counts.
- On `SL-PDC_20261026_213705` (env_0, k = 1 / k = 5), as (len, theta, continuous):
  - trunk: (0.95, 0.89, 1.0) / (0.95, 0.86, 1.0)
  - branch: (0.13, 0.07, 0.03) / (0.02, 0.01, 0.01)
  - spur: (0.26, 0.13, 0.16) / (0.04, 0.02, 0.01)
- Tests: 3 extent tests in `tests/test_instance_coverage.py` (40 total).
- Per-class thresholds: `continuity_min_length_frac` (L) and `continuity_max_gap_bins` (g) accept a scalar (all
  classes) or a dict keyed by class, with an optional `"default"` key. Missing classes and unknown instances fall
  back to 0.5 / 2. Defaults are unchanged. The logger flattens dicts to `<name>/<class>` attrs. One more test (41
  total).

## 2026-10-26: Instance-level coverage (plan Q-J)

- `nbv/nbv.py`:
  - `INSTANCE_CLASSES = ("trunk", "branch", "spur")` and `instance_class(prim_path)`: the class the leaf prim name
    starts with (longest match). This is the same rule as `scripts/moo/instance_parser.parse_instance`; a test
    keeps them in sync, because the optimizer can't import the package outside Isaac Sim.
  - NBV semantic weights use it too. They previously matched substrings of the full path; the result is the same
    on current prim names.
  - `CoverageConfig(instance_min_hits=(1, 5))`, passed to `NBVEvaluator(coverage_config=...)`.
  - `NBVEvaluator.instance_class_ids[env]`: per custom id, an index into `INSTANCE_CLASSES` (-1 = unknown).
  - `NBVEvaluator.instance_coverage(min_hits=None)` returns
    `{env: {trunk|branch|spur|total: {"n_instances", "seen": {k: fraction}}}}`. An instance is seen at k once it
    has ≥ k hits summed over all its bins.
- `scripts/data_utils/evaluation_data_logger.py`: `save_instance_coverage()` writes
  `instance_coverage/<env>/<class>` attrs `n_instances` and `seen_ge_<k>`.
- `scripts/run_sensor_learning.py`:
  - prints instance coverage per env each step, next to the existing bin print;
  - saves it to the evaluation h5 with the final bin counts.
- Check on `SL-PDC_20261026_174422` (final counts): env_0 is trunk 1.0/1.0, branch 0.42/0.16, spur 0.62/0.34,
  total 0.56/0.28 (≥1 / ≥5 hits).
- Tests: new `tests/test_instance_coverage.py`, covering classifier parity with `instance_parser` and per-class
  and total fractions (37 tests total).

## 2026-10-26: NBV step budget

- **Bug fix, `scripts/run_sensor_learning.py`:** `--max_nbv_steps` was compared with `common_step_counter`, which
  keeps counting from the capture pass (4200 steps ≈ 35 s). The NBV in `SL-PDC_20261026_174422` therefore ran
  48 s instead of 83 s. The budget now counts from `nbv_start_step`. Logged timestamps are unchanged (still
  absolute sim time).

## 2026-10-26: Simulation bug fixes (coverage measurement and NBV motion)

- **Bug fix, NBV never moved the robot (`scripts/run_sensor_learning.py`).**
  - After capture the script set `robot.sim_state = "evaluate"`, an attribute on the `Articulation` that
    nothing reads. The env's own `sim_state` stayed `"capture"`, where `_apply_action` is a no-op, so the DiffIK
    controller never ran.
  - This explains the debug run that got one goal per env and stalled ~0.4 m away.
  - Now calls `env.unwrapped.update_sim_state("evaluate")`.
- **Bug fix, UV bins came from the lit RGB render.**
  - `sensor/generate_rig.py`: cameras request `"albedo"` (unlit `DiffuseAlbedoSD`, RGBA uint8; Isaac Sim ≥ 6.0)
    instead of `"rgb"`. Without any color output requested, the renderer also skips color rendering.
  - `scripts/data_utils/capture_data_logger.py`: stores `sensors/<name>/albedo` (uint8, RGB channels) instead of
    `rgb` (float32). That is 4× smaller per pixel.
  - `nbv/nbv.py`:
    - `BinnedTree.rgb_to_bin` (exact-match 256³ LUT, run on the CPU) is replaced by `uv_color_to_bin`, an
      arithmetic decode on the tensor's device: `theta = (R + 0.5)·36/256`, `length = (G + 0.5)·20/256`.
    - It agrees with the old LUT on 82% of colors and is off by one bin elsewhere. Those are colors on region
      edges, where the LUT's choice was arbitrary (last write wins).
    - Colors are always treated as 0–255. The old "max ≤ 1.01 ⇒ multiply by 255" guess could misread dark
      patches.
    - `cv2`/`uv.png` are no longer loaded at runtime.
  - Stored captures read `albedo`, or fall back to `rgb` with a warning (legacy files bin incorrectly).
- **Bug fix, live and predicted binning used different pixel masks.**
  - New module-level `valid_pixel_mask` (depth in `[z_near, z_far)`, incidence ≤ `max_incidence_angle` about
    the convention's optical axis) is used by both paths.
  - Live binning previously only checked `isfinite(depth) & depth > 0`.
  - Live uses `quat_w_ros` and the depth limits from the capture file's sensor attrs.
- **Bug fix, each 15 Hz frame was binned ~8× in the NBV loop.**
  - `update_bin_count_scores_from_observation(..., frames=)` takes each sensor's `Camera.frame` (per-env render
    counter) and skips sensor/env pairs whose counter didn't advance.
  - `run_sensor_learning.py` passes `{name: sensor.frame.torch}`. `frames=None` keeps the old behaviour;
    `scripts/run_nbv.py` still uses it.
- **Capture: forced renders made explicit.**
  - `scene.update(decimation)` (which passes dt = 1 s and worked only because `decimation == 1`) is replaced by
    `scene.update(capture_dt)`, where `capture_dt` is the longest sensor `update_period`.
  - The capture loop now raises if any sensor's frame counter didn't advance for a pose, i.e. it would store a
    stale image.
  - The NBV setup step uses `scene.update(physics_dt)`.
- `run_sensor_learning.py`:
  - `--max_capture_poses N` (debug; the capture file is sized to N poses);
  - `--max_nbv_steps` (default 10,000, previously hard-coded).
- Removed `_prepare_sensor_batch` (its dict-input branch was unused), `_slice_sensor_batch` and
  `_build_sensor_data_for_env`.
- Tests (30 total):
  - `test_binned_tree.py` rewritten for the arithmetic decode, including a full-range regression test.
  - New `test_live_binning.py`: live vs predicted agreement, frame-counter skipping.
  - Legacy-rgb fallback test.
  - `conftest.render_pixels` helper; capture fixtures write `albedo` by default.

## 2026-10-26: Camera optical axis, quaternion audit, tests

- **Bug fix, `nbv/nbv.py`, `_load_masked_sensor_batch`.**
  - The incidence mask took the z axis of the stored sensor quaternion. Captures stored `quat_w_world`, whose
    forward axis is +X, so normals were compared against the camera's *up* vector. On
    `SL-PDC_20261026_134557.h5` it kept ~9% of tree pixels; it now keeps ~53%.
  - The coverage ceiling over all poses went from 4.8% to 8.9% (bins ≥ 1 hit).
  - The optical axis is now chosen from the sensor group attr `pose_w_convention` via `_OPTICAL_AXIS`:
    `ros` → +Z, `opengl` → −Z, `world` → +X. Files without the attr are read as `world` (all existing captures).
- `scripts/data_utils/capture_data_logger.py`: stores `quat_w_ros` in `sensors/<name>/pose_w` (optical axis
  +Z, matching `OffsetCfg.convention = "ros"` in `generate_rig`), and writes `pose_w_convention = "ros"` on each
  sensor group.
- Quaternion audit: everything is (x, y, z, w). Two w-first leftovers fixed:
  - `utils/usd_utils.py`, `load_usd`: no longer reorders to (w, x, y, z) before `sim_utils.create_prim`, which
    takes (x, y, z, w) in Isaac Lab 3. It currently has no callers.
    - `create_prim` itself converts to USD's (w, x, y, z) `Gf.Quatd`, so the old manual swap would have
      reordered twice and authored a wrong `xformOp:orient`.
    - USDA files showing (w, x, y, z) are expected: that is USD's native order, not this project's.
  - `controllers/goal_ik_solver.py`: `math_utils.mat_from_quat` (doesn't exist in Isaac Lab 3) →
    `matrix_from_quat`. Also unused right now.
  - `utils/quaternion_utils.py` still has `xyzw_to_wxyz` / `wxyz_to_xyzw` helpers. They are only referenced
    from commented-out code; left in place for talking to w-first libraries.
- New `tests/` (24 tests, run with `pytest` from the project root; Omniverse UI modules are stubbed, no Isaac
  Sim needed). `pyproject.toml`: `testpaths = ["tests"]`. pytest was installed into the IsaacLab venv.
  - `conftest.py`: writes synthetic capture h5 files with the `CaptureDataLogger` layout.
  - `test_binned_tree.py`: uv.png encoding, all 720 bins reachable, bin-center colors round-trip.
  - `test_nbv_evaluator.py`:
    - pixel validity (depth range, incidence limit, unlabelled ids);
    - the optical axis for each convention (regression test for the bug above);
    - batching/cache invariance, and the cache reading each pose once;
    - semantic weights;
    - the closed-form score;
    - best-pose selection, saturation, ties, env filtering;
    - search-radius expansion.
  - `test_h5py_utils.py`: `attrs_only`.

## 2026-10-26: Rig yaml location (plan Q-D)

Rigs change often, so they now live only in `<repo>/data/rigs/`, not in the package config.

- Removed `source/isaaclab_sensor_learning/isaaclab_sensor_learning/config/rigs/test_rig0.yaml` and the now-empty
  `config/rigs/` dir. It was a stale copy: seven duplicate `tof1` names and no `layout` block.
  `data/rigs/test_rig0.yaml` is a strict superset of it.
- `isaaclab_sensor_learning/__init__.py`: added `RIG_DIR = DATA_DIR/rigs`.
- `tasks/direct/isaaclab_sensor_learning/common_env_cfg_params.py`: added `default_rig_yaml_path`, pointing to
  `data/rigs/test_rig0.yaml`. It is resolved from the file location, not from the package, to avoid a circular
  import (the package `__init__` imports `tasks` before defining its `*_DIR` constants).
- `evaluation_env_cfg.py`, `pose_data_capture_env_cfg.py`: `rig_yaml_path` defaults to
  `env_params.default_rig_yaml_path` (previously `config/rigs/test_rig0.yaml`).
- `scripts/run_sensor_learning.py`: the `--rig_yaml_path` default is now relative to the script
  (`scripts/../data/rigs/test_rig0.yaml`) instead of a hard-coded `/home/luke/...` path, so it also works on HPC.
- `sensor/yaml_to_cfg.py`: `RIG_CFG_DIR` now points to `RIG_DIR`. Fixed its `main()`, which did `str / str`
  and named a non-existent `rig0.yaml`.
- `sensor/generate_rig.py`: `main()` uses `RIG_DIR` (it also did `str / str`). Added `import os`.
  The rest of this `main()` is still stale (it calls `RigGenerator(sensor_cfgs=...)`); not touched.

## 2026-10-26: NBV evaluator reads candidate poses from disk (VRAM fix)

- `nbv/nbv.py`, `NBVEvaluator`:
  - Takes `trial_data_path` instead of a fully loaded `trial_data` dict. Only metadata (poses, trees, sensor
    attrs) is loaded at init. The h5 file stays open; call `close()` or use the evaluator as a context manager.
  - Candidate poses are read per env, `ObservationConfig.candidate_batch_size` (default 16) at a time, and
    masked by depth and incidence angle exactly as before.
  - Each candidate's sparse bin delta is cached (`ObservationConfig.cache_pose_deltas`, default on).
    Captured data is static, so only the score is recomputed against live `bin_counts`.
  - `compute_env_info_gain_from_pose_idxs(env_ids, closest_idxs)`: dropped the `trial_data_dict` argument.
  - `score_pose_delta` takes a coalesced sparse delta.
  - Removed `_build_masked_dict`, `_select_best_pose_for_env`, `_h5_to_device` and `sensor_tensors`.
  - The live-observation update path is unchanged.
  - Verified against the previous implementation on a 300-pose subset with random live counts: identical
    best poses and bit-identical scores, with the cache on and off.
- `utils/h5py_utils.py`: `h5_to_dict(node, attrs_only=(...))` skips datasets under the named groups. New
  `h5_attrs_to_dict` helper.
- `scripts/run_sensor_learning.py`, `scripts/run_nbv.py`: updated to the new evaluator API. `run_nbv.py` loads
  trial metadata with `attrs_only=("sensors",)`.
- `scripts/run_sensor_learning.py`: now calls `cdlog.save_labels_dict()` after the capture loop. Without it,
  `trees/env_*/instance_labels` was never written, so every stored candidate scored 0.
