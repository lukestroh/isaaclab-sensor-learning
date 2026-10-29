# Sensor-config NSGA-II: simulation evaluation plan

## Goal

Each NSGA-II individual is one sensor configuration (a rig). Evaluating it means running one simulation and
returning three objectives:

| # | Objective | Direction | Source |
|---|-----------|-----------|--------|
| 1 | Tree coverage, **per class (trunk / branch / spur) and total** | maximize | final `bin_counts` from live observations binned along the NBV trajectory |
| 2 | EEF movement (proxy for time) | minimize | EEF pose trajectory logged during NBV (`path_evaluation.eef_motion_metrics`) |
| 3 | Sensor cost | minimize | sum of `cost` (USD) from each sensor's model yaml; no simulation needed |

## Evaluation pipeline (per individual)

1. **Build rig** from the rig yaml passed as `--rig_yaml_path` (written by the NSGA-II decoder; see Phase 4),
   then spawn the env.
2. **Capture pass:** teleport through the precomputed joint states (`joint_states.h5`) and record what every
   sensor sees at every pose.
3. **NBV pass (greedy):**
   - Start at the default pose.
   - Find candidate poses within `pose_search_radius`, skipping ones already visited.
   - Score each candidate with the captured data against the current live `bin_counts`.
   - Drive to the best candidate with the differential IK controller, binning live observations at every step.
   - Repeat until the termination condition is met.
4. **Return** `SimOutputs(poses, t_final, bin_counts, instance_map)` to the optimizer. The sim reports
   coverage per class and in total; it does not aggregate across classes or envs. That choice belongs to the
   NSGA-II side (see Q-C).

The captured data is used only to *predict* a candidate's gain. Actual coverage always comes from live
observations along the controller's trajectory.

## Instance labels (audited 2026-10-26 on `SL-PDC_20261026_134557.h5`)

- **Custom map** (`trees/env_i` attrs, from `trial_setup.get_tree_metadata`) maps every non-mesh tree prim to a
  dense id. It sets the `bin_counts` rows, so prims that are never seen still count toward the coverage total.
  - env_0: 2233 prims. env_1: 1822 prims.
- **Render labels** (`trees/env_i/instance_labels`) map render ids to prim paths. They are merged from each
  frame's `idToLabels`.
  - env_0: 2199 render labels. env_1: 1795.
  - **All of them resolve to a custom-map prim (0 misses).**
- Custom prims never seen from any captured pose: 34 (env_0) and 27 (env_1). These are expected: they are
  occluded or out of view, and are the reason the custom map is needed.
- Render ids in the images without a label: 2 per env. These are probably BACKGROUND/UNLABELLED (ids 0/1);
  confirm.
- **Risk:** the merged map assumes a render id refers to the same prim in every frame. Replicator only
  guarantees `idToLabels` per frame. Phase 1 removes this assumption by mapping each frame's render ids to
  custom ids at capture time.

## Conventions
- **Quaternions are (x, y, z, w) everywhere**, as in Isaac Lab 3 (`math_utils`, `create_prim`, `OffsetCfg.rot`)
  and scipy's `as_quat`. Audited 2026-10-26; the two w-first leftovers were fixed (see changelog).
  - **The one exception is USD itself.** `Gf.Quatd`/`Gf.Quatf` and `xformOp:orient` (including `.usda` text)
    are (w, x, y, z). Isaac Lab 3 swaps at that boundary internally (`isaaclab/sim/utils/transforms.py`: it
    builds `Gf.Quatd(q[3], q[0], q[1], q[2])` on write and returns (x, y, z, w) on read). So a USDA file
    showing w-first is expected and correct.
  - Rule: code in this project only reorders when it calls the `pxr` API directly (`Gf.Quat*`,
    `UsdGeom.XformOp`). It never reorders when calling Isaac Lab functions. As of 2026-10-26 the project
    makes no direct `Gf.Quat*` / `xformOp:orient` calls.
- **Camera optical axis is +Z (ROS convention).**
  - `generate_rig` builds offsets with `OffsetCfg.convention = "ros"`, and the spherical layout aligns each
    sensor's +Z with the outward surface normal.
  - `OffsetCfg.convention` only sets how the *offset* is interpreted. `CameraData` always reports the same
    orientation three ways: `quat_w_world` (+X forward), `quat_w_ros` (+Z), `quat_w_opengl` (−Z).
  - The capture logger stores `quat_w_ros` and tags each sensor group with `pose_w_convention = "ros"`; the
    evaluator requires the tag.
- **No backward compatibility while developing (2026-10-28).** Code reads only the current data format;
  re-capture after format changes instead of adding fallbacks.

## Coverage ceiling (measured 2026-10-26 on `SL-PDC_20261026_134557.h5`)

**How the ceiling is computed.**
- For every one of the 4200 captured poses, take the bin hits all 8 sensors would add there. This is exactly
  what NBV scoring predicts for that pose.
- Sum them all: the robot visits *every* pose once. No NBV ordering can do better with this rig, pose grid and
  metric, because NBV only ever visits a subset of these poses.
- (Live binning along the path between poses adds some views, but those come from the same region of space.)
- The scratch script is `scratchpad/ceiling.py`; it is to become `scripts/tools/coverage_ceiling.py`.

**Results.** "Before" uses the original code; "after axis fix" uses the corrected incidence mask (env_0; env_1
is within ~1 point).

| metric | before | after axis fix |
|---|---|---|
| bins ≥1 hit, all / trunk / branch / spur | 4.8 / 30.7 / 3.3 / 5.5% | 8.9 / 36.7 / 6.9 / 9.9% |
| bins ≥5 hits, all | 1.3% | 3.9% |
| instances with ≥5 hits, all | 68.8% | 85.2% |
| length bins hit (any theta), all | 30.3% | 41.2% |
| median bins hit per pose (8 sensors) | 142 | ~670 |

**What limits it**, in order of how much each one costs:
1. **Incidence mask used the wrong camera axis.** *Fixed, no re-capture needed.*
   - It took z of `quat_w_world`, which is the camera's *up* axis; that convention's forward is +X.
   - It kept ~9% of tree pixels, effectively at random. It now keeps ~53%.
   - Going forward, poses are stored in ROS convention and +Z is used (see Conventions).
2. **UV colors come from the lit/tonemapped RGB render.** *Needs a code fix and a re-capture.*
   - `uv.png` encodes theta in R and length in G (0–255), with B = 0.
   - Rendered tree pixels reach only R, G ≲ 154, with B up to 109.
   - So length bins ≥ 12/20 and theta bins ≥ 22/36 are unreachable; only 12 × 22 = 264 of 720 bins (37%) can
     be hit. Every other pixel lands in a bin that is too low.
   - After the axis fix the trunk's ≥1 coverage is exactly 36.7% = 264/720: it saturates the whole reachable
     region. **This is now the binding limit for large instances.**
   - Fix: bin from the camera's `"albedo"` output (unlit) and decode bins arithmetically from R and G.
   - Expected effect: the addressable grid grows about 2.7×. Small instances gain less (see 3).
3. **Bins are far smaller than a ToF pixel on small parts.** *Metric design (Q-J); the UV fix does not address
   this.*
   - Every instance gets 20 × 36 bins. A 60×60 ToF with 65° FOV has a ~1.5 cm pixel footprint at 1 m.
   - A ~10 cm × 1 cm spur has bins of about 5 mm × 0.9 mm, so each view lands only a handful of rays on it.
     Filling its 720 bins, five times each, would take thousands of distinct rays.
   - This is why branch and spur bin coverage stays ~7–10% even over all poses, while ~85–90% of those
     instances *are* seen.
4. **One-sided view (Q-K).** The pose grid is a cylindrical shell on one side of the row, so at most ~half of
   each instance's theta bins can ever face a sensor. The 45° incidence limit narrows this further.

Debug-run "1%": that log (`SL-PDEVAL_20261026_134557.h5`) shows each env received **one** NBV goal and never
reached it. It stalled ~0.4 m and ~70–90° from the goal for the remaining ~48 s. So the run measured a single
viewpoint, with each 15 Hz frame binned ~8× (≥1-hit and ≥5-hit coverage are identical: 0.39%). See Phase 2.

## Coverage on the 8×8 VL53L8CX (measured 2026-10-26, `SL-PDC_20261026_174422`, albedo, 4200 poses)

The VL53L8CX yaml is now its real 8×8 zone resolution (it was 60×60), so each frame carries 64 rays per sensor
instead of 3600. That is about 56× fewer, and it accounts for the drop from the earlier 60×60 numbers. The share of
tree pixels passing the masks is unchanged (~52%).

| env_0 | bins ≥1 | bins ≥5 | hits |
|---|---|---|---|
| all 4200 grid poses | 2.44% | 0.20% | 112,812 |
| the 59 visited goal poses only | 0.16% | 0.00% | 3,301 |
| live (goals + path), 48 s | 0.92% | 0.07% | 31,678 |

- **The grid sum is not a ceiling for live coverage.** It bounds what NBV *predicts* and what the goal views
  alone see. Live binning along the DiffIK path is ~6× the goal views, and 58% of the bins it hits are never hit
  from any grid pose.
- Live is still below the grid sum because 59 goals are ~1.4% of the grid. 48 s × 15 Hz ≈ 720 frames per sensor
  (against 4200 grid frames), and consecutive frames overlap heavily.
- Implication for Q-J: a zone covers ~11 cm at 1 m, so on an 8×8 ToF the per-instance 20×36 UV bins (sub-mm on
  spurs) are mostly unreachable. The simulation also samples one pinhole ray per zone, while the real sensor
  integrates over the zone. The metric (and possibly the zone model) has to match this sensor class.

## Phases

### Phase 0: data correctness
- [x] Incidence mask uses the optical axis of the stored pose convention (was the camera's up axis).
- [x] Capture stores `quat_w_ros` and tags `pose_w_convention`; quaternion audit (all (x, y, z, w)).
- [x] `tests/` with unit tests for UV binning, pixel masking, candidate scoring/selection, and h5 helpers
      (`pytest` from the project root; no Isaac Sim needed).
- [x] **Bin from `"albedo"`, not `"rgb"`, and decode bins arithmetically** (see "Coverage ceiling" above).
      Existing captures must be re-captured to benefit; legacy `rgb` files still load, with a warning.
- [ ] Re-measure the coverage ceiling after the albedo fix (`scripts/tools/coverage_ceiling.py`).
- [x] One shared validity mask for live and predicted binning (`valid_pixel_mask`). Today live binning
      (`_prepare_sensor_batch`) only checks `isfinite(depth) & depth > 0`: no `z_near`/`z_far`, no incidence
      limit. So the coverage a goal actually yields can't be compared with the gain predicted for it.
- [x] Call `cdlog.save_labels_dict()` after the capture loop (render labels were missing from every earlier file).
- [x] Re-run `run_joint_state_collection.py` with the fixed-base robot USD, then regenerate `joint_states.h5`
      (done by the user, 2026-10-28).
- [ ] Remove `skipped_poses` plumbing (no longer produced). Keep the evaluator tolerant if the dataset is absent.
- [ ] During capture, assert that no render id maps to two different prim paths across frames (cheap check;
      it tells us whether the stored merged map can be trusted for older files).

### Phase 1: capture stores per-pose bin deltas instead of raw images
- At capture time, for each (env, pose):
  - Apply the depth and incidence-angle mask.
  - Convert rgb to (length_bin, theta_bin) and the render id to the custom id, using **that frame's** `idToLabels`.
  - Sum over sensors and store the sparse triplets with their counts.
- Storage is ragged: `deltas/env_i/{instance_idx, length_bin, theta_bin, count}` plus a `pose_offsets` array,
  with the mask/bin settings stored as attrs. This should be MBs instead of GBs.
- `NBVEvaluator` loads all deltas into memory at init. This removes the h5 lazy-load path and the
  `render_id -> custom id` LUT on the scoring side. Scoring is unchanged (`score_pose_delta` already takes a
  sparse delta).
- Raw images become an optional debug dump (`--save_raw`), off by default for NSGA runs (decision Q-B).
  - The dump goes in a separate dir, never next to the summaries:
    `data/capture_raw/<run_id>/<rig_id>/SL-PDC_<datetime_id>_raw.h5`.
  - The summary file stays at `data/capture/<run_id>/<rig_id>/SL-PDC_<datetime_id>.h5`.
  - Both files carry the same `datetime_id`, `rig_id` and `run_id` attrs. The summary also stores
    `raw_data_path` (empty if no dump), so either file can find the other.
  - The raw dump keeps the current layout: `sensors/<name>/{rgb, depth, normals, instance_segmentation,
    pose_w}`, plus the per-frame `idToLabels` that the summary was built from.
  - Today's lazy-loading `NBVEvaluator` code becomes the offline tool that recomputes deltas from a raw dump.
- Equivalence check: deltas computed at capture must match deltas recomputed from a raw dump of the same run.

### Phase 2: NBV loop: sensor rate, per-env termination, cleanup (`run_sensor_learning.py`)

**Sensor-rate bug (found 2026-10-26).**
- Isaac Lab *does* honour `update_period = 1/data_rate_hz`. A camera only re-renders once its period has
  elapsed (`SensorBase.update` → `update_timestamp_kernel` → `_is_outdated`).
- But `_get_observations` returns `sensor.data` every physics step (120 Hz). Between renders, that is the
  previous frame, unchanged.
- The NBV loop bins every step, so each real 15 Hz frame is binned about 8 times. Counts are inflated, and the
  "≥ 5 hits" threshold can be met by a single frame. This does not explain *low* coverage; it inflates it.
- **Fixed 2026-10-26.** Fix:
  - `Camera.frame` is a per-env counter that only increments on a real render.
  - Keep the last seen `frame` per (sensor, env), and only bin pairs whose frame advanced this step.
  - Pass that mask into `update_bin_count_scores_from_observation`.
  - Each sensor then contributes at its own rate, which is what makes faster sensors worth their cost.
- Capture pass: `scene.update(decimation)` passes `dt=1` (seconds), so every sensor is forced to re-render
  after each teleport. It is correct by accident (only because `decimation == 1`).
  - **Fixed 2026-10-26:** `scene.update(capture_dt)` with `capture_dt` = the longest sensor `update_period`,
    plus a per-pose check that every sensor's `frame` advanced.

**Per-env termination (Q-A).**
- Each env stops independently. Isaac Lab steps all envs together, so a finished env keeps being simulated
  but is *frozen*:
  - its action is held at its current EEF pose;
  - its observations are no longer binned;
  - its path length and pose log stop growing;
  - its `t_final` is recorded.
- The sim exits when every env is done. Objectives are averaged over envs afterwards.
- Conditions, any of which ends an env (reason logged per env):
  1. **Coverage target reached:** e.g. total coverage ≥ X%, possibly per-class targets later.
     - X must sit below the achievable ceiling (see "Coverage ceiling").
     - The "50% of the tree ⇒ whole tree seen" hypothesis can't be checked until the albedo fix lands.
     - Suggestion: express X as a fraction of the ceiling measured for a fixed *reference rig* and pose grid,
       so the target means the same thing for every candidate rig.
  2. **Time budget:** a sim-seconds cap rather than a step count, so it doesn't change if `physics_dt` changes.
     10,000 steps at 1/120 s ≈ 83 s.
  3. **NBV goal budget:** a maximum number of views/goals.
  4. **Diminishing returns:** the best candidate's predicted score is < ε, or realized coverage gain over the
     last k goals is < ε.
  5. **Candidates exhausted:** no unvisited candidate after expanding the search radius to a max. This replaces
     today's `raise ValueError("no unseen_closes_idxs")`.
  6. **Unreachable goals:** a per-goal timeout. On timeout, mark the pose unreachable and pick the next one.
     After too many timeouts, end the env with status `failed`, so the optimizer can penalize it.
- Not a condition: a path-length cap. Path length is objective 2, so capping it would distort that objective.
- Interaction with the objectives:
  - An env that hits the coverage target stops early, so it gets a short path but its coverage is capped at
    the target.
  - An env that never reaches the target runs to the time or goal budget, so its long path is the penalty.
  - Both are intended. Note that the coverage objective can't tell rigs apart above the target.

**Goal reaching (found 2026-10-26). Root cause fixed:** the script set `robot.sim_state` instead of calling
`env.unwrapped.update_sim_state("evaluate")`, so the IK controller never ran. The items below remain worth checking
once the robot moves.
- In the latest debug run, neither env ever reached its first goal; it stalled ~0.4 m and ~1.2–1.6 rad away.
- The goal tolerances are also very tight: 3 mm per axis, 0.0065 rad (0.37°).
- Investigate before tuning termination:
  - Is the goal reachable? The joint states were collected before the fixed base; the plan is to re-run.
  - Is IK driving the same body whose pose the goals come from (`ur10e__wrist_3_link`)?
  - Do the DiffIK command frames match (base vs world)?
- The per-goal timeout (condition 6) makes a stall like this end the env instead of silently using the budget.

**Unreachable goals: diagnosis (2026-10-28, `SL-PDC_20261028_114336`).** The cause is IK branch drift and joint
wind-up, not unreachable poses. Every stalled goal has a valid joint configuration in `joint_states.h5` (the capture
reached it).
- DiffIK is local and has no joint-limit or posture awareness, with 7 DoF (slider plus UR10e, so 1 redundant).
- `wrist_1` (±360°, continuous) winds steadily (−3.14 → −4.3 rad over ~60 s). Then the arm flips to the other
  wrist/elbow branch: `wrist_2` goes − → +, the elbow goes + → −, `wrist_3` gains ~+π. env_0 flipped at 63 s and
  env_1 at 75 s (wrist) / 122 s (elbow).
- Within ~10 s of a flip, `wrist_1` reaches −2π (its limit), and from then on the EEF stalls 1–4 cm and 1–3° from
  goals that need large reorientations (34–59°).
- env_0 also had the slider pinned at ±0.4 m (its default, 0.3999, sits at the upper limit) and, at goal 85,
  the elbow at 0 (fully stretched, singular).
- Separately, a near-miss: env_1 goal 145 sat within 1.5 mm and 0.37° for 5 s, against a 0.372° (0.0065 rad)
  orientation tolerance.

Fix options (pending decision):
1. **Joint-space goals.** Command the captured `joint_states.h5` configuration for the goal pose, interpolated
   from the current q. It always lands on a valid, known branch and needs no IK. Continuous joints must be
   unwrapped to the nearest equivalent angle within limits, to avoid needless 2π spins. EEF paths are no longer
   Cartesian-straight; path length is still measured on the EEF.
2. **DiffIK plus null-space posture.** Use the redundant slider DoF and the null space to pull toward the goal's
   captured q (or a nominal posture), with joint-limit avoidance. This keeps Cartesian motion, but local
   minima remain possible.
3. Either way, loosen the goal tolerance (e.g. 5 mm / 1°) or accept "settled within a looser tolerance". Consider
   a slider default away from the limit.

**Proposed starting values (for discussion; revisit after the albedo fix and once goals are reached).**
These go in a `TerminationConfig` dataclass, written to `run.yaml`.

| # | condition | parameter | proposed | reasoning |
|---|---|---|---|---|
| 1 | coverage target | `coverage_target_frac` | 0.8 × reference-rig ceiling (total, ≥ `min_count` hits) | an absolute % depends on the metric; a fraction of a fixed reference ceiling means the same for every rig |
| 2 | time budget | `max_sim_time_s` | 60–75 s | measured 2026-10-26 (short run, 40-pose grid): ~1.0–1.25 s per goal, so 50 goals ≈ 50–63 s; re-measure on the full grid |
| 3 | goal budget | `max_goals` | 50 | bounds runtime independently of controller speed |
| 4a | predicted gain floor | `min_rel_predicted_gain` | best candidate score < 1% of the first goal's best score | relative, so independent of score scale and bin count |
| 4b | realized plateau | `plateau_window`, `plateau_min_gain` | last 5 goals add < 0.25% of the ceiling each | catches "keeps moving, sees nothing new" |
| 5 | candidates exhausted | `max_search_radius` | 2 × `pose_search_radius` (1.0 m) | the radius expansion is bounded; beyond it the env is done |
| 6 | unreachable goals | `goal_timeout_s`, `max_consecutive_timeouts`, `max_total_timeouts` | 5 s, 3, 10 | a timeout skips that pose; repeated timeouts end the env as `failed` |

**Cleanup.**
- One helper, `assign_goals(env_ids)`, used both before the loop and inside it. It replaces the duplicated goal
  block and fixes the initial-goal double indexing (`goal_pose_w[env_idxs, ...]`).
- Termination state lives in a small per-env struct: `done`, `reason`, `t_final`, `n_goals`, `n_timeouts`.
- Factor the capture pass and the NBV pass into functions that return `SimOutputs`, so `sim_worker.py` can call
  them.

### Phase 2b: coverage reporting
- Add a per-class coverage summary to the sim outputs, built with the existing
  `coverage_evaluation.coverage_metrics`. For each env:
  - `{trunk, branch, spur, total}` × `{bins_seen, bins_total, pct_bins_seen}`
  - the `min_count` threshold (currently 5) stored alongside.
- Class comes from the prim path, using the same rule as `_semantic_weight_from_prim_path` (spur > branch >
  trunk). Move that rule into one shared function (`instance_parser.parse_instance`) so NBV scoring and
  coverage can't disagree.
- Coverage is unweighted per class. The semantic weights stay inside NBV scoring only.

### Phase 3: `sim_worker.run_simulation`
- rig yaml path → env → capture → NBV → `SimOutputs`.
- One subprocess per evaluation (already designed in `sensor_cfg_problem.simulate_on_gpu`). The worker gets
  `--rig_yaml_path` and `--run_id`, never an in-memory dict, so every evaluated rig is on disk and can be re-run.

### Phase 4: rig encoding, storage and pose assignment

**Current behaviour.**
- `RigGenerator.generate_eef_layout` runs Lloyd relaxation on a spherical cap (or plane) to get N evenly spaced
  mount poses.
- `generate_eef_sensor_cfgs` gives the i-th sensor in the yaml the i-th pose. The comment there asks whether
  this should be randomized.
- The genome is therefore effectively (layout type, radius/colatitude, ordered list of models). Mount
  positions are never searched.

**Genome (decision Q-E: per-sensor mount poses).** A fixed-length vector of `K` slots
(`K = sensors.max_total` from the search-space yaml), so it maps directly onto pymoo `MixedVariable`
(`Choice` + `Real`) and pymoo's built-in mixed-variable operators:
- per slot: `model ∈ {none} ∪ catalog` (Choice), plus 2 surface coordinates (Real):
  - spherical: `(azimuth_deg, polar_deg)`
  - planar: `(x, y)`
- per rig: surface parameters, `radius` (sphere) or `width`/`height` (plane). Whether the surface *type* is a
  gene is open (Q-G).
- The orientation is the mount surface's outward normal; there is no tilt gene.
- Sensor count is implicit: the number of non-`none` slots.
- Constraints (pymoo `G`): at least 1 sensor, `min_spacing` between used slots.

The ranges and `min_spacing` come from `scripts/moo/config/*_search_space.yaml`, which already defines these
fields.
- Assignment is no longer a separate step: each sensor carries its own pose. This answers the "which sensor
  goes where" question. It also replaces the Cursor plan's "slot assignment" gene.
- Baselines are injected as seeded individuals, not genes: Lloyd/even-spread layouts and the Cursor plan's
  "foveated" decode (highest-resolution sensor on the axial slot).
- The genome is a *set*. Canonicalize it by sorting genes (model, then coordinates) before hashing and before
  `ConfigDuplicateElimination`, so permutations of the same rig aren't simulated twice.
- `min_spacing` violations are repaired in mutation/crossover (push apart, or resample), or returned as a
  pymoo constraint `G`.
- Lloyd layouts are kept as a **sampling seed**: they give a good, well-spread initial population. They are no
  longer the only source of poses.

**Decoder / encoder** (`scripts/moo/moo_decoder.py`, currently empty):
- `decode(genome) -> rig dict` writes explicit per-sensor poses:
  `sensors: [{name, model, sensor_type, pose: [x, y, z, qx, qy, qz, qw]}]`, plus `layout: {type: explicit}`
  and `genome` stored verbatim.
- `encode(rig dict) -> genome` lets hand-written rigs (like `test_rig0.yaml`) be injected into the population.
- `RigGenerator` change: if every sensor entry has a `pose`, use it directly. Otherwise fall back to the Lloyd
  or planar layout. Existing rigs keep working.
- Sensor names are generated (`<model>_<k>`), so uniqueness is guaranteed.

**Storage:**
- `data/rigs/<run_id>/<rig_id>.yaml`, where `rig_id` is a short hash of the canonical genome.
- `data/rigs/<run_id>/index.csv` holds one row per evaluated rig: `rig_id, generation, objectives,
  capture path, eval path, status`.
- The index lets a run resume, skips re-simulating a `rig_id` already evaluated, and gives the Pareto front
  traceable back to sim data.
- `data/rigs/test_rig0.yaml` stays as the hand-written placeholder (outside any run dir).

**Catalog.** Every sensor yaml needs `cost`. `helios2.yaml` is missing it.

**Problem fixes:**
- Objective vector: replace the pooled `mean_coverage` with a weighted sum of per-class coverage (Q-F).
  `n_obj` stays 3. The class weights go in the optimizer config.
- Cost lookup: costs are keyed by model yaml stem, but `sensor_cfg_cost` iterates sensor *names*. It should sum
  `sensor_costs[s["model"]]` over the rig's sensor list.
- Implement Sampling, Crossover and Mutation over the genome above (separate planning pass).

### NSGA-II design (proposed for lock-in 2026-10-28; supersedes the slot genome in Phase 4 and the draft site
### lattice)

**State of the code.**
- pymoo and dill are not installed.
- `run_nsga2.py` and `sensor_cfg_problem.py` are scaffolding around a dict genome with stub operators;
  `callbacks.py` is empty.
- `RigGenerator` assigns the i-th sensor in the rig yaml to the i-th Lloyd site (site 0 = the pinned pole).
- `spherical_sensor_layout_generator.py` imports only numpy, plotly and dataclasses, so the optimizer can load it
  by file path (like `coverage_metrics.py`) and compute the same sites the simulation will use.
- Catalog: `helios2.yaml` is empty; `as-dt1.yaml` lacks `width`/`height`/`dfov`; `d780` is excluded.

**Decisions.**
- Compute: one local GPU now; a 24 GB GPU next. HPC is unavailable (Apptainer only on Rocky Linux; Docker
  uncertain). Keep the GPU-slot runner abstraction so multi-GPU stays possible, but don't prioritize it.
- Approved: checkpointing, an evaluation cache, duplicate checks, the naming scheme.
- N1: NSGA evaluations run to the time budget. The coverage-target termination is **off** in NSGA mode;
  plateau, candidates-exhausted and unreachable-goals still apply. Objective 1 is final coverage, so good rigs
  stay separable.
- N2: objective 2 is measured against a **reference coverage** (0.4): the movement needed to first reach it.
- N3: (a) class-agnostic total over instances. Per-class values are still logged.
- N4: only `vl53l8cx` while debugging; the user completes the catalog later.
- N5: max sensors is a hard bound. Cost is unconstrained. VRAM has no hard limit: a CUDA OOM marks the rig
  infeasible and is logged, and the run continues.
- N6: **sites come only from the Lloyd generator.** The colatitude is a platform-shape input to it, not a site
  lattice: the GA learns the mount platform (radius, colatitude) and the sensor configuration together.
- N7: 20 envs per evaluation (the user is testing performance); the same trees every time (`configure_seed`);
  the best rig is tested on new trees afterwards.
- N8: env averaging is enough for now.
- N9: capture density is a later check (see "Later checks").

**Genome (integer vector).**
```
[ radius_idx, colat_idx, n_sensors, model_0, model_1, ..., model_{N_max-1} ]
```
- `radius_idx` indexes `radii_m`, e.g. `[0.10, 0.15, 0.20, 0.25, 0.30]` (0.05 m steps).
- `colat_idx` indexes `colatitudes_deg`, e.g. `[10, 20, 30, 40, 50, 60]` (10° steps).
- `n_sensors ∈ [1, N_max]`, with `N_max = sensors.max_total` (the hard max).
- `model_i ∈ [0, K−1]` indexes the allowed models. Only the first `n_sensors` are used: `model_i` goes on Lloyd
  site i (site 0 = pole). This makes the sensor → site assignment explicit.
- Decode is a pure function: Lloyd(radius, colat, n_sensors) is deterministic. The rig yaml lists sensors in
  site order, and `RigGenerator` already places sensor i on site i; it should also assert that the count
  matches.
- Canonical form for hashing and duplicates: `(radius_idx, colat_idx, n_sensors, model_0..model_{n−1})`, with
  unused model genes ignored.
- Note: with one allowed model, the space is only |radii| × |colats| × N_max (e.g. 5 × 6 × 20 = 600 rigs), and
  the model genes are inert. The GA machinery is meant for the multi-model case.

**Feasibility constraint (checked before simulating, so infeasible rigs cost no GPU time).**
- After Lloyd, the minimum center-to-center distance between neighboring sites must be ≥ the larger footprint
  diagonal of the two sensors (from each yaml's `dimensions`): `G = footprint − min_dist ≤ 0`.
- Infeasible rigs get `G > 0` and placeholder F values, and are never simulated. pymoo's constraint domination
  ranks them below all feasible rigs.

**Objectives (minimize); each is the mean over envs.**
1. `F1 = −coverage`: the final fraction of all instances with ≥5 hits (class-agnostic total), after a
   fixed-budget run.
2. `F2 = path length` (m) until the env's total coverage first reaches `c_ref = 0.4`. An env that never reaches
   it gets its total path × (1 + (c_ref − c_final) / c_ref), i.e. worse the further short it falls. Time to
   `c_ref` is logged as well. *(Confirm path vs time.)*
3. `F3 = Σ sensor cost` (USD), from the catalog; no simulation needed.

**Operators.**
- **Sampling:** pymoo integer random sampling plus seeds: `test_rig0` (r 0.20, colat 25 → nearest 20/30, 8 ×
  vl53l8cx) and spread single-model rigs over a few (r, colat, n).
- **Ordinal genes** (radius, colatitude, n_sensors): SBX crossover + polynomial mutation with `RoundingRepair`,
  pymoo's standard integer recipe. Mutation moves are mostly ±1 step.
- **Categorical genes** (models): uniform crossover + random-reset mutation (a custom operator, composed with
  the above).
- **Repair:** clip to bounds, `n_sensors ≤ N_max`.
- **Duplicates:** `ElementwiseDuplicateElimination` on the canonical form, plus the evaluation cache.

**Population.**
- About 16 min per rig measured (2 envs, laptop); 20 envs will change this, so re-measure first.
- Start at pop 20–24 and 25–40 generations. Track hypervolume per generation to decide when to stop.
- With one model, compare against simply sweeping the 600-rig grid (same cost order); the GA pays off once
  several models are allowed.

**Checkpointing.**
- `CheckpointCallback(Callback)`: after every generation, dill the algorithm (resume: load it and call
  `minimize(problem, algorithm, …, copy_algorithm=False)`). Also write `generations/gen_XXXX/population.csv`
  (genome, rig_id, F, G, rank, crowding) and the hypervolume.
- The problem is picklable: the GPU pool / subprocess runner is excluded via `__getstate__`.
- Evaluation cache: `index.csv` keyed by rig_id. `_evaluate` reuses finished `metrics.json`, so an interruption
  loses only the running simulation(s).

**Naming and files.**
- run_id `nsga_<YYYYmmdd_HHMMSS>[_<tag>]`; rig_id = the first 10 hex chars of the SHA-1 of the canonical genome
  plus the search-space hash; sensor names `<model>_s<site:02d>`.
- `data/runs/<run_id>/{run.yaml, index.csv, generations/gen_XXXX/, rigs/<rig_id>.yaml,
  evals/<rig_id>/{capture.h5, nbv.h5, metrics.json, sim.log}}`.

**Rig yaml (extends `test_rig0.yaml`; read via `--rig_yaml_path`).**
```yaml
schema_version: 1
name: rig_3f2a9c1b7e                 # = rig_id
run_id: nsga_20261028_140000
generation: 7
parents: [rig_aa.., rig_bb..]
search_space: spherical_v1           # search-space name + hash
genome: {radius_idx: 2, colat_idx: 1, n_sensors: 4, models: [0, 0, 0, 0]}
layout: {type: spherical, radius: 0.20, colatitude: 20}   # Lloyd inputs (same keys as test_rig0)
sensors:                             # site order: sensors[i] sits on Lloyd site i (0 = pole)
  - {name: vl53l8cx_s00, model: vl53l8cx, sensor_type: tof, site: 0}
  - ...
derived:                             # informational, recomputable from genome + Lloyd
  n_sensors: 4
  cost_usd: 99.80
  min_site_spacing_m: 0.041
  site_poses: [[x, y, z, qx, qy, qz, qw], ...]   # EEF frame, +Z = optical axis
```

**Simulation worker changes (`run_sensor_learning.py`).**
- `--out_dir`: write `capture.h5` / `nbv.h5` / `metrics.json` there instead of `data/capture|evaluation`.
- `--nsga`: turns off the coverage-target termination; computes the F1/F2 inputs.
- `metrics.json`: per env and mean, with
  - final coverage (total and per class, ≥1/≥5),
  - extent,
  - path and time to `c_ref`,
  - total path,
  - `t_end` and termination reason,
  - timing,
  - and a rig summary.
- Wrap the run so CUDA OOM or crashes produce `metrics.json` with `status: failed` and a reason.

**pymoo integration.**
- `ElementwiseProblem(n_var = 3 + N_max, n_obj = 3, n_ieq_constr = 1, xl, xu, vtype=int)`.
- `NSGA2(pop_size, sampling, crossover, mutation, repair, eliminate_duplicates)`, run through `minimize` with
  the checkpoint callback.
- `elementwise_runner = StarmapParallelization` over a GPU-slot thread pool (1 slot now).
- Each evaluation:
  1. check the cache;
  2. decode → feasibility;
  3. write the rig yaml;
  4. run the subprocess;
  5. read `metrics.json`;
  6. fill F and G.

**Implementation order (step 2, after this plan is confirmed).** Steps 1–5 DONE 2026-10-28 (see changelog). The simulation script is
`scripts/evaluate_rig.py`; it takes `--coverage-target none` instead of `--nsga`, and all flags use hyphens.
1. Install pymoo and dill. Add a standalone `moo/rig_codec.py`: genome ↔ canonical form ↔ rig_id ↔ rig yaml.
   Lloyd is loaded by path. Includes the feasibility check. Unit tests.
2. Sim worker: `--out_dir`, `--nsga`, `metrics.json`, failure capture. `RigGenerator` asserts the sensor count.
3. `SensorCfgProblem` (eval cache, runner, F/G) and the operators (sampling with seeds, ordinal + categorical
   variation, repair, duplicates). Unit tests with a fake simulator.
4. `CheckpointCallback`, the run directory, `run.yaml` (seed, git hash, catalog snapshot, search space, configs),
   and resume.
5. `run_nsga2.py` CLI. A dry run with the fake simulator, then one real generation with a tiny pop.

**Later checks (step 3).**
- N9: capture-grid density vs NBV ranking fidelity (full vs every 2nd/4th pose, on 3–4 rigs). Not now.
- Runtime with 20 envs, and the 24 GB GPU; then revisit pop size / generations. **Measured 2026-10-28 (8 GB laptop
  GPU, 8 sensors, 20 envs):** capture 0.609 s/pose and NBV RTF 0.046, i.e. ~1.8 h per full rig (4200 poses, 180 s).
  Pop 20 × 30 generations would be ~1100 GPU-hours. Capture density (N9), envs per evaluation, the budget, and
  the 24 GB GPU's speed decide what is feasible. Note (smoke run, 2026-10-28): cost per rig
  grows with sensor count (capture 0.037 → 0.172 s/pose, NBV RTF 0.64 → 0.17 for 1 → 19 sensors), so budget for the
  large-rig end.
- DiffIK limitations (TODO section).

### Phase 5: NSGA-II run, checkpointing and artifacts (merged from the Cursor plan, 2026-10-26)
- `scripts/moo/nsga2.py` (does not exist yet):
  - `MixedVariable` problem, `NSGA2`, `minimize`.
  - The optimizer process never imports Omniverse.
  - Each evaluation is one subprocess running capture + NBV in a single `AppLauncher` (Phase 3).
  - A failed sim marks the individual infeasible through `G`; it is not silently given a penalty value.
- Workers: GPU queue in `sensor_cfg_problem.py`. Serial first (local laptop: 1× RTX 5060, 8 GB); parallel on HPC.
- Checkpoint (both layers):
  1. `dill` of the pymoo algorithm after every generation, so a run resumes with the same population, crowding
     and RNG.
  2. Decoded members on disk.
  - CLI: `--resume PATH`, `--checkpoint_every 1`.
- **Run dir** (this replaces the Phase 1 and Phase 4 storage paths with a single root):
  ```
  data/runs/<run_id>/
    run.yaml                 # seed, git hash, catalog snapshot, cost table, pymoo settings, n_envs, termination cfg
    index.csv                # one row per evaluated rig: rig_id, generation, objectives, status, paths
    generations/gen_XXXX/    # population.csv (X, F, G, pareto flag), algorithm.pkl, wall time
    rigs/<rig_id>.yaml       # decoded rig (explicit poses + genome)
    evals/<rig_id>/
      metrics.json           # per-env + mean: per-class coverage, path length, t_final, termination reason, cost
      nbv_path.h5            # EvaluationDataLogger output: poses, joints, goals, sparse bin updates, bin_counts
      capture_summary.h5     # per-pose sparse deltas + tree/label metadata (Phase 1)
  data/capture_raw/<run_id>/<rig_id>/SL-PDC_<datetime_id>_raw.h5   # only with --save_raw
  ```
- Raw capture for Pareto members: re-run those rigs with `--save_raw` at the end of a run (same seed), rather
  than saving raw images for everyone and pruning later. Pareto membership isn't final until the run ends.
- The worker writes `metrics.json` plus h5 files. This replaces the `SimOutputs` pickle.
- Later analysis: hypervolume per generation, Pareto plots, per-tree spread, trajectory overlays from
  `nbv_path.h5`.

**Superseded items in the Cursor plan:**
- Rig yamls under `scripts/moo/config/rigs`: rigs now live under `data/` (Q-D).
- Per-sensor poses listed as out of scope: they are now in scope (Q-E).
- Integer count per model plus a slot-assignment gene: replaced by the slot genome.
- Objective 1 = mean pooled `pct_bins_seen`: now a weighted sum of per-class coverage (Q-F).
- Skipped poses: gone.
- Capture "in memory": satisfied by Phase 1 (deltas are small).
- Out of scope, unchanged: changing trees or robot mid-run.

## TODO (lower priority)
- **DiffIK limitations** (diagnosed 2026-10-28; see "Unreachable goals: diagnosis" in Phase 2). Deferred while
  the focus is on the sensor-learning output.
  - Requirement: keep the EEF pointed at the target. Orientation tracking matters for coverage, so a fix must
    preserve it; this argues for option 2 (DiffIK plus null-space posture and joint-limit avoidance) or a hybrid,
    over a plain joint-space move.
  - Also: the goal tolerance (0.0065 rad is tighter than some near-misses), and the slider default at its +0.4 m
    limit.

## Deferred (decided 2026-10-28 to revisit after termination)

### Junction-aware coverage
A spur is only useful for pruning if it can be placed on its parent, so junction information matters as much as
the spur itself. The idea is a dependency-aware score: a spur is **usable** when
- (a) it is continuous;
- (b) its base is seen;
- (c) the parent's surface near the attachment point is seen.

The coverage target would then become "fraction of usable spurs ≥ X". For now the coverage score stays as is
(instance detection plus extent).

Finding (2026-10-27): the USD hierarchy is flat. Every branch and spur is a direct child of `trunkId_0`, so
spur → parent links are not in the prim tree.

Open questions:
1. Does the tree generator record each spur's parent and the attachment location (curve parameter)? If not,
   infer it geometrically: the branch centerline closest to the spur's start point.
2. What counts as "junction seen": the spur's first k length bins? A stretch of the parent around the attachment
   (how long)? This depends on G = 0 being the base (see the extent boundary-conditions TODO).
3. Which branches matter: all, or only spur-bearing ones?
4. Should the target use detection (≥5 hits) or continuity (closer to "the reconstruction makes sense", but
   ~2% of spurs at ≥5 hits currently)?
5. Strictness: one dependency-aware score, or per-class floors (e.g. spurs ≥ 0.5 and branches ≥ 0.3)?
   Independent per-class targets can't express the spur → parent dependency.

### Relative coverage targets: rejected as a bound
The grid sum is not an upper bound on continuous exploration: 58% of the bins hit live are never hit from any
grid pose. For a grid to bound it, neighboring poses would have to move each zone's ray on the surface by less
than one bin. On an 8×8 ToF at ~1 m a spur bin is ~0.5 × 0.1 cm, so that needs sub-mm and sub-degree spacing.
Any normalization should be empirical instead: a fixed reference rig's live result on the same trees.

### 3D run visualization (back burner, assessed feasible 2026-10-28)
- **EEF path in 3D, colored by time: DONE (2026-10-28)**, as the third figure of `plot_nbv_run.py`. Shown in the base
  frame, without the tree, until env origins / tree poses are aligned. Original notes:
  - The evaluation h5 `pose` holds EEF positions in the base frame. The capture h5 stores each tree's
    `position` / `orientation` and USD path.
  - Draw it as a plotly `Scatter3d` line colored by NBV time, with goal markers and stop markers.
- **Seen bins in 3D: PAUSED** until the tree UV bugs are fixed upstream (waiting on the lpy_treesim author).
  Feasible offline, because `pxr` imports in the venv without Isaac Sim.
  - Open the tree USD from `trees/env_i.attrs["tree_usd_path"]`. For each instance mesh, read its vertices and its
    UV primvar.
  - Map each vertex's UV to (length_bin, theta_bin) with the same decode as `BinnedTree.uv_color_to_bin`.
  - Place each bin at the mean of its vertices, then transform by the tree pose.
  - Scatter the bins colored by hit count (0 / 1 / ≥5).
  - Check first: that the UV axis matching R (theta) / G (length) is u / v and not flipped. The prim → custom-id
    map is already in the capture.
- Probably best as a second HTML report, next to the coverage report from `plot_nbv_run.py`.

### Pre-screening NSGA children (future phase, after NBV)
User's idea: precompute dense per-sensor-type observations for fixed trees, and skip children that can't beat
their parent. Notes:
- **Ray-cast surrogate instead of a dense pose table.** 8×8 ToF zones are essentially rays. Ray-casting the
  static tree meshes (Warp mesh ray casting / Isaac Lab `MultiMeshRayCaster`) gives the instance and UV for any
  rig and EEF pose, with no table and no rendering. It is probably more faithful than a pinhole render for ToF.
  It requires the same trees in every evaluation.
- **Rank children rather than eliminate them.** Sound elimination needs dominance on all three objectives.
  Cost is exact before simulation, but there's no useful lower bound on path length or time, and the surrogate
  gives a coverage estimate, not a bound. So use **surrogate-assisted NSGA-II**:
  - score every child cheaply;
  - fully simulate only the most promising per generation;
  - keep checking the surrogate's ranking against the simulation.

## Pending decisions

- **Q-A, termination. IMPLEMENTED (2026-10-28)** in `nbv/termination.py`, with the per-env reasons recorded in
  the evaluation h5 `termination` table.
  - Values: coverage target = total instances with ≥5 hits ≥ 0.5; 180 s; 300 goals; gain floor 1%; plateau
    0.005 over 20 goals; 1.0 m search radius; 5 s goal timeout (3 consecutive / 10 total).
  - Measured on `SL-PDC_20261026_213705`: median 0.8 s per goal; total instances with ≥5 hits at 0.37 after 83 s
    and still rising ~2 points per 10 s.
  - Revisit the 180 s budget against the NSGA cost per evaluation once runs are timed.
- **Q-B, what capture keeps. DECIDED:** deltas, plus an optional raw dump behind `--save_raw` in a separate,
  labelled dir (Phase 1).
- **Q-C, coverage weighting. PARTLY DECIDED.**
  - NBV *scoring* is already weighted: `score_pose_delta` multiplies gains by `instance_semantic_weight`
    (spur 10, branch 5, trunk 1, matched by prim-path substring with spur checked first).
  - The NSGA *objective* is not weighted yet: `SensorCfgProblem._evaluate` uses the pooled `pct_bins_seen`.
  - **Decided:** the sim reports per-class and total coverage, unweighted (Phase 2b). NSGA-II decides how to
    compare classes and envs. Reason: sparse trunk coverage is enough to infer its structure, while spurs are
    short and harder to detect, so a single fixed weighting shouldn't be baked into the sim.
- **Q-D, rig file. DONE:** every default now points to `data/rigs/test_rig0.yaml`, and
  `config/rigs/test_rig0.yaml` has been removed (see changelog). Rigs live under `data/`, outside the package
  config. Note: `data/` is gitignored, so rigs are not version-controlled. Rigs for a run are recorded through
  `data/rigs/<run_id>/index.csv` (Phase 4).
- **Q-E, genome. DECIDED:** per-sensor mount poses (Phase 4). No tilt gene: each sensor points along the mount
  surface's outward normal, as the Lloyd layouts do today. Tilt can be added later if needed.
- **Q-F, coverage objectives. DECIDED:** the sim outputs per-class coverage only. `SensorCfgProblem._evaluate`
  combines the classes with a weighted sum (weights live in the optimizer config, not the sim). That keeps
  3 objectives: coverage, movement, cost. NSGA-II stays.
- **Q-G, surface type. DECIDED: spherical only for now.**
  - A large-radius sphere cap is nearly planar, so the `radius` range can cover near-planar mounts.
  - If a plane is needed later: give mount coordinates as normalized `(u, v) ∈ [0, 1]²` mapped onto either
    surface. The genome then has the same shape for both, and only the decode changes.
- **Q-H, sensor catalog location. DECIDED:** keep sensor yamls in the package `config/sensors`; snapshot them
  into `run.yaml` at run start.
- **Q-I, slot count `K` and physical fit. PINNED** until population generation.
  - 20 × vl53l8cx likely fits in 8 GB VRAM; 20 × RealSense-class cameras likely won't.
  - Sensors must also physically fit on the mount without overlapping. That means a footprint-based spacing
    constraint (from each yaml's `dimensions`) instead of one global `min_spacing`, and possibly a VRAM budget
    per model.
- **Q-J, coverage metric. DECIDED (2026-10-26): instance-level first.**
  - Per class and in total: the fraction of instances with ≥ k hits (k ∈ `CoverageConfig.instance_min_hits`,
    default 1 and 5). Implemented as `NBVEvaluator.instance_coverage()`, printed in the loop, saved to the
    evaluation h5.
  - Bin-level coverage stays logged (`bin_counts`) for later. Footprint-sized or coarser spur bins are revisited
    only if instance-level detection proves too coarse.
  - Consequences:
    - The termination coverage target (condition 1) and NSGA objective 1 use these instance fractions.
    - `scripts/moo/coverage_evaluation.py` (bin-based) needs an instance-level counterpart in Phase 4.
  - **Extent metrics added (2026-10-27).**
    - Per instance: length coverage, theta coverage, longest length gap, continuous (L = 0.5, g = 2 bins, end
      gaps included).
    - All are reported at ≥1 and ≥5 hits. ≥5 stands in for robustness to real sensor noise, since the sim has no
      noise.
    - Theta is kept as a proxy for branch-radius estimation and robot reachability.
    - The sim reports everything; how these combine into objective 1 is decided downstream on the NSGA side.
    - Physical gap tolerance: **decided against USD lengths for now.** Instances are Bézier curves with cylindrical
      meshes, so "length" is ambiguous, and a USD traversal would be needed. L and g are set per class instead
      (`CoverageConfig` accepts per-class dicts). Defaults (2026-10-27): L = 0.5 for all classes; g = 4 for spurs
      (~2 cm on a ~10 cm spur), 2 for branches (~5 cm on a ~50 cm branch) and 2 for the trunk.
    - **TODO, boundary conditions of the extent metric** (deferred 2026-10-27, revisit after termination):
      - End gaps count toward g, but some instance ends can never be seen: the trunk base at the ground, ends
        attached to a parent or hidden at a junction, ends behind the trellis wire.
      - Within-class length variation: a fixed g is a different physical gap on a 5 cm spur than on a 15 cm one.
      - Instances too small to resolve with an 8×8 zone at working distance.
      - Whether the length-bin direction (G = 0 → 255) runs base → tip consistently for every instance.
      - **Found 2026-10-28: UV `v` exceeds 1 on many instances, and the texture repeats (`wrapT = repeat`).**
        - In `lpy_envy_00000_uv.usda`, 32% of branches and 59% of spurs have v up to 1.33–2.0 (the trunk up to
          1.011). Past v = 1 the texture starts over, so one part of the instance reports the same length bins as
          another part (aliasing).
        - Example: a spur with v ∈ [0, 1.33] reports its last quarter as the first third's bins. Its length
          coverage and continuity are then wrong, and that can't be undone from colors alone.
        - Fix in the tree assets: normalize `st.v` to [0, 1] per instance when generating the UV USDs.
        - Also measured: v = 0 is at the base (attachment end) of a branch. (Correction: branch meshes have many
          rings, e.g. 39 on `branchL_0_Id_000`, with 8 vertices per ring; an earlier ring-count note was wrong.)
        - **Root cause found upstream (2026-10-28), in `/home/luke/lpy_treesim`.** Plain Python; no lpy involved.
          - `tree_generation/lpy_scene_to_mesh.py`, `_compute_t_values`: returns N length fractions for the N + 1
            rings of an N-cylinder part, so the final 1.0 is missing. In `_stitch_cylinder`, the overwrite loop
            therefore never updates the last ring, which keeps the placeholder v = N·v_delta. With
            `v_delta = 1/(N−1)` (line 97, should be 1/N), that is N/(N−1). Verified on all 2233 meshes: last ring
            = N/(N−1), all other rings < 1.
          - Same file, lines 111–116 / 140–147: u = vi/(n_split−1) puts u = 0…1 on 8 distinct vertices (0°…315°).
            The seam face (vertex 7 → 0) interpolates u from 1 back to 0, so theta bins are squeezed into 315° and
            mirrored again inside the last 45°. Fix: a duplicated seam column with u = 1 (or face-varying `st`),
            with u = vi/n_split.
          - `tree_generation/tree_to_usd.py`, `create_uv_material`, lines 480–481: `wrapT = repeat`. Once v is in
            [0, 1], use `clamp` for T, so v = 1.0 and bilinear filtering at the tip don't sample the base row.
          - After the fix: regenerate the `_uv.usda` trees and re-capture; existing data uses the old UVs.
- **Q-K, row sides. ANSWERED:** one side only (cylindrical shell projected from the robot base). At most
  ~half of each instance's theta bins are reachable. Q-J should account for this, either in the denominator
  or by setting targets relative to a reference ceiling.
