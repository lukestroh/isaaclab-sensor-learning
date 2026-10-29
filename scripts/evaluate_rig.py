#!/usr/bin/env python3

from isaaclab.app import AppLauncher
import argparse
import os



def _float_or_none(value: str) -> float | None:
    return None if value.strip().lower() == "none" else float(value)


# add argparse arguments (flags use hyphens; argparse exposes them as args_cli.with_underscores)
parser = argparse.ArgumentParser(
    description="Evaluate one sensor rig: capture the pose grid, then run the NBV loop (one Isaac Sim process)."
)
parser.add_argument(
    "--disable-fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num-envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Template-Sensor-Learning-Direct-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=0, help="Seed argument for deterministic runs.")
parser.add_argument(
    "--allow-root", action="store_true", default=True, help="Allow running as root user (not recommended)."
)
parser.add_argument(
    "--rig-yaml-path",
    type=str,
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "rigs", "test_rig0.yaml"),
    help="Path to the rig yaml file to evaluate.",
)
parser.add_argument(
    "--max-capture-poses", type=int, default=None, help="Debug: only capture the first N joint states (default: all)."
)
parser.add_argument(
    "--quiet", action="store_true", default=True, help="Skip the per-step coverage printouts (faster)."
)
parser.add_argument(
    "--max-nbv-time-s", type=float, default=None, help="NBV sim-time budget in seconds (default: TerminationConfig)."
)
parser.add_argument(
    "--coverage-target",
    type=_float_or_none,
    default=0.5,
    help="Stop an env once this fraction of instances has >= 5 hits; 'none' disables it (NSGA runs: final coverage "
    "then separates good rigs). Other termination conditions always apply.",
)
parser.add_argument(
    "--reference-coverage",
    type=float,
    default=0.4,
    help="Coverage at which time / path to reference are recorded (NSGA objective 2).",
)
parser.add_argument(
    "--out-dir",
    type=str,
    default=None,
    help="Write capture.h5, nbv.h5 and metrics.json here instead of data/capture and data/evaluation.",
)
parser.add_argument(
    "--robot",
    type=str,
    default="ur10e",
    help="Name of the robot to use in the evaluation.",
    choices=["fr3", "ur10e", "panda"],
)


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import data_utils.capture_data_logger as cdata_logger
import data_utils.evaluation_data_logger as edata_logger
import gymnasium as gym
import h5py
import json
import yaml
import isaaclab.sim as sim_utils
import isaaclab.sim.views as sim_views
import isaaclab.utils.math as math_utils
import isaaclab_sensor_learning.tasks  # noqa: F401
import numpy as np
import pprint as pp
import os
import re
import time
import torch
import utils.pose_generator as pose_generator
import utils.trial_setup as trial_setup


from isaaclab.sim.utils import stage as stage_utils
from isaaclab.sim.utils import semantics as semantic_utils
from pxr import Usd, UsdGeom

from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene
from isaaclab.utils.seed import configure_seed

from isaaclab_sensor_learning import DATA_DIR
from isaaclab_sensor_learning.nbv.nbv import INSTANCE_CLASSES, CoverageConfig, GainConfig, NBVEvaluator, ObservationConfig
from isaaclab_sensor_learning.nbv.reference_coverage import ReferenceCoverageTracker
from isaaclab_sensor_learning.nbv.termination import TerminationConfig, TerminationReason, TerminationTracker
from isaaclab_tasks.utils import parse_env_cfg




def build_metrics(nbv_evaluator, termination, reference, termination_cfg, timing: dict, n_envs: int) -> dict:
    """
    Raw per-env results for the NSGA problem (scripts/moo/sensor_cfg_problem.py), which turns them into objectives.

    Coverage is instance-level ("seen" = fraction of instances with >= k hits); keys k are strings for JSON.
    """
    with open(args_cli.rig_yaml_path) as f:
        rig = yaml.safe_load(f)
    coverage = nbv_evaluator.instance_coverage()
    extent = nbv_evaluator.instance_extent()
    k_ref = termination_cfg.coverage_min_hits
    envs = []
    for env_id, (term_row, ref_row) in enumerate(zip(termination.summary(), reference.summary())):
        env_key = f"env_{env_id}"
        envs.append({
            **term_row,
            **ref_row,
            "coverage_total": coverage[env_key]["total"]["seen"][k_ref],
            "coverage": {cls: {str(k): v for k, v in st["seen"].items()} for cls, st in coverage[env_key].items()},
            "extent": {cls: {str(k): v for k, v in by_k.items()} for cls, by_k in extent[env_key].items()},
        })
    return {
        "status": "ok",
        "rig": {"name": rig.get("name"), "rig_yaml_path": os.path.abspath(args_cli.rig_yaml_path),
                "n_sensors": len(rig["sensors"]), "derived": rig.get("derived")},
        "config": {
            "n_envs": n_envs,
            "reference_coverage": args_cli.reference_coverage,
            "coverage_min_hits": k_ref,
            "termination": termination.config_dict(),
        },
        "envs": envs,
        "timing": timing,
    }


def write_progress(phase: str, **fields) -> None:
    """
    With --out-dir: overwrite <out_dir>/progress.json (atomically) so the NSGA driver can show progress within this
    rig. phase is "capture" (done / total poses) or "nbv" (t / budget seconds of NBV sim time).
    """
    if not args_cli.out_dir:
        return
    os.makedirs(args_cli.out_dir, exist_ok=True)
    path = os.path.join(args_cli.out_dir, "progress.json")
    with open(path + ".tmp", "w") as f:
        json.dump({"phase": phase, **fields}, f)
    os.replace(path + ".tmp", path)


def write_metrics(out_dir: str, metrics: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=float)


def main():
    wall_start = time.perf_counter()  # wall-clock timing of setup / capture / NBV, printed and saved at the end
    configure_seed(seed=args_cli.seed)

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, rig_yaml_path=args_cli.rig_yaml_path)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # reset environment
    env.reset()
    robot: Articulation = env.unwrapped.robot
    robot_entity_cfg = env.unwrapped.robot_entity_cfg
    sensors: dict = env.unwrapped.sensors
    scene: InteractiveScene = env.unwrapped.scene
    sim: sim_utils.SimulationContext = env.unwrapped.sim
    cuda_device = env.unwrapped.device
    n_envs = env.unwrapped.num_envs
    physics_dt = env.unwrapped.physics_dt
    decimation = env.unwrapped.cfg.decimation

    discrete_poses, trial_metadata = trial_setup.default_discrete_poses(seed=args_cli.seed, n_envs=n_envs, robot=robot)
    with h5py.File(os.path.join(DATA_DIR, "joint_states/joint_states.h5"), "r") as f:
        joint_states_data: np.ndarray = f["joint_states"][: args_cli.max_capture_poses]
    # The capture file is sized by the poses actually visited (fewer than the grid with --max_capture_poses)
    n_poses = len(joint_states_data)
    discrete_poses = discrete_poses[:n_poses]
    trial_metadata["poses"], trial_metadata["n_poses"] = discrete_poses, n_poses
    tree_metadata = trial_setup.get_tree_metadata(env=env)
    sensor_metadata = trial_setup.get_sensor_metadata(sensors=sensors)

    # Set up capture data logger
    # same id as trial_metadata, so the capture file name can't drift from it (the logger used its own now())
    cdlog = cdata_logger.CaptureDataLogger(
        datetime_id=trial_metadata["datetime_id"],
        file_path=os.path.join(args_cli.out_dir, "capture.h5") if args_cli.out_dir else None,
    )
    cdlog.save_trial_metadata(trial_metadata=trial_metadata)
    cdlog.save_tree_metadata(tree_metadata=tree_metadata)
    cdlog.save_scene_frames(env_origins=scene.env_origins, robot_base_pose_w=robot.data.root_pose_w.torch)
    cdlog.save_sensor_metadata(sensor_metadata=sensor_metadata, n_poses=n_poses)

    # # Print list of all prims in the scene for debugging
    # from isaaclab.sim.utils.stage import get_current_stage
    # stage = get_current_stage()

    # print(sum(1 for _ in stage.Traverse()), sum(1 for p in stage.Traverse() if p.IsA(UsdGeom.Xformable)), sum(1 for p in stage.Traverse() if p.IsA(UsdGeom.Camera)))
    # import sys; sys.exit(0)

    # simulate environment
    pose_idx = 0
    # step_count = 0
    # step_count_limit = 500
    # max_steps = 0
    # avg_step_count = 0
    # total_steps = 0
    # skipped_poses = []
    # last_obs = False
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel)
    robot.actuators.target_command.set_position_index(value=joint_pos, joint_ids=robot_entity_cfg.joint_ids)
    # Capture teleports between poses, so sim time is meaningless here. Advancing the scene by the longest sensor
    # update period marks every camera outdated, forcing a fresh render after each teleport.
    capture_dt = max(float(sensor.cfg.update_period) for sensor in sensors.values())
    sim.step()
    scene.update(capture_dt)

    time.sleep(5.0)  # wait for the simulation to stabilize

    wall_capture_start = time.perf_counter()
    for joint_state in joint_states_data:
        print(f"\rpose iter: {pose_idx+1}/{len(joint_states_data)}", end="")
        if pose_idx % 10 == 0:
            write_progress("capture", done=pose_idx, total=len(joint_states_data))
        joint_state_t = torch.tensor(joint_state, dtype=torch.float32, device=cuda_device).repeat(n_envs, 1)
        joint_vel_t = torch.zeros_like(joint_state_t, device=cuda_device)
        robot.write_joint_state_to_sim_index(position=joint_state_t, velocity=joint_vel_t)
        robot.actuators.target_command.set_position_index(value=joint_state_t, joint_ids=robot_entity_cfg.joint_ids)
        frames_before = {name: sensor.frame.torch.clone() for name, sensor in sensors.items()}
        sim.step()
        scene.update(capture_dt)  # force a fresh render of every camera at the new pose (see capture_dt)

        robot_eef_w = robot.data.body_pose_w.torch[:, robot.find_bodies("ur10e__wrist_3_link")[0]].reshape(-1, 7)

        observations, rewards, terminated, truncated, info = env.step(action=joint_state_t)
        stale = [name for name, sensor in sensors.items() if not torch.all(sensor.frame.torch > frames_before[name])]
        if stale:
            raise RuntimeError(f"Pose {pose_idx}: no new frame rendered for {stale}; capture would store stale images.")

        
        
        if pose_idx == len(joint_states_data) - 1:
            cdlog.save_pose_observations(observations=observations, idx=pose_idx, last_obs=True)
            cdlog.save_actual_poses(poses=robot_eef_w.detach(), idx=pose_idx, last_obs=True)
        else:
            cdlog.save_pose_observations(observations=observations, idx=pose_idx, last_obs=False)
            cdlog.save_actual_poses(poses=robot_eef_w.detach(), idx=pose_idx, last_obs=False)
        pose_idx += 1
        time.sleep(0.01)
    cdlog.save_labels_dict()  # render id -> prim path maps, needed by the NBV evaluator to bin stored observations
    wall_capture_end = time.perf_counter()


    env.unwrapped.update_sim_state("evaluate")  # enables the IK controller in _apply_action
    print(f"\n[INFO]: Pose data capture complete. Proceeding to NBV evaluation loop.")

    # reset environment
    env.reset()
    #################################################################################################################
    #################################################################################################################

    # trial data path
    trial_data_path = os.path.abspath(cdlog.datafile_path)
    # print(os.path.exists(trial_data_path))
    if not os.path.exists(trial_data_path):
        raise FileNotFoundError (f"Trial data file not found: {trial_data_path}")

    # Set up nbv evaluation data logger
    capture_data_dict = {"n_envs": n_envs, "datetime_id": trial_metadata["datetime_id"]}
    evaluation_metadata = {
        "seed": args_cli.seed,
        "datetime_id": trial_metadata["datetime_id"],
        "associated_data": f"SL_PDC_{trial_metadata['datetime_id']}",
    }
    timestamp_metadata = {"physics_dt": physics_dt, "decimation": decimation}
    evlog = edata_logger.EvaluationDataLogger(
        capture_data_dict=capture_data_dict,
        file_path=os.path.join(args_cli.out_dir, "nbv.h5") if args_cli.out_dir else None,
    )
    evlog.save_evaluation_metadata(evaluation_metadata=evaluation_metadata)
    evlog.save_timestamp_metadata(timestamp_metadata=timestamp_metadata)
    evlog.save_joint_position_metadata(joint_position_metadata={"n_dof": robot.num_joints})
    evlog.save_nbv_goal_metadata()
    evlog.save_sparse_bin_count_metadata()
    evlog.save_pose_metadata()

    # Coverage evaluator
    observation_cfg = ObservationConfig()
    gain_cfg = GainConfig()
    coverage_cfg = CoverageConfig(
        
    )  # instance-level coverage thresholds (hits per instance)
    nbv_evaluator = NBVEvaluator(
        n_envs=n_envs,
        trial_data_path=trial_data_path,
        observation_config=observation_cfg,
        gain_config=gain_cfg,
        coverage_config=coverage_cfg,
    )

    # NBV sim setup
    # Reset robot to default joint positions and velocities
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel)
    robot.reset()
    sim.step()
    scene.update(physics_dt)
    # Get the robot's end-effector pose in world frame
    robot_eef_w = robot.data.body_pose_w.torch[:, robot.find_bodies("ur10e__wrist_3_link")[0]].reshape(-1, 7)
    robot_base_w = robot.data.root_pose_w.torch
    evlog.save_scene_frames(env_origins=scene.env_origins, robot_base_pose_w=robot_base_w)

    # Per-env termination (see nbv/termination.py); finished envs hold their pose and stop binning
    termination_cfg = TerminationConfig(coverage_target=args_cli.coverage_target)
    if args_cli.max_nbv_time_s is not None:
        termination_cfg.max_sim_time_s = args_cli.max_nbv_time_s
    tracker = TerminationTracker(n_envs=n_envs, config=termination_cfg)
    reference_tracker = ReferenceCoverageTracker(n_envs=n_envs, reference_coverage=args_cli.reference_coverage)
    eef_body_id = robot.find_bodies("ur10e__wrist_3_link")[0]

    visited = torch.zeros(nbv_evaluator.poses.shape[:2], dtype=torch.bool, device=cuda_device)  # poses used as goals
    actions = torch.zeros((n_envs, 7), dtype=torch.float32, device=cuda_device)  # (pos in base frame, quat world)

    def hold_pose(env_id: int, eef_w: torch.Tensor, base_w: torch.Tensor) -> None:
        """Freeze a finished env at its current EEF pose."""
        actions[env_id] = torch.cat((eef_w[env_id, 0:3] - base_w[env_id, 0:3], eef_w[env_id, 3:7]))

    def assign_goals(env_ids: list[int], eef_w: torch.Tensor, base_w: torch.Tensor, t: float, step: int, t_abs: float):
        """Choose and command the next best pose for env_ids; envs that cannot continue are finished and held."""
        if not env_ids:
            return
        candidates, exhausted = nbv_evaluator.get_candidate_pose_idxs(
            curr_poses=eef_w, env_ids=env_ids, visited=visited, max_radius=termination_cfg.max_search_radius
        )
        for env_id in exhausted:
            tracker.finish(env_id, TerminationReason.CANDIDATES_EXHAUSTED, t, step)
            hold_pose(env_id, eef_w, base_w)
        remaining = [e for e in env_ids if e not in exhausted]
        if not remaining:
            return
        best, scores = nbv_evaluator.compute_env_info_gain_from_pose_idxs(
            env_ids=torch.tensor(remaining, device=cuda_device), closest_idxs=candidates, return_scores=True
        )
        for (env_id, pose_id), score in zip(best.tolist(), scores.tolist()):
            if tracker.on_goal_assigned(env_id, t, step, best_score=score):  # gain floor or goal budget reached
                hold_pose(env_id, eef_w, base_w)
                continue
            visited[env_id, pose_id] = True
            goal_pose_w = nbv_evaluator.poses[env_id, pose_id]
            actions[env_id] = torch.cat((goal_pose_w[0:3] - base_w[env_id, 0:3], goal_pose_w[3:7]))
            evlog.save_nbv_goal(env_id=env_id, timestamp=t_abs, nbv_goal=actions[env_id])

    # Initial goals from the default pose
    nbv_start_step = env.unwrapped.common_step_counter  # common_step_counter keeps counting from the capture pass
    nbv_start_time = physics_dt * decimation * nbv_start_step
    robot_eef_w = robot.data.body_pose_w.torch[:, eef_body_id].reshape(-1, 7)
    robot_base_w = robot.data.root_pose_w.torch
    assign_goals(list(range(n_envs)), robot_eef_w, robot_base_w, t=0.0, step=0, t_abs=nbv_start_time)

    last_obs = False

    # NBV loop
    while simulation_app.is_running():
        with torch.inference_mode():
            # Log all timestamps
            current_step = env.unwrapped.common_step_counter
            curr_time = physics_dt * decimation * current_step
            nbv_step, nbv_time = current_step - nbv_start_step, curr_time - nbv_start_time
            evlog.save_timestamp(timestamp=curr_time, last_obs=last_obs)
            if nbv_step % 120 == 0:  # ~1 s of sim time
                write_progress(
                    "nbv", t=nbv_time, budget=termination_cfg.max_sim_time_s, active_envs=len(tracker.active_env_ids)
                )

            # Get observations and save the sparse bin updates (finished envs are no longer binned)
            observations, rewards, terminated, truncated, info = env.step(actions)
            # Sensors render at data_rate_hz, slower than physics; frame counters let the evaluator skip repeats.
            frames = {name: sensor.frame.torch for name, sensor in sensors.items()}
            info_gains, dense_bin_updates, sparse_bin_updates = nbv_evaluator.update_bin_count_scores_from_observation(
                env_ids=torch.tensor(tracker.active_env_ids, dtype=torch.long), observations=observations, frames=frames
            )
            evlog.save_sparse_bin_counts(timestamp=curr_time, sparse_bin_updates=sparse_bin_updates, last_obs=last_obs)

            # #################################################################
            # check if the robot's end-effector is close to the goal pose
            # #################################################################
            robot_base_w = robot.data.root_pose_w.torch
            robot_eef_w = robot.data.body_pose_w.torch[:, eef_body_id].reshape(-1, 7)

            # Save eef poses in base frame
            robot_position_b = robot_eef_w[:, 0:3] - robot_base_w[:, 0:3]
            robot_eef_b = torch.cat([robot_position_b, robot_eef_w[:, 3:7]], axis=1)
            evlog.save_poses(poses=robot_eef_b, last_obs=last_obs)

            # Save joint positions
            robot_joint_pos = robot.data.joint_pos.torch
            evlog.save_joint_positions(joint_positions=robot_joint_pos, last_obs=last_obs)

            # Per-step termination checks: coverage target and time budget
            coverage_now = nbv_evaluator.instance_coverage(min_hits=(termination_cfg.coverage_min_hits,))
            coverage_total = {
                env_id: coverage_now[f"env_{env_id}"]["total"]["seen"][termination_cfg.coverage_min_hits]
                for env_id in range(n_envs)
            }
            reference_tracker.update(
                nbv_time, coverage_total, robot_eef_b[:, 0:3].cpu().numpy(), active_env_ids=tracker.active_env_ids
            )
            for env_id in tracker.update(nbv_time, nbv_step, coverage_total):
                hold_pose(env_id, robot_eef_w, robot_base_w)

            goal_position_w = actions.clone().detach()[:, 0:3] + robot_base_w[:, 0:3]
            # position
            positions_close = torch.isclose(goal_position_w[:, 0:3], robot_eef_w[:, 0:3], rtol=0.0, atol=0.003)
            # orientation
            quat_diff_err = math_utils.quat_error_magnitude(actions[:, 3:7], robot_eef_w[:, 3:7])
            orientations_close = torch.isclose(quat_diff_err, torch.tensor(0.0), rtol=0.0, atol=0.0065)
            goal_reached = positions_close.all(dim=1) & orientations_close

            # Active envs that reached their goal or timed out on it get a new goal
            envs_change_goal = []
            for env_id in tracker.active_env_ids:
                if bool(goal_reached[env_id]):
                    stopped = tracker.on_goal_reached(env_id, nbv_time, nbv_step)  # may stop (plateau)
                elif tracker.goal_timed_out(env_id, nbv_time):
                    stopped = tracker.on_goal_timeout(env_id, nbv_time, nbv_step)  # may stop (unreachable)
                else:
                    continue
                if stopped:
                    hold_pose(env_id, robot_eef_w, robot_base_w)
                else:
                    envs_change_goal.append(env_id)
            assign_goals(envs_change_goal, robot_eef_w, robot_base_w, t=nbv_time, step=nbv_step, t_abs=curr_time)

            if last_obs:
                evlog.save_bin_counts(bin_counts=nbv_evaluator.binned_tree.bin_counts)
                evlog.save_instance_coverage(instance_coverage=nbv_evaluator.instance_coverage())
                evlog.save_instance_extent(
                    instance_extent=nbv_evaluator.instance_extent(),
                    per_instance={
                        env_key: {
                            k: nbv_evaluator.instance_extent_per_instance(env_key, k)
                            for k in coverage_cfg.instance_min_hits
                        }
                        for env_key in nbv_evaluator.binned_tree.bin_counts
                    },
                    instance_class_ids=nbv_evaluator.instance_class_ids,
                    class_names=INSTANCE_CLASSES,
                    config={
                        "instance_min_hits": list(coverage_cfg.instance_min_hits),
                        "continuity_min_length_frac": coverage_cfg.continuity_min_length_frac,
                        "continuity_max_gap_bins": coverage_cfg.continuity_max_gap_bins,
                    },  # per-class dicts are flattened to "<name>/<class>" attrs by the logger
                )
                evlog.save_termination(rows=tracker.summary(), config=tracker.config_dict())
                wall_end = time.perf_counter()
                timing = {
                    "setup_s": wall_capture_start - wall_start,  # includes the 5 s settle sleep
                    "capture_s": wall_capture_end - wall_capture_start,
                    "nbv_s": wall_end - wall_capture_end,
                    "total_s": wall_end - wall_start,
                    "nbv_sim_time_s": nbv_time,
                    "n_capture_poses": n_poses,
                    "quiet": bool(args_cli.quiet),
                }
                evlog.save_run_timing(timing)
                print(
                    f"[TIMER]: setup {timing['setup_s']:.1f} s | capture {timing['capture_s']:.1f} s "
                    f"({n_poses} poses, {timing['capture_s'] / max(n_poses, 1):.3f} s/pose) | "
                    f"NBV {timing['nbv_s']:.1f} s for {nbv_time:.1f} s sim "
                    f"(real-time factor {nbv_time / max(timing['nbv_s'], 1e-9):.3f}) | total {timing['total_s']:.1f} s"
                )
                if args_cli.out_dir:
                    metrics = build_metrics(
                        nbv_evaluator=nbv_evaluator,
                        termination=tracker,
                        reference=reference_tracker,
                        termination_cfg=termination_cfg,
                        timing=timing,
                        n_envs=n_envs,
                    )
                    write_metrics(args_cli.out_dir, metrics)
                    print(f"[INFO]: Wrote {os.path.join(args_cli.out_dir, 'metrics.json')}")
                break

            for env_key, env_bin_counts in ({} if args_cli.quiet else nbv_evaluator.binned_tree.bin_counts).items():
                more_than_n_counts_mask = torch.nonzero(env_bin_counts >= 5, as_tuple=False)
                # print(more_than_n_counts_mask)
                # num_completed_bins = torch.numel(more_than_n_counts_mask)
                num_completed_bins = more_than_n_counts_mask.shape[0]
                env_total_bins = np.prod(tuple(env_bin_counts.shape))

                seen_bins = num_completed_bins / env_total_bins

                print(current_step, ": ", env_key, seen_bins)

            # Instance-level coverage: fraction of instances per class with >= k hits (coverage_cfg.instance_min_hits)
            for env_key, by_class in ({} if args_cli.quiet else nbv_evaluator.instance_coverage()).items():
                seen = {cls: {k: round(v, 3) for k, v in stats["seen"].items()} for cls, stats in by_class.items()}
                print(current_step, ": ", env_key, "instances seen", seen)

            # Extent: per class, means over all instances of (length coverage, theta coverage, fraction continuous)
            for env_key, by_class in ({} if args_cli.quiet else nbv_evaluator.instance_extent()).items():
                extent = {
                    cls: {k: tuple(round(v, 3) for v in stats.values()) for k, stats in by_k.items()}
                    for cls, by_k in by_class.items()
                }
                print(current_step, ": ", env_key, "extent (len, theta, continuous)", extent)

            # Terminating condition: every env has stopped (one more pass logs and saves the final state)
            if tracker.all_done and not last_obs:
                for row in tracker.summary():
                    print(f"[INFO]: env_{row['env_id']} stopped: {row['reason_name']} at {row['t_end']:.2f} s "
                          f"({row['n_goals_reached']}/{row['n_goals']} goals, coverage {row['final_coverage']:.3f})")
                last_obs = True

    nbv_evaluator.close()
    env.close()

    return


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        if args_cli.out_dir:  # NSGA: record the failure so the optimizer marks this rig infeasible and moves on
            message = f"{type(e).__name__}: {e}"
            write_metrics(args_cli.out_dir, {
                "status": "failed",
                "error": message,
                "out_of_memory": "out of memory" in message.lower(),
                "traceback": traceback.format_exc(),
            })
    finally:
        simulation_app.close()


