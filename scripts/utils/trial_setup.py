#!/usr/bin/env python3
import datetime as dt
import numpy as np
import os
import re
import torch
import isaaclab.sim as sim_utils
from isaaclab.sim.utils import stage as stage_utils
import isaaclab.utils.math as math_utils

from . import pose_generator
from isaaclab.assets import Articulation
from pxr import Usd, UsdGeom


def default_discrete_poses(seed: int, n_envs: int, robot: Articulation) -> tuple[np.ndarray, dict]:
    # Generate discrete poses
    r_range = (0.9, 1.4)
    r_size = 5
    theta_cyl_range = (-np.pi / 8, np.pi / 8)
    theta_cyl_size = 10
    z_range = (1.0, 1.8)
    z_size = 5
    theta_ori_range = (-np.pi / 4, np.pi / 4)
    theta_ori_size = 5
    phi_ori_range = (-np.pi / 2, np.pi / 2)
    phi_ori_size = 5
    start_orientation = np.array([0, 1, 0])
    discrete_poses = pose_generator.generate_uniform_cylindrical_poses(
        r_range=r_range,
        r_size=r_size,
        theta_cyl_range=theta_cyl_range,
        theta_cyl_size=theta_cyl_size,
        z_range=z_range,
        z_size=z_size,
        theta_ori_range=theta_ori_range,
        theta_ori_size=theta_ori_size,
        phi_ori_range=phi_ori_range,
        phi_ori_size=phi_ori_size,
        start_orientation=start_orientation,
    )
    discrete_poses = np.asarray(discrete_poses)

    datetime_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    # trial metadata
    trial_metadata = {
        "datetime_id": datetime_str,
        "seed": seed,
        "num_envs": n_envs,
        "r_range": r_range,
        "r_size": r_size,
        "theta_cyl_range": theta_cyl_range,
        "theta_cyl_size": theta_cyl_size,
        "z_range": z_range,
        "z_size": z_size,
        "theta_ori_range": theta_ori_range,
        "theta_ori_size": theta_ori_size,
        "phi_ori_range": phi_ori_range,
        "phi_ori_size": phi_ori_size,
        "poses": discrete_poses,
        "n_poses": discrete_poses.shape[0],
        "robot_joint_names": robot.data.joint_names,
        "robot_body_names": robot.data.body_names,
    }
    return discrete_poses, trial_metadata


def get_tree_metadata(env) -> dict:
    # tree metadata
    tree_metadata = {}
    tree_prims = [prim for prim in stage_utils.get_current_stage().Traverse() if prim.GetName() == "tree"]
    exclude_re = re.compile(r"/(?:Looks|skeleton)(?:/|$)")
    instance_path_to_id_map = {f"env_{i}": {} for i in range(env.unwrapped.num_envs)}

    # get all tree part prims for indexing bin counts
    for tree_prim in tree_prims:
        next_id = 0
        for prim in Usd.PrimRange(tree_prim):
            path = str(prim.GetPath())
            if path.endswith("tree"):  # This is just the base prim, not a trunk
                continue
            if exclude_re.search(path):
                continue
            if prim.IsA(UsdGeom.Mesh):
                continue
            env_match = re.search(r"env_(\d+)", path)
            if env_match:
                env_idx = int(env_match.group(1))
                instance_path_to_id_map[f"env_{env_idx}"][path] = next_id
            next_id += 1

    # pp.pprint(instance_path_to_id_map)
    for tree_prim in tree_prims:
        # next_id = 0
        prim_path = tree_prim.GetPath()
        match = re.search(r"env_(\d+)", str(prim_path))
        if match:
            env_idx = int(match.group(1))

        usd_path = tree_prim.GetMetadata("references").prependedItems[0].assetPath
        # tree_pose = tree_prim.GetAttribute("xformOp:transform").Get()
        xformable = UsdGeom.Xformable(tree_prim)
        tree_pose = xformable.GetLocalTransformation()
        tree_pose_np = np.array(tree_pose).reshape(4, 4).T
        tree_pos, tree_rot_mat = math_utils.unmake_pose(torch.tensor(tree_pose_np, device=env.unwrapped.device))
        tree_str = os.path.basename(usd_path).split(".")[0].strip("_uv")
        tree_namespace, tree_type, tree_id = tree_str.split("_")
        tree_metadata[env_idx] = {
            "tree_usd_path": usd_path,
            "prim_path": str(prim_path),
            "env_idx": env_idx,
            "tree_namespace": tree_namespace,
            "tree_type": tree_type,
            "tree_id": tree_id,
            "position": tree_pos.detach().cpu().numpy(),
            "orientation": tree_rot_mat.detach().cpu().numpy(),
            "instance_path_to_id_map": instance_path_to_id_map[f"env_{env_idx}"],
        }

    return tree_metadata


def get_sensor_metadata(sensors) -> dict:
    # sensors
    sensor_metadata = {}
    for sensor_name, sensor in sensors.items():
        # print(sensor.cfg)
        # print(sensor.cfg.update_period)
        # pp.pprint(sensor.cfg.asdict())
        sensor_metadata[sensor_name] = {
            "class_type": str(sensor.cfg.class_type),
            "width": sensor.cfg.width,
            "height": sensor.cfg.height,
            "data_rate_hz": 1 / sensor.cfg.update_period,
            "receiver_offset": np.concatenate([sensor.cfg.offset.pos, sensor.cfg.offset.rot]),
            "depth_clipping_behavior": sensor.cfg.depth_clipping_behavior,
            "data_types": sensor.cfg.data_types,
            "z_near": sensor.cfg.spawn.clipping_range[0],
            "z_far": sensor.cfg.spawn.clipping_range[1],
            "focal_length": sensor.cfg.spawn.focal_length,
            "focus_distance": sensor.cfg.spawn.focus_distance,
            "f_stop": sensor.cfg.spawn.f_stop,
            "horizontal_aperture": sensor.cfg.spawn.horizontal_aperture,
            "vertical_aperture": sensor.cfg.spawn.vertical_aperture,
        }
    return sensor_metadata