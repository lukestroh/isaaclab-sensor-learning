#!/usr/bin/env python3
import os

import isaaclab.sim as sim_utils
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

from isaaclab_sensor_learning import USD_DIR, URDF_DIR


def load_usd(
    usd_path: str,
    name: str = "Object",
    env: int = 0,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> str:
    if not os.path.exists(usd_path):
        raise FileNotFoundError(f"USD file not found: {usd_path}")
    prim_path = f"/World/envs/env_{env}/{name}"
    # cfg = sim_utils.UsdFileCfg(usd_path=usd_path)
    # cfg.func(prim_path, cfg)
    sim_utils.create_prim(
        prim_path=prim_path,
        prim_type="Xform",
        usd_path=usd_path,
        position=position,
        orientation=(
            orientation[3],
            orientation[0],
            orientation[1],
            orientation[2],
        ),  # convert from (x, y, z, w) to (w, x, y, z)
        semantic_label=name,
    )
    return prim_path


def unload_usd(prim_path: str) -> bool:
    prim_deleted = sim_utils.delete_prim(prim_path=prim_path)
    return prim_deleted


def generate_usd_from_urdf(
    urdf_path: str,
    output_usd_dir: str = USD_DIR,
    robot_cfg: dict = {
        "base": "amiga",
        "arm": "fr3",
        "eef": None,
        "use_prismatic_axis": True,
    },
    stiffness: dict = {
        "fr3_joint1": 600.0,
        "fr3_joint2": 600.0,
        "fr3_joint3": 600.0,
        "fr3_joint4": 600.0,
        "fr3_joint5": 250.0,
        "fr3_joint6": 150.0,
        "fr3_joint7": 50.0,
    },
    damping: dict = {
        "fr3_joint1": 30.0,
        "fr3_joint2": 30.0,
        "fr3_joint3": 30.0,
        "fr3_joint4": 30.0,
        "fr3_joint5": 10.0,
        "fr3_joint6": 10.0,
        "fr3_joint7": 5.0,
    },
) -> None:
    """
    Generate a USD file from a URDF file using the `usd_from_urdf` function from `isaaclab.sim.utils`.
    """
    # sim_utils.usd_from_urdf(urdf_path=urdf_path, output_usd_path=output_usd_path)
    usd_file_name = os.path.splitext(os.path.basename(urdf_path))[0]
    usd_dir = os.path.join(USD_DIR, "fr3")

    pd_gains_cfg = UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=stiffness, damping=damping)

    joint_drive_cfg = UrdfConverterCfg.JointDriveCfg(
        gains=pd_gains_cfg,
    )

    usd_cvrtr_cfg = UrdfConverterCfg(
        fix_base=True,
        asset_path=os.path.join(URDF_DIR, "fr3/fr3.urdf"),
        usd_dir=usd_dir,
        usd_file_name=usd_file_name + ".usda",
        force_usd_conversion=True,
        make_instanceable=True,
        joint_drive=joint_drive_cfg,
        merge_fixed_joints=False,
        self_collision=True,
    )

    usd_converter = UrdfConverter(cfg=usd_cvrtr_cfg)

    print(usd_converter)

    return usd_converter
