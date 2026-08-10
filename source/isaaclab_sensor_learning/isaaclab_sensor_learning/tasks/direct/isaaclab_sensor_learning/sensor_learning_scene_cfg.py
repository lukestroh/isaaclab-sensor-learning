#!/usr/bin/env python3
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import MultiUsdFileCfg, RigidBodyBaseCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.universal_robots import UR10e_CFG
from isaaclab_sensor_learning import USD_DIR, URDF_DIR, TREES_DIR
from isaaclab_sensor_learning.articulations.ag_robot import AG_ROBOT_UR10e_CFG, AG_ROBOT_UR5e_CFG
from isaaclab_newton.physics import NewtonCfg

import glob
import os

from isaaclab.sim.schemas import CollisionPropertiesCfg


@configclass
class SensorLearningSceneCfg(InteractiveSceneCfg):
    """Configuration for the sensor learning scene."""

    # terrain
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", debug_vis=False)

    # trees
    trees = AssetBaseCfg(
        prim_path="/World/envs/env_.*/tree",
        spawn=MultiUsdFileCfg(
            usd_path=glob.glob(os.path.join(TREES_DIR, "models", "*_uv.usda")),
            random_choice=False,
            rigid_props=RigidBodyBaseCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 2.5, 0.0)),
    )

    robot: ArticulationCfg = AG_ROBOT_UR10e_CFG
