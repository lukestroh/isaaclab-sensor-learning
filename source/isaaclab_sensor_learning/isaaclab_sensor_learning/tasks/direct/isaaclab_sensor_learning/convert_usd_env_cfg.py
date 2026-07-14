# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import (
    SimulationCfg,
    RenderCfg,
    DomeLightCfg
)
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg

import isaaclab_sensor_learning.utils.usd_utils as usd_utils
# from pose_data_capture.sensor.yaml_to_cfg import rig_yaml_to_sensor_cfgs
import isaaclab.sim as sim_utils
from isaaclab_sensor_learning import USD_DIR, URDF_DIR
from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab.assets.articulation import ArticulationCfg, Articulation

import os


@configclass
class ConvertUSDSceneCfg(InteractiveSceneCfg):
     # terrain
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", debug_vis=False)

    dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0)))

    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(
                USD_DIR,
                "linear_slider",
                usd_utils.generate_usd_from_urdf(
                    robot_name="linear_slider",
                    urdf_path=os.path.join(URDF_DIR, f"linear_slider/linear_slider.urdf"),
                    stiffness={"linear_slider__joint1": 200.0},
                    damping={"linear_slider__joint1": 30.0}
                ).usd_file_name
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos = {"linear_slider__joint1": 0.0}
        ),
        actuators={"linear_slider": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=87.0,
            stiffness=200.0,
            damping=30.0
        )}
    )

@configclass
class ConvertUSDEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 60.0
    # - spaces definition
    action_space = 7
    observation_space = 4
    state_space = 0
    num_envs = 1
    physics_dt = 1 / 120

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=physics_dt,
        render_interval=decimation,
        render=RenderCfg(enable_global_illumination=True, antialiasing_mode="Off"),  # , physics=NewtonCfg()
    )

    # scene
    scene = ConvertUSDSceneCfg(
        num_envs=num_envs,
        lazy_sensor_update=False,  # Change to false for evaluation
        replicate_physics=False,
        env_spacing=5.0,
    )