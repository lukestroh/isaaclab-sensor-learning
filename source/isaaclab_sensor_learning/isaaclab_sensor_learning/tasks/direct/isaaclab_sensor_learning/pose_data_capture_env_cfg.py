# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import (
    ArticulationRootPropertiesCfg,
    SimulationCfg,
    RenderCfg,
    MultiUsdFileCfg,
    SpawnerCfg,
    DomeLightCfg,
    MultiAssetSpawnerCfg,
    UsdFileCfg,
    RigidBodyPropertiesCfg,
    CollisionPropertiesCfg,
    PhysicsMaterialCfg,
)
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.universal_robots import UR10e_CFG
from isaaclab_sensor_learning.articulations.ag_robot import AG_ROBOT_UR10e_CFG, AG_ROBOT_UR5e_CFG
from isaaclab_newton.physics import NewtonCfg


from isaaclab_sensor_learning import USD_DIR, URDF_DIR, TREES_DIR
import glob
import os


from isaaclab.controllers import pink_ik
from pathlib import Path

from isaaclab.actuators import ImplicitActuatorCfg


@configclass
class SensorLearningSceneCfg(InteractiveSceneCfg):
    """
    """

    # terrain
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", debug_vis=False)

    # trees
    trees = AssetBaseCfg(
        prim_path="/World/envs/env_.*/tree",
        spawn=MultiUsdFileCfg(
            usd_path=glob.glob(os.path.join(TREES_DIR, "models", "*_uv.usda")),
            random_choice=False,
            rigid_props=RigidBodyPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 2.5, 0.0)),
    )

    #  robot
    robot: ArticulationCfg = AG_ROBOT_UR10e_CFG

    # # lights # TODO: Remove for sim
    # dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0)))


@configclass
class PoseDataCaptureEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 60.0
    # - spaces definition
    action_space = 7
    observation_space = 4
    state_space = 0
    num_envs = 2
    physics_dt = 1 / 120

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=physics_dt,
        render_interval=decimation,
        render=RenderCfg(enable_global_illumination=True, antialiasing_mode="Off"),  # , physics=NewtonCfg()
    )

    # scene
    scene: SensorLearningSceneCfg = SensorLearningSceneCfg(
        num_envs=num_envs,
        lazy_sensor_update=False,  # Change to false for evaluation
        replicate_physics=False,
        env_spacing=5.0,
    )

    

    # pink_ik_cfg = pink_ik.PinkIKControllerCfg(
    #     usd_path=os.path.join(USD_DIR, "ag_robot__ur10e", "ag_robot__ur10e.usda"),
    #     urdf_path=os.path.join(URDF_DIR, "ag_robot", "ag_robot__ur10e.urdf"),
    #     joint_names=[
    #         "linear_slider__joint1",
    #         "ur10e__shoulder_pan_joint",
    #         "ur10e__shoulder_lift_joint",
    #         "ur10e__elbow_joint",
    #         "ur10e__wrist_1_joint",
    #         "ur10e__wrist_2_joint",
    #         "ur10e__wrist_3_joint"
    #     ],        # mesh_path=os.path.join(MESH_DIR, "ag_robot__ur10e"),
    #     all_joint_names=[
    #         "linear_slider__joint1",
    #         "ur10e__shoulder_pan_joint",
    #         "ur10e__shoulder_lift_joint",
    #         "ur10e__elbow_joint",
    #         "ur10e__wrist_1_joint",
    #         "ur10e__wrist_2_joint",
    #         "ur10e__wrist_3_joint",
    #     ],
    #     num_hand_joints=0,
    #     base_link_name="amiga__base",
    #     show_ik_warnings=True,
    #     fail_on_joint_limit_violation=False,
    #     variable_input_tasks=[
    #         pink_ik.FrameTaskCfg(
    #             frame="ur10e__wrist_3_link",
    #             position_cost=7000.0,  # [cost] / [m]
    #             orientation_cost=8000.0,  # [cost] / [rad]
    #             lm_damping=1.0,  # dampening for solver for step jumps
    #             gain=1.0,
    #         ),
    #         # pink_ik.DampingTaskCfg(
    #         #     cost=0.1,  # [cost] * [s] / [rad]
    #         # ),
    #         pink_ik.NullSpacePostureTaskCfg(
    #             cost=2.0,
    #             lm_damping=1.0,
    #             controlled_frames=[
    #                 "ur10e__wrist_3_link",
    #             ],
    #             controlled_joints=[
    #                 "linear_slider__joint1",
    #                 # "ur10e__shoulder_pan_joint",
    #                 # "ur10e__shoulder_lift_joint",
    #                 # "ur10e__elbow_joint",
    #                 # "ur10e__wrist_1_joint",
    #                 # "ur10e__wrist_2_joint",
    #                 # "ur10e__wrist_3_joint",
    #             ],
    #         ),
    #     ],
    #     fixed_input_tasks=[],
    # )

    rig_yaml_path: str = (
        Path(__file__).parent.parent.parent.parent / "config/rigs/test_rig0.yaml"
    )  # NOTE: Standin for now, will be replaced by arguments from evolutionary outputs
    # # rig_yaml_path: str = "pose_data_capture/pose_data_capture/assets/rigs/rig0.yaml"
