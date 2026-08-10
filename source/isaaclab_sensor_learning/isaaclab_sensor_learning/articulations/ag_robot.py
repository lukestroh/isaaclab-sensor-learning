#!/usr/bin/env python3
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from isaaclab_sensor_learning import USD_DIR, URDF_DIR
import os

AG_ROBOT_UR5e_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/luke/IsaacLab/source/isaaclab-sensor-learning/source/isaaclab_sensor_learning/isaaclab_sensor_learning/usd/ag_robot__ur5e/ag_robot__ur5e.usda",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        # articulation_props=sim_utils.schemas.ArticulationRootBaseCfg(
        #     articulation_enabled=True,
        # ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "linear_slider__joint1": 0.0,
            "ur5e__shoulder_pan_joint": 3.141592653589793,
            "ur5e__shoulder_lift_joint": -1.5707963267948966,
            "ur5e__elbow_joint": 1.5707963267948966,
            "ur5e__wrist_1_joint": -1.5707963267948966,
            "ur5e__wrist_2_joint": -1.5707963267948966,
            "ur5e__wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "linear_slider": ImplicitActuatorCfg(
            joint_names_expr=["linear_slider__joint1"],
            effort_limit_sim=87.0,
            velocity_limit=0.4,
            stiffness=10000.0,
            damping=5000.0,
        ),
        "shoulder_pan": ImplicitActuatorCfg(
            joint_names_expr=["ur5e__shoulder_pan_joint"],
            stiffness=9400.5009765625,
            damping=1000,
            friction=0.0,
            armature=0.0,
        ),
        "shoulder_lift": ImplicitActuatorCfg(
            joint_names_expr=["ur5e__shoulder_lift_joint"],
            stiffness=10020.919921875,
            damping=1000,
            friction=0.0,
            armature=0.0,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["ur5e__elbow_joint"],
            stiffness=10230.220703125,
            damping=500,
            friction=0.0,
            armature=0.0,
        ),
        "wrist_1": ImplicitActuatorCfg(
            joint_names_expr=["ur5e__wrist_1_joint"],
            stiffness=3940.62890625,
            damping=20,
            friction=0.0,
            armature=0.0,
        ),
        "wrist_2": ImplicitActuatorCfg(
            joint_names_expr=["ur5e__wrist_2_joint"],
            stiffness=3940.62890625,
            damping=0.06109166145324707,
            friction=0.0,
            armature=0.0,
        ),
        "wrist_3": ImplicitActuatorCfg(
            joint_names_expr=["ur5e__wrist_3_joint"],
            stiffness=1000.0808715820312,
            damping=0.004323445726186037,
            friction=0.0,
            armature=0.0,
        ),
    },
)


AG_ROBOT_UR10e_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/robot",
    # articulation_root_prim_path="/Geometry/amiga__base",
    spawn=sim_utils.UsdFileCfg(
        # usd_path="/home/luke/IsaacLab/source/isaaclab-sensor-learning/source/isaaclab_sensor_learning/isaaclab_sensor_learning/usd/ag_robot__ur10e/ur10_amiga_flattened.usda",
        usd_path="/home/luke/IsaacLab/source/isaaclab-sensor-learning/source/isaaclab_sensor_learning/isaaclab_sensor_learning/usd/ag_robot__ur10e/ag_robot__ur10e.usda",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.schemas.ArticulationRootBaseCfg(
            articulation_enabled=True,
            # fix_root_link=True
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "linear_slider__joint1": 0.0,
            "ur10e__shoulder_pan_joint": 0.0,
            "ur10e__shoulder_lift_joint": -1.5707963267948966,
            "ur10e__elbow_joint": 1.5707963267948966,
            "ur10e__wrist_1_joint": -3.141592653589793,
            "ur10e__wrist_2_joint": -1.5707963267948966,
            "ur10e__wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "linear_slider": ImplicitActuatorCfg(
            joint_names_expr=["linear_slider__joint1"],
            effort_limit_sim=87.0,
            velocity_limit=0.4,
            stiffness=40000.0,
            damping=5000.0,
        ),
        "shoulder_pan": ImplicitActuatorCfg(
            joint_names_expr=["ur10e__shoulder_pan_joint"],
            stiffness=1320.0,
            damping=72.6636085,
            friction=0.0,
            armature=0.0,
        ),
        "shoulder_lift": ImplicitActuatorCfg(
            joint_names_expr=["ur10e__shoulder_lift_joint"],
            stiffness=220_000.0,
            damping=10000.0,
            friction=0.0,
            armature=0.0,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["ur10e__elbow_joint"],
            stiffness=300_000.0,
            damping=12_000.0,
            friction=0.0,
            armature=0.0,
        ),
        "wrist_1": ImplicitActuatorCfg(
            joint_names_expr=["ur10e__wrist_1_joint"],
            stiffness=1500.0,
            damping=200,
            friction=0.0,
            armature=0.0,
        ),
        "wrist_2": ImplicitActuatorCfg(
            joint_names_expr=["ur10e__wrist_2_joint"],
            stiffness=1000.0,
            damping=29.39387691,
            friction=0.0,
            armature=0.0,
        ),
        "wrist_3": ImplicitActuatorCfg(
            joint_names_expr=["ur10e__wrist_3_joint"],
            stiffness=216.0,
            damping=29.39387691,
            friction=0.0,
            armature=0.0,
        ),
    },
)

# "shoulder": ImplicitActuatorCfg(
#     joint_names_expr=["ur10e__shoulder_.*"],
#     stiffness=1320.0,
#     damping=72.6636085,
#     friction=0.0,
#     armature=0.0,
# ),
# "elbow": ImplicitActuatorCfg(
#     joint_names_expr=["ur10e__elbow_joint"],
#     stiffness=600.0,
#     damping=34.64101615,
#     friction=0.0,
#     armature=0.0,
# ),
# "wrist": ImplicitActuatorCfg(
#     joint_names_expr=["ur10e__wrist_.*"],
#     stiffness=216.0,
#     damping=29.39387691,
#     friction=0.0,
#     armature=0.0,
# ),
