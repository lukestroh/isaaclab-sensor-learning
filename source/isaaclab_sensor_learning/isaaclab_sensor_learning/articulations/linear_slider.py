#!/usr/bin/env python3
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from isaaclab_sensor_learning import USD_DIR, URDF_DIR
import os

LINEAR_SLIDER_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/luke/IsaacLab/source/isaaclab-sensor-learning/source/isaaclab_sensor_learning/isaaclab_sensor_learning/usd/linear_slider/linear_slider.usda",
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
        pos = (0.0, 0.0, 0.0),
        rot = (0.0, 0.0, 0.0, 1.0),
        joint_pos = {"linear_slider__joint1": 0.0}
    ),
    actuators={"linear_slider": ImplicitActuatorCfg(
        joint_names_expr=[".*"],
        effort_limit_sim=87.0,
        stiffness=200.0,
        damping=30.0
    )}
)