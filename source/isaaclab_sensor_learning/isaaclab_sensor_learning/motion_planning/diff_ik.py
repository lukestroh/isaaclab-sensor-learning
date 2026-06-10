#!/usr/bin/env python3

import torch
from isaaclab.utils.math import subtract_frame_transforms


def _slerp(q1, q2, t):
    """Spherical linear interpolation between two quaternions."""
    dot = (q1 * q2).sum(dim=-1, keepdim=True).clamp(-1, 1)
    # Ensure shortest path
    q2 = torch.where(dot < 0, -q2, q2)
    dot = dot.abs()
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    # Fall back to lerp when angle is small
    safe = sin_theta > 1e-6
    w1 = torch.where(safe, torch.sin((1 - t) * theta) / sin_theta, 1 - t)
    w2 = torch.where(safe, torch.sin(t * theta) / sin_theta, t)
    return torch.nn.functional.normalize(w1 * q1 + w2 * q2, dim=-1)


def _plan_waypoints(start_pose, end_pose, num_waypoints=20):
    """Linear interpolation in pose space between two EEF poses."""
    pos_start, quat_start = start_pose
    pos_end, quat_end = end_pose

    waypoints = []
    for i in range(num_waypoints):
        t = i / (num_waypoints - 1)

        # Lerp position
        pos = (1 - t) * pos_start + t * pos_end

        # Slerp orientation
        quat = _slerp(quat_start, quat_end, t)

        waypoints.append((pos, quat))
    return waypoints


# def run_ik_to_pose(
#     robot,
#     ik_controller,
#     robot_entity_cfg,
#     ee_jacobi_idx,
#     scene,
#     sim,
#     goal_pos_w: torch.Tensor,
#     goal_quat_w: torch.Tensor,
#     step_size_m: float = 0.01,
#     steps_per_waypoint: int = 50,
#     eef_marker=None,
#     goal_marker=None,
# ):
#     # Get current EEF pose in world frame as start
#     eef_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
#     start_pos_w = eef_pose_w[:, 0:3]
#     start_quat_w = eef_pose_w[:, 3:7]

#     # Plan waypoints in world frame
#     waypoints = _plan_waypoints((start_pos_w, start_quat_w), (goal_pos_w, goal_quat_w), num_waypoints=20)

#     for wp_pos_w, wp_quat_w in waypoints:
#         # Convert waypoint to base frame
#         root_pose_w = robot.data.root_pose_w
#         wp_pos_b, wp_quat_b = subtract_frame_transforms(
#             root_pose_w[:, 0:3], root_pose_w[:, 3:7],
#             wp_pos_w, wp_quat_w,
#         )
#         ik_commands = torch.cat([wp_pos_b, wp_quat_b], dim=-1)
#         ik_controller.set_command(ik_commands)

#         for _ in range(steps_per_waypoint):
#             jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
#             eef_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
#             root_pose_w = robot.data.root_pose_w
#             joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

#             eef_pos_b, eef_quat_b = subtract_frame_transforms(
#                 root_pose_w[:, 0:3], root_pose_w[:, 3:7],
#                 eef_pose_w[:, 0:3], eef_pose_w[:, 3:7],
#             )

#             joint_pos_des = ik_controller.compute(eef_pos_b, eef_quat_b, jacobian, joint_pos)
#             robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
#             scene.write_data_to_sim()
#             sim.step()
#             scene.update(sim.cfg.dt)

#             if eef_marker is not None:
#                 eef_state_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
#                 eef_marker.visualize(eef_state_w[:, 0:3], eef_state_w[:, 3:7])
#             if goal_marker is not None:
#                 env_origins = scene.env_origins if scene.env_origins is not None else torch.zeros(scene.num_envs, 3, device=robot.device)
#                 goal_marker.visualize(goal_pos_w + env_origins, goal_quat_w)
#     return

# The key difference is staring us in the face. In the working code:
# python

# ik_controller.set_command(ik_commands)  # ik_commands is in BASE frame already

# The goals are set directly without any frame transform — ik_commands contains base-frame coordinates because the tutorial defines goals in base frame.

# In your function you're transforming world-frame goals to base frame with subtract_frame_transforms — but what if that transform is introducing an error?

# Also notice in the working code set_joint_position_target is called outside the if/else block, meaning it always uses the last computed joint_pos_des. In your function it's inside the loop.

# The simplest fix — just define your goals in base frame and skip the transform entirely in the function:
# python


def run_ik_to_pose(
    robot,
    ik_controller,
    robot_entity_cfg,
    ee_jacobi_idx,
    scene,
    sim,
    goal_pos_b: torch.Tensor,  # base frame directly
    goal_quat_b: torch.Tensor,
    num_steps: int = 500,
    eef_marker=None,
    goal_marker=None,
):
    
    ik_controller.set_command(torch.cat([goal_pos_b, goal_quat_b], dim=-1))

    for _ in range(num_steps):
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        eef_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        root_pose_w = robot.data.root_pose_w
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

        eef_pos_b, eef_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            eef_pose_w[:, 0:3],
            eef_pose_w[:, 3:7],
        )

        joint_pos_des = ik_controller.compute(eef_pos_b, eef_quat_b, jacobian, joint_pos)

        lower = robot.data.soft_joint_pos_limits[:, robot_entity_cfg.joint_ids, 0]
        upper = robot.data.soft_joint_pos_limits[:, robot_entity_cfg.joint_ids, 1]
        joint_pos_des = torch.clamp(joint_pos_des, lower, upper)

        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.cfg.dt)

        if eef_marker is not None:
            eef_state_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            eef_marker.visualize(eef_state_w[:, 0:3], eef_state_w[:, 3:7])
        if goal_marker is not None:
            env_origins = (
                scene.env_origins
                if scene.env_origins is not None
                else torch.zeros(scene.num_envs, 3, device=robot.device)
            )
            goal_marker.visualize(goal_pos_b + env_origins, goal_quat_b)

    return
