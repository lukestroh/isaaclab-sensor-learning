#!/usr/bin/env python3
import torch


class LimitAwareIK:
    def __init__(self, ik_controller, num_envs, num_joints, device):
        self.ik_controller = ik_controller
        self.joint_locked = torch.zeros(num_envs, num_joints, dtype=torch.bool, device=device)

    def reset(self, env_ids=None):
        """Call when a NEW target pose is issued (or on episode reset) — not every tick."""
        if env_ids is None:
            self.joint_locked.zero_()
        else:
            self.joint_locked[env_ids] = False

    def solve(self, eef_pos_b, eef_quat_b, jacobians, joint_pos, joint_limits, max_iters=10, eps=1e-3):
        working_jac = jacobians.clone()

        env_idx, joint_idx = torch.where(self.joint_locked)
        working_jac[env_idx, :, joint_idx] = 0.0
        joint_pos_des = self.ik_controller.compute(eef_pos_b, eef_quat_b, working_jac, joint_pos)
        joint_pos_des = torch.where(self.joint_locked, joint_pos, joint_pos_des)

        for _ in range(max_iters):
            at_lower = joint_pos <= joint_limits[..., 0] + eps
            at_upper = joint_pos >= joint_limits[..., 1] - eps
            pushing_lower = torch.logical_and(at_lower, joint_pos_des < joint_pos)
            pushing_upper = torch.logical_and(at_upper, joint_pos_des > joint_pos)
            violate = torch.logical_or(pushing_lower, pushing_upper)

            new_violations = torch.logical_and(violate, torch.logical_not(self.joint_locked))
            if not new_violations.any():
                break

            self.joint_locked = torch.logical_or(self.joint_locked, new_violations)
            env_idx, joint_idx = torch.where(self.joint_locked)
            working_jac[env_idx, :, joint_idx] = 0.0

            joint_pos_des = self.ik_controller.compute(eef_pos_b, eef_quat_b, working_jac, joint_pos)
            joint_pos_des = torch.where(self.joint_locked, joint_pos, joint_pos_des)

        return torch.clamp(joint_pos_des, joint_limits[..., 0], joint_limits[..., 1])
