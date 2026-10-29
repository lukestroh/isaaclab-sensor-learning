#!/usr/bin/env python3
import isaaclab.utils.math as math_utils
import torch



class GoalIKSolver:
    def __init__(self, device: str):
        self.device = device
        return

    @torch.no_grad()
    def calibrate(self, jac_w, root_quat, q_ref, eef_pos_b, eef_quat_b):
        """Extract spatial screw axis at a reference configuration
        @param jac_w: world frame jacobian of shape (B, 6, N)
        @param root_quat: world frame root orientation of shape (B, 4)
        @param q_ref: reference joint positions of shape (B, N)
        @param eef_pos_b: end-effector position in base frame of shape (B, 3)
        @param eef_quat_b: end-effector orientation in base frame of shape (B, 4)
        """
        r_root = math_utils.matrix_from_quat(root_quat)  # root_quat is (x, y, z, w)
        

        return

    def fk_jac(self, q):
        return

    @torch.no_grad()
    def validate(self):
        return

    @torch.no_grad()
    def solve(self, ):
        return

    