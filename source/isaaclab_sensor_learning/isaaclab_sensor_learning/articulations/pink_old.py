        # eps = 1e-6
        # at_lower = joint_pos <= joint_limits[..., 0] + eps
        # at_upper = joint_pos >= joint_limits[..., 1] - eps
        # pushing_lower = torch.logical_and(at_lower, joint_pos_des < joint_pos)
        # pushing_upper = torch.logical_and(at_upper, joint_pos_des > joint_pos)
        # violate = torch.logical_or(pushing_lower, pushing_upper)  # shape: (num_envs, num_joints)


        # if violate.any():
        #     constrained_jac = jacobians.clone()
        #     # zero the Jacobian column for every (env, joint) pair that's violating
        #     env_idx, joint_idx = torch.where(violate)
        #     constrained_jac[env_idx, :, joint_idx] = 0.0

        #     joint_pos_des_constrained = self.ik_controller.compute(eef_pos_b, eef_quat_b, constrained_jac, joint_pos)

        #     joint_pos_des = torch.where(violate, joint_pos, joint_pos_des_constrained)


        # frame_task: pink_ik.FrameTask = self.cfg.pink_ik_cfg.variable_input_tasks[0]
        # xyzw_quat = self.actions[0, 3:7].cpu().numpy()
        # # print(xyzw_quat)
        # frame_task.set_target(
        #     transform_target_to_world=pin.SE3(
        #         pin.Quaternion(
        #             float(xyzw_quat[3]), float(xyzw_quat[0]), float(xyzw_quat[1]), float(xyzw_quat[2])
        #         ),
        #         self.actions[0, 0:3].cpu().numpy()
        #     )
        # )
        # # print("Current joint pos: ", joint_pos[0].cpu().numpy())
        # joint_pos_des2 = self.pink_ik_controller.compute(curr_joint_pos=joint_pos[0].cpu().numpy(), dt=self.sim.get_physics_dt())
        # # print("Joint pos des: ", joint_pos_des2)




        # self.pink_ik_controller = pink_ik.PinkIKController(
        #     cfg=self.cfg.pink_ik_cfg,
        #     robot_cfg=self.robot.cfg,
        #     device=self.device,
        #     controlled_joint_indices=self.robot_entity_cfg.joint_ids,
        # )
        # self.pink_frame_task_cfg = pink_ik.FrameTaskCfg(
        #     frame="ur10e__wrist_3_link",
        #     position_cost=1.0,
        #     orientation_cost=1.0
        # )
        # self.pink_frame_task = pink_ik.FrameTask(self.pink_frame_task_cfg)



        # # --- secondary task: null-space pull toward joint mid-range ---
        # if "position" in self.ik_controller.cfg.command_type:
        #     jac = jacobians[:, 0:3]
        # else:
        #     jac = jacobians
        # jac_pinv = torch.linalg.pinv(jac)
        # identity = torch.eye(jac.shape[-1], device=jac.device).unsqueeze(0)
        # null_space_proj = identity - jac_pinv @ jac  # (N, num_joints, num_joints)

        # joint_limits = self.robot.data.soft_joint_pos_limits[:, self.robot_entity_cfg.joint_ids]  # (N, num_joints, 2)
        # joint_mid = (joint_limits[..., 0] + joint_limits[..., 1]) / 2.0
        # k_secondary = 0.4  # tune this
        # delta_joint_pos_secondary = -k_secondary * (joint_pos - joint_mid)
        # delta_joint_pos_null = (null_space_proj @ delta_joint_pos_secondary.unsqueeze(-1)).squeeze(-1)

        # joint_pos_des = joint_pos_des + delta_joint_pos_null