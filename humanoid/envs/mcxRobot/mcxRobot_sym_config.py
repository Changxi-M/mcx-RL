# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class mcxRobotCfg_sym(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 35
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 61  # 增加4维用于步态曲线信息
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 8
        num_envs = 4096
        episode_length_s = 24     # episode length in seconds
        use_ref_actions = False   # speed up training by using reference actions

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/mcx_test1/urdf/assembled.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/assembled/urdf/assembled.urdf'

        name = "mcxRobot"
        foot_name = "4"
        knee_name = "3"
        # feet_names = ["L_link4", "R_link4"]
        # knee_names = ["L_link3", "R_link3"]

        terminate_after_contacts_on = ['base_link']   # 与环境接触终止
        penalize_contacts_on = ["base_link", "2"]  # 与环境接触惩罚
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        # mesh_type = 'curiculum'
        mesh_type = 'trimesh'
        # mesh_type = 'heightfield'
        # mesh_type = 'plane'
        curriculum = True
        max_curriculum_level = 10.0
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down(平面、障碍物，均匀，上坡，下坡，楼梯上楼，楼梯下楼)
        terrain_proportions = [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]
        restitution = 0.

    class noise:
        add_noise = True
        noise_level = 0.8    # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.96]

        # default_joint_angles = {  # = target angles [rad] when action = 0.0
        #     'left_leg_roll_joint': 0.,
        #     'left_leg_yaw_joint': 0.,
        #     'left_leg_pitch_joint': 0.,
        #     'left_knee_joint': 0.,
        #     'left_ankle_pitch_joint': 0.,
        #     'left_ankle_roll_joint': 0.,
        #     'right_leg_roll_joint': 0.,
        #     'right_leg_yaw_joint': 0.,
        #     'right_leg_pitch_joint': 0.,
        #     'right_knee_joint': 0.,
        #     'right_ankle_pitch_joint': 0.,
        #     'right_ankle_roll_joint': 0.,
        # }
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'R_joint1': 0.,
            'R_joint2': -0.4,
            'R_joint3': 0.9,
            'R_joint4': 0.5,
            'L_joint1': 0.,
            'L_joint2': -0.4,
            'L_joint3': 0.9,
            'L_joint4': 0.5,
        }
        # default_joint_angles = { # target angles when action = 0.0
        #     "joint2": -0.3, 
        #     "joint3": 0.66,
        #     "joint4": 0.36}


    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        # stiffness = {'leg_roll': 200.0, 'leg_pitch': 350.0, 'leg_yaw': 200.0,
        #              'knee': 350.0, 'ankle': 15}
        # damping = {'leg_roll': 10, 'leg_pitch': 10, 'leg_yaw':
        #            10, 'knee': 10, 'ankle': 10}
        stiffness = {'R_joint1': 100.0, 'R_joint2': 220.0, 'R_joint3': 250.0, 'R_joint4': 30.0,
                     'L_joint1': 100.0, 'L_joint2': 220.0, 'L_joint3': 250.0, 'L_joint4': 30.0}
        damping = {'R_joint1': 10., 'R_joint2': 10., 'R_joint3': 10., 'R_joint4': 10.,
                   'L_joint1': 10., 'L_joint2': 10., 'L_joint3': 10., 'L_joint4': 10.}

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_base_mass = True
        added_base_mass_range = [-5., 5.]
        randomize_link_mass = True
        multiplied_link_mass_range = [0.8, 1.5]
        randomize_base_com = True
        added_base_com_range = [-0.05, 0.05]
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.4
        # dynamic randomization
        action_delay = 0.5
        action_noise = 0.02

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.3, 0.6]   # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3] # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 0.91
        min_dist = 0.2
        max_dist = 0.8
        # target_joint_pos_scale1 = 0.7     # 增加关节位置容差
        # target_joint_pos_scale2 = 0.836     # 增加关节位置容差
        target_joint_pos_scale1 = 0.26     # 增加关节位置容差
        target_joint_pos_scale2 = 2 * target_joint_pos_scale1     # 增加关节位置容差
        target_feet_height = 0.10       # 增加目标抬腿高度，从0.15增加到0.20
        cycle_time = 0.64               # 步态周期
        only_positive_rewards = True
        tracking_sigma = 5            # 跟踪精度要求
        max_contact_force = 700        # 接触力限制
        feet_orientation = 1.0         # 脚底法向奖励权重

        class scales:
            # reference motion tracking
            joint_pos = 1.6             # 关节位置跟踪权重
            feet_clearance = 2.0        # 降低抬腿高度权重
            feet_contact_number = 1.2   # 增加接触次数权重
            feet_swing_height = -1.0
            # gait
            feet_air_time = 1.0        # 增加空中时间权重，从10.0增加到12.0
            foot_slip = -0.05          # 增加滑移惩罚
            feet_distance = 0.6        # 增加步距权重
            knee_distance = 0.6
            # contact
            feet_contact_forces = -0.01 # 增加接触力惩罚
            # vel tracking
            tracking_lin_vel = 1.2     # 降低速度跟踪权重
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5
            low_speed = 0.2
            track_vel_hard = 0.5
            stand_still = -2
            # base pos
            default_joint_pos = 0.5   # 默认关节位置权重
            orientation = 1.0          # 增加姿态权重
            base_height = 0.2           # 增加高度权重
            base_acc = 0.2            # 加速度权重
            feet_orientation = 0.1      # 增加脚底法向奖励权重
            # energy
            action_smoothness = -0.002     # 动作平滑度惩罚
            torques = -1e-5            # 增大力矩惩罚
            dof_vel = -5e-4             # 关节速度惩罚
            dof_acc = -1e-7             # 关节加速度惩罚
            collision = -1            # 增大碰撞惩罚
            

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 18.
        clip_actions = 18.


class mcxRobotCfgPPO_sym(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner_SYM'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001           # 增加熵系数以鼓励探索
        learning_rate = 1e-5           # 略微增加学习率
        num_learning_epochs = 2        # 增加学习轮数 
        gamma = 0.994                  # 增加折扣因子
        lam = 0.9                     # 增加GAE参数
        num_mini_batches = 4
        sym_loss = True
        sym_coef = 1.0               # 增加对称性损失权重
        obs_permutation = [-0.0001, -1, 2, -3, -4,\
                           -9, -10, -11, -12, -5, -6, -7, -8,\
                            -17, -18, -19, -20, -13, -14, -15, -16,\
                            -25, -26, -27, -28, -21, -22, -23, -24,\
                            -29, 30, -31, -32, 33, -34]
        act_permutation = [-4, -5, -6, -7, -0.0001, -1, -2, -3]
        frame_stack = 15

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO_SYM'
        num_steps_per_env = 60         # 增加每环境步数
        max_iterations = 3001         # 保持迭代次数不变

        # logging
        save_interval = 100  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'mcxRobot_ppo_sym'
        run_name = ''
        # log_root=="default"
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
