# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
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

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import cv2
import numpy as np
from isaacgym import gymapi
from isaacgym.torch_utils import *  # 确保gymtorch在这里导入
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import *
from humanoid.utils import  get_args, export_policy_as_jit, task_registry, Logger

import torch
from tqdm import tqdm
from datetime import datetime
import json
import pickle
import csv
from humanoid.tester.tester import Tester
from isaacgym import gymtorch
from humanoid.utils.video_utils import VideoRecorder

def play(args):
    # # 是否在patch中心附近随机初始化机器人位置
    RANDOMIZE_INIT_POS = True  # 需要随机时改为True
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.sim.max_gpu_contact_pairs = 2**10
    # env_cfg.terrain.mesh_type = 'plane'  # 注释掉这行，使用训练时的地形配置
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    # override some parameters for testing
    env_cfg.env.num_envs = env_cfg.terrain.num_rows * env_cfg.terrain.num_cols
    env_cfg.terrain.curriculum = False     
    env_cfg.terrain.max_init_terrain_level = 5
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.push_robots = True 
    env_cfg.domain_rand.joint_angle_noise = 0.
    env_cfg.noise.curriculum = False
    env_cfg.noise.noise_level = 0.5
    env_cfg.env.episode_length_s = 60


    # 添加相机配置
    class camera:
        width = 1920
        height = 1080
        position = [2.0, 0.0, 1.0]  # 相机位置
        lookat = [0.0, 0.0, 0.0]    # 相机目标点
        fov = 60                    # 视场角
    env_cfg.camera = camera
    
    if RANDOMIZE_INIT_POS:
        # 随机生成每个机器人的x, y初始位置
        x_range = (-1, 1)
        y_range = (-1, 1)
        x_pos = np.random.uniform(x_range[0], x_range[1])
        y_pos = np.random.uniform(y_range[0], y_range[1])
        print(f"已随机分布机器人的初始x,y位置({x_pos},{y_pos})")
    else:
        x_pos = 0
        y_pos = 0
    z_pos = env_cfg.init_state.pos[2]
    env_cfg.init_state.pos = [x_pos, y_pos, z_pos]
    # env_cfg.init_state.default_joint_angles = {
    #     'R_joint1': 0.,
    #     'R_joint2': -0.35,
    #     'R_joint3': 0.95,
    #     'R_joint4': 0.25,
    #     'L_joint1': 0.,
    #     'L_joint2': -0.35,
    #     'L_joint3': 0.95,
    #     'L_joint4': 0.25,
    # }
    # env_cfg.init_state.default_joint_angles = train_cfg.init_state.default_joint_angles

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.seed = 123145
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

    # 创建视频录制器
    video_recorder = None
    if RENDER:
        video_recorder = VideoRecorder(env, train_cfg, args, video_type="2x2")

    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    # 创建日志目录
    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', f"{train_cfg.runner.experiment_name}_play")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(log_dir, f"{timestamp}_{args.run_name}")
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    # 根据是否使用Tester模式选择不同的测试策略
    if hasattr(args, 'use_tester') and args.use_tester:
        # 使用Tester模式
        print("使用Tester模式进行分段测试...")
        
        # 创建Tester实例
        tester_config_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'humanoid', 'tester', 'tester_config.yaml')
        test_length = 400  # 每个测试器的测试长度
        
        tester = Tester(
            env,
            test_length,
            env.dt,
            run_dir,
            tester_config_path,
            record_video=False,  # 设置为False，因为我们使用Isaac Gym原生视频录制
            extra_args={"robot_index": 0},
        )
        
        # ----------- 新增：由用户指定robot_ids，绑定摄像头并2x2拼接视频 -----------
        # 用户手动指定需要录制的4个机器人id
        robot_ids = [1, 2, 3, 4]  # TODO: 用户请在此处填写你想录制的4个机器人id
        print(f"2x2拼接视频选用的机器人id: {robot_ids}")
        if RENDER:
            video_recorder.setup_cameras(robot_ids)
        
        # 设置迭代范围
        iteration_range = tqdm(range(tester.num_testers * test_length))
        
        # 初始化环境
        tester.set_env(0)
        
        # 主测试循环前，添加推力计时器和持续时间
        push_timer = torch.zeros(env_cfg.env.num_envs, device=env.device)
        push_duration = 1  # 推动持续时间（秒）
        # 主测试循环
        for iter in iteration_range:
            actions = policy(obs.detach())
            
            # 执行环境步骤
            obs, critic_obs, rews, dones, infos = env.step(actions.detach())
            
            # 推力计时器与清零机制
            push_timer += env.dt
            for robot_idx in range(env_cfg.env.num_envs):
                if push_timer[robot_idx] > push_duration:
                    env.rand_push_force[robot_idx, :] = 0.0
                    env.rand_push_torque[robot_idx, :] = 0.0
                    push_timer[robot_idx] = 0.0
            
            # 处理视频渲染
            if RENDER:
                env.gym.fetch_results(env.sim, True)
                env.gym.step_graphics(env.sim)
                env.gym.render_all_camera_sensors(env.sim)
                
                # 准备每个机器人的文字信息
                text_info_list = []
                current_tester_name = tester.tester_names[iter // test_length]
                for rid in robot_ids:
                    text_info = {
                        f"Tester: {current_tester_name}": (0, 255, 0),
                        f"Step: {iter % test_length}/{test_length}": (0, 165, 255),
                        f"Target Speed: ({env.commands[rid, 0].item():.2f}, {env.commands[rid, 1].item():.2f}) m/s": (0, 255, 0),
                        f"Actual Speed: ({env.base_lin_vel[rid, 0].item():.2f}, {env.base_lin_vel[rid, 1].item():.2f}) m/s": (0, 165, 255),
                        f"Push Vel: ({env.rand_push_force[rid, 0].item():.2f}, {env.rand_push_force[rid, 1].item():.2f}) m/s": (255, 0, 0),
                        f"Push Torque: ({env.rand_push_torque[rid, 0].item():.2f}, {env.rand_push_torque[rid, 1].item():.2f}, {env.rand_push_torque[rid, 2].item():.2f}) rad/s": (255, 0, 0)
                    }
                    text_info_list.append(text_info)
                
                # 记录2x2拼接视频帧
                video_recorder.record_2x2_frame(robot_ids, text_info_list)
            
            # 更新Tester状态
            if iter + 1 < tester.num_testers * test_length:
                tester.set_env(iter + 1)
            
            # 记录数据
            tester.step(iter, {"actions": actions})
            
            # 处理环境重置
            if infos["episode"]:
                # 重置机器人（条件：当机器人摔倒重置所有环境）
                # num_episodes = torch.sum(env.reset_buf).item()
                # if num_episodes > 0:
                #     env.reset()
                
                # 只重置摔倒了的那个机器人
                reset_indices = torch.where(env.reset_buf)[0]
                if len(reset_indices) > 0:
                    env.reset_idx(reset_indices)
                    obs = env.get_observations()
        # 关闭视频录制器
        if RENDER and video_recorder:
            video_recorder.close()
        print(f"\nTester测试完成！共测试了 {tester.num_testers} 个测试器")
        print(f"测试器列表: {tester.tester_names}")
        
    else:
        # 使用原有的多机器人同时测试模式
        print("使用多机器人同时测试模式...")
        
        # 创建多个logger实例，每个机器人一个
        loggers = [Logger(env.dt) for _ in range(env_cfg.env.num_envs)]
        robot_index = 0  # 用于绘图显示的机器人索引
        joint_index = 1  # 用于绘图的关节索引
        stop_state_log = 1200  # 记录步数
        
        # 为所有机器人生成速度值
        num_robots = env_cfg.env.num_envs
        lin_vel_x = env_cfg.commands.ranges.lin_vel_x  # 从env_cfg中获取速度范围
        command_speeds = np.linspace(lin_vel_x[0], lin_vel_x[1], num_robots)
        print(f"为{num_robots}个机器人生成速度值: {command_speeds}")
        print(f"x方向速度范围: {lin_vel_x[0]} 到 {lin_vel_x[1]} m/s")
        
        # 用户手动指定需要录制的4个机器人id
        robot_ids = [1, int(num_robots/4), int(2*num_robots/3), -2]  # TODO: 用户请在此处填写你想录制的4个机器人id
        print(f"2x2拼接视频选用的机器人id: {robot_ids}")
        print(f"对应的目标速度: {[command_speeds[rid] for rid in robot_ids]} m/s")
        
        # 保存机器人配置信息
        config_info = {
            'num_robots': env_cfg.env.num_envs,
            'command_speeds': command_speeds.tolist(),
            'robot_positions': [(i * 2.0, 0.0, 0.96) for i in range(env_cfg.env.num_envs)],
            'timestamp': timestamp,
            'selected_robot_ids': robot_ids,
            'selected_speeds': [command_speeds[rid] for rid in robot_ids]
        }
        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            json.dump(config_info, f, indent=4)

        # 创建CSV文件记录数据
        csv_files = {}
        for robot_idx in range(env_cfg.env.num_envs):
            csv_path = os.path.join(run_dir, f'robot_{robot_idx}_data.csv')
            csv_files[robot_idx] = open(csv_path, 'w', newline='')
            writer = csv.writer(csv_files[robot_idx])
            # 写入表头
            writer.writerow([
                'time_step',
                'dof_pos_target',
                'dof_pos',
                'dof_vel',
                'dof_torque',
                'command_x',
                'command_y',
                'command_yaw',
                'base_vel_x',
                'base_vel_y',
                'base_vel_z',
                'base_vel_yaw',
                'contact_force_z',
                'push_vel_x',
                'push_vel_y',
                'push_torque_x',
                'push_torque_y',
                'push_torque_z',
                'left_foot_pos_x',
                'left_foot_pos_z',
                'right_foot_pos_x',
                'right_foot_pos_z'
            ])

        # 添加推动计时器
        push_timer = torch.zeros(env_cfg.env.num_envs, device='cuda:0')
        push_duration = 1  # 推动持续时间（秒）

        # ----------- 新增：创建2x2拼接视频相关配置 -----------
        if RENDER:
            video_recorder.setup_cameras(robot_ids)
        # ----------- 新增结束 -----------

        for i in tqdm(range(stop_state_log)):
            actions = policy(obs.detach())
            
            if FIX_COMMAND:
                # 为每个机器人设置不同的命令速度
                for robot_idx in range(env_cfg.env.num_envs):
                    env.commands[robot_idx, 0] = command_speeds[robot_idx]  # x方向速度
                    env.commands[robot_idx, 1] = 0.0                        # y方向速度
                    env.commands[robot_idx, 2] = 0.                         # yaw角速度
                    env.commands[robot_idx, 3] = 0.                         # heading

            obs, critic_obs, rews, dones, infos = env.step(actions.detach())

            # 更新推动计时器并重置过期的推动值
            push_timer += env.dt
            for robot_idx in range(env_cfg.env.num_envs):
                if push_timer[robot_idx] > push_duration:
                    env.rand_push_force[robot_idx, :] = 0.0
                    env.rand_push_torque[robot_idx, :] = 0.0
                    push_timer[robot_idx] = 0.0

            if RENDER:
                env.gym.fetch_results(env.sim, True)
                env.gym.step_graphics(env.sim)
                env.gym.render_all_camera_sensors(env.sim)
                
                # 准备每个机器人的文字信息
                text_info_list = []
                for rid in robot_ids:
                    text_info = {
                        f"Target Speed: ({env.commands[rid, 0].item():.2f}, {env.commands[rid, 1].item():.2f}) m/s": (0, 255, 0),
                        f"Actual Speed: ({env.base_lin_vel[rid, 0].item():.2f}, {env.base_lin_vel[rid, 1].item():.2f}) m/s": (0, 165, 255),
                        f"Push Vel: ({env.rand_push_force[rid, 0].item():.2f}, {env.rand_push_force[rid, 1].item():.2f}) m/s": (255, 0, 0),
                        f"Push Torque: ({env.rand_push_torque[rid, 0].item():.2f}, {env.rand_push_torque[rid, 1].item():.2f}, {env.rand_push_torque[rid, 2].item():.2f}) rad/s": (255, 0, 0)
                    }
                    text_info_list.append(text_info)
                
                # 记录2x2拼接视频帧
                video_recorder.record_2x2_frame(robot_ids, text_info_list)

            # 记录所有机器人的状态到CSV
            for robot_idx in range(env_cfg.env.num_envs):
                writer = csv.writer(csv_files[robot_idx])
                writer.writerow([
                    float(i),
                    float(actions[robot_idx, joint_index].item() * env.cfg.control.action_scale),
                    float(env.dof_pos[robot_idx, joint_index].item()),
                    float(env.dof_vel[robot_idx, joint_index].item()),
                    float(env.torques[robot_idx, joint_index].item()),
                    float(env.commands[robot_idx, 0].item()),
                    float(env.commands[robot_idx, 1].item()),
                    float(env.commands[robot_idx, 2].item()),
                    float(env.base_lin_vel[robot_idx, 0].item()),
                    float(env.base_lin_vel[robot_idx, 1].item()),
                    float(env.base_lin_vel[robot_idx, 2].item()),
                    float(env.base_ang_vel[robot_idx, 2].item()),
                    float(env.contact_forces[robot_idx, env.feet_indices, 2].cpu().numpy().mean()),
                    float(env.rand_push_force[robot_idx, 0].item()),
                    float(env.rand_push_force[robot_idx, 1].item()),
                    float(env.rand_push_torque[robot_idx, 0].item()),
                    float(env.rand_push_torque[robot_idx, 1].item()),
                    float(env.rand_push_torque[robot_idx, 2].item()),
                    float(env.rigid_state[robot_idx, env.feet_indices[0], 0].item()),
                    float(env.rigid_state[robot_idx, env.feet_indices[0], 2].item()),
                    float(env.rigid_state[robot_idx, env.feet_indices[1], 0].item()),
                    float(env.rigid_state[robot_idx, env.feet_indices[1], 2].item())
                ])
                
                # 记录状态到logger
                loggers[robot_idx].log_states(
                    {
                        'dof_pos_target': actions[robot_idx, joint_index].item() * env.cfg.control.action_scale,
                        'dof_pos': env.dof_pos[robot_idx, joint_index].item(),
                        'dof_vel': env.dof_vel[robot_idx, joint_index].item(),
                        'dof_torque': env.torques[robot_idx, joint_index].item(),
                        'command_x': env.commands[robot_idx, 0].item(),
                        'command_y': env.commands[robot_idx, 1].item(),
                        'command_yaw': env.commands[robot_idx, 2].item(),
                        'base_vel_x': env.base_lin_vel[robot_idx, 0].item(),
                        'base_vel_y': env.base_lin_vel[robot_idx, 1].item(),
                        'base_vel_z': env.base_lin_vel[robot_idx, 2].item(),
                        'base_vel_yaw': env.base_ang_vel[robot_idx, 2].item(),
                        'contact_force_z': env.contact_forces[robot_idx, env.feet_indices, 2].cpu().numpy().mean(),
                        'push_vel_x': env.rand_push_force[robot_idx, 0].item(),
                        'push_vel_y': env.rand_push_force[robot_idx, 1].item(),
                        'push_torque_x': env.rand_push_torque[robot_idx, 0].item(),
                        'push_torque_y': env.rand_push_torque[robot_idx, 1].item(),
                        'push_torque_z': env.rand_push_torque[robot_idx, 2].item(),
                        'left_foot_pos_x': env.rigid_state[robot_idx, env.feet_indices[0], 0].item(),
                        'left_foot_pos_z': env.rigid_state[robot_idx, env.feet_indices[0], 2].item(),
                        'right_foot_pos_x': env.rigid_state[robot_idx, env.feet_indices[1], 0].item(),
                        'right_foot_pos_z': env.rigid_state[robot_idx, env.feet_indices[1], 2].item()
                    }
                )
            
            if infos["episode"]:
                # num_episodes = torch.sum(env.reset_buf).item()
                # if num_episodes > 0:
                #     for robot_idx in range(env_cfg.env.num_envs):
                #         if env.reset_buf[robot_idx]:
                #             loggers[robot_idx].log_rewards(infos["episode"], 1)
                reset_indices = torch.where(env.reset_buf)[0]
                if len(reset_indices) > 0:
                    for robot_idx in reset_indices:
                        loggers[robot_idx].log_rewards(infos["episode"], 1)
                    env.reset_idx(reset_indices)
                    obs = env.get_observations()

        # 关闭所有CSV文件
        for csv_file in csv_files.values():
            csv_file.close()

        # 显示选定机器人的图表
        print(f"\n显示机器人 {robot_index} 的状态图表...")
        try:
            loggers[robot_index].plot_states()
            print("状态图表绘制完成")
        except Exception as e:
            print(f"绘制状态图表时出错: {str(e)}")
        
        print(f"\n数据已保存到: {run_dir}")
        print("您可以稍后使用plot_play_data.py脚本查看数据图表")
    
        if RENDER and video_recorder:
            video_recorder.close()
        
        # input("按回车键关闭程序...")

if __name__ == '__main__':
    EXPORT_POLICY = True
    RENDER = True
    FIX_COMMAND = True
    
    args = get_args()
    
    # 添加use_tester参数（如果不存在的话）
    if not hasattr(args, 'use_tester'):
        args.use_tester = False  # 默认不使用Tester模式
    
    play(args)
