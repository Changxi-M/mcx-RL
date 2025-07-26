import math
import time
import numpy as np
import mujoco, mujoco.viewer
import pybullet as p
import pybullet_data
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import mcxRobotCfg_sym
import torch

from pynput import keyboard

class Polynomialfitting():
    '''
    用于初始化时，读取关节初始和目标位置，进行插值
    '''
    def __init__(self, adjust_time):
        # 定义一些变量和常量
        self.max_adjust_time = adjust_time  # 调姿时间
        self.parameters_curve = [0,0,0,0,0,0,0,0,0,0,0,0]

    def beginPostureAdjust(self,init_states, des_states):
        # 定义一个函数，用于开始进行调姿，计算拟合系数，初始化时间
        first_variable_init, second_variable_init, third_variable_init = init_states[0], init_states[1], init_states[2]
        first_variable_des, second_variable_des, third_variable_des = des_states[0], des_states[1], des_states[2]

        A5_curveone = first_variable_init
        A4_curveone = 0 # 速度
        A3_curveone = 0
        A0_curveone = (-A3_curveone * pow(self.max_adjust_time,2) - 3 * A4_curveone * self.max_adjust_time -
                        6 * A5_curveone + 6 * first_variable_des) / pow(self.max_adjust_time, 5)
        A1_curveone = (3 * A3_curveone * pow(self.max_adjust_time,2) + 8 * A4_curveone * self.max_adjust_time +
                        15 * A5_curveone - 15 * first_variable_des) / pow(self.max_adjust_time, 4)
        A2_curveone = (-3 * A3_curveone * pow(self.max_adjust_time,2) - 6 * A4_curveone * self.max_adjust_time -
                        10 * A5_curveone + 10 * first_variable_des) / pow(self.max_adjust_time, 3)
        
        A5_curvetwo = second_variable_init
        A4_curvetwo = 0 # 速度
        A3_curvetwo = 0
        A0_curvetwo = (-A3_curvetwo * pow(self.max_adjust_time,2) - 3 * A4_curvetwo * self.max_adjust_time -
                        6 * A5_curvetwo + 6 * second_variable_des) / pow(self.max_adjust_time, 5)
        A1_curvetwo = (3 * A3_curvetwo * pow(self.max_adjust_time,2) + 8 * A4_curvetwo * self.max_adjust_time +
                        15 * A5_curvetwo - 15 * second_variable_des) / pow(self.max_adjust_time, 4)
        A2_curvetwo = (-3 * A3_curvetwo * pow(self.max_adjust_time,2) - 6 * A4_curvetwo * self.max_adjust_time -
                        10 * A5_curvetwo + 10 * second_variable_des) / pow(self.max_adjust_time, 3)
        
        A5_curvethree = third_variable_init
        A4_curvethree = 0 # 速度
        A3_curvethree = 0
        A0_curvethree = (-A3_curvethree * pow(self.max_adjust_time,2) - 3 * A4_curvethree * self.max_adjust_time -
                        6 * A5_curvethree + 6 * third_variable_des) / pow(self.max_adjust_time, 5)
        A1_curvethree = (3 * A3_curvethree * pow(self.max_adjust_time,2) + 8 * A4_curvethree * self.max_adjust_time +
                        15 * A5_curvethree - 15 * third_variable_des) / pow(self.max_adjust_time, 4)
        A2_curvethree = (-3 * A3_curvethree * pow(self.max_adjust_time,2) - 6 * A4_curvethree * self.max_adjust_time -
                        10 * A5_curvethree + 10 * third_variable_des) / pow(self.max_adjust_time, 3)      
        
        para_one = [A0_curveone, A1_curveone, A2_curveone, A3_curveone, A4_curveone, A5_curveone]
        para_two = [A0_curvetwo, A1_curvetwo, A2_curvetwo, A3_curvetwo, A4_curvetwo, A5_curvetwo]
        para_three = [A0_curvethree, A1_curvethree, A2_curvethree, A3_curvethree, A4_curvethree, A5_curvethree]
        return  para_one, para_two , para_three


    def polyFitting(self,time_running_adjust, parameters_curve):
        # 定义一个函数，用于五次多项式拟合，控制电机转角
        A0_curveone, A1_curveone, A2_curveone = parameters_curve[0][0], parameters_curve[0][1], parameters_curve[0][2]
        A3_curveone, A4_curveone, A5_curveone = parameters_curve[0][3], parameters_curve[0][4], parameters_curve[0][5]
        A0_curvetwo, A1_curvetwo, A2_curvetwo = parameters_curve[1][0], parameters_curve[1][1], parameters_curve[1][2]
        A3_curvetwo, A4_curvetwo, A5_curvetwo = parameters_curve[1][3], parameters_curve[1][4], parameters_curve[1][5]
        A0_curvethree, A1_curvethree, A2_curvethree = parameters_curve[2][0], parameters_curve[2][1], parameters_curve[2][2]
        A3_curvethree, A4_curvethree, A5_curvethree = parameters_curve[2][3], parameters_curve[2][4], parameters_curve[2][5]
        curveone_pos = A0_curveone * pow(time_running_adjust, 5) + A1_curveone * pow(time_running_adjust,4) +\
                    A2_curveone * pow(time_running_adjust, 3) + A3_curveone * pow(time_running_adjust,2) + \
                    A4_curveone * time_running_adjust + A5_curveone
        curveone_vel = 5 * A0_curveone * pow(time_running_adjust, 4) + 4 * A1_curveone * pow(time_running_adjust,3) + \
                    3 * A2_curveone * pow(time_running_adjust, 2) + 2 * A3_curveone * time_running_adjust + A4_curveone
        curvetwo_pos = A0_curvetwo * pow(time_running_adjust, 5) + A1_curvetwo * pow(time_running_adjust,4) + \
                    A2_curvetwo * pow(time_running_adjust, 3) + A3_curvetwo * pow(time_running_adjust,2) + \
                    A4_curvetwo * time_running_adjust + A5_curvetwo
        curvetwo_vel = 5 * A0_curvetwo * pow(time_running_adjust, 4) + 4 * A1_curvetwo * pow(time_running_adjust, 3) + \
                    3 * A2_curvetwo * pow(time_running_adjust, 2) + 2 * A3_curvetwo * time_running_adjust + A4_curvetwo
        curvethree_pos = A0_curvethree * pow(time_running_adjust, 5) + A1_curvethree * pow(time_running_adjust,4) + \
                    A2_curvethree * pow(time_running_adjust, 3) + A3_curvethree * pow(time_running_adjust,2) + \
                    A4_curvethree * time_running_adjust + A5_curvethree
        curvethree_vel = 5 * A0_curvethree * pow(time_running_adjust, 4) + 4 * A1_curvethree * pow(time_running_adjust, 3) + \
                    3 * A2_curvethree * pow(time_running_adjust, 2) + 2 * A3_curvethree * time_running_adjust + A4_curvethree
        
        return [curveone_pos, curveone_vel], [curvetwo_pos, curvetwo_vel], [curvethree_pos, curvethree_vel]


    def Real_pos_vel(self, time_running_adjust, init_states, target_state):
        self.des_joint_pos = target_state
        # 计算实时的位置以及速度曲线
        if time_running_adjust == 0:
            self.parameters_curve = self.beginPostureAdjust(init_states, self.des_joint_pos)
            first_posdata,second_posdata,thired_posdata = init_states[0],init_states[1],init_states[2]
            first_veldata,second_veldata,thired_veldata = 0, 0, 0 

        elif time_running_adjust < self.max_adjust_time:
            # exp_eul_x, exp_gyr_x, exp_eul_y, exp_gyr_y
            real_data = self.polyFitting(time_running_adjust, self.parameters_curve)
            first_posdata,second_posdata,thired_posdata = real_data[0][0],real_data[1][0],real_data[2][0]
            first_veldata,second_veldata,thired_veldata = real_data[0][1],real_data[1][1],real_data[2][1]

        else:
            # 超过最大时间后执行最大值不变：max_adjust_time
            real_data = self.polyFitting(self.max_adjust_time, self.parameters_curve)
            first_posdata,second_posdata,thired_posdata = real_data[0][0],real_data[1][0],real_data[2][0]
            first_veldata,second_veldata,thired_veldata = real_data[0][1],real_data[1][1],real_data[2][1]

        return [first_posdata,first_veldata], [second_posdata,second_veldata], [thired_posdata,thired_veldata]


class Simulator:

    def __init__(self, cfg):
        self.cfg = cfg
        self.bottom = 3
        # 初始化时间
        self.Initial_to_squat_time = 1
        self.loadcount = 0
        # 读取键盘输入
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        # 双足位置复位
        self.left_reset = Polynomialfitting(adjust_time = self.Initial_to_squat_time)
        self.right_reset = Polynomialfitting(adjust_time = self.Initial_to_squat_time)
        # 添加新的变量用于速度控制
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_dyaw = 0.0
        self.current_vx = 0.0
        self.current_vy = 0.0
        self.current_dyaw = 0.0
        self.velocity_ramp_time = 10.0  # 速度渐变时间（秒）
        self.velocity_ramp_start_time = 0.0
        self.is_ramping_velocity = False
        # 添加落地稳定检测相关变量
        self.landing_start_time = 0.0
        self.is_landing = False
        self.landing_stable_time = 1.0  # 落地稳定等待时间（秒）
        self.landing_velocity_threshold = 0.1  # 落地稳定速度阈值
        self.landing_height_threshold = 0.05  # 落地稳定高度阈值
        self.last_height = 0.0
        self.last_velocity = np.zeros(3)

    class cmd:
        vx = 0.2    ##速度再小就会抽风，还没试别的前进速度
        vy = 0.0
        dyaw = 0.0

    def quaternion_to_euler_array(self, quat):
        # 确保四元数格式为 [x, y, z, w]
        x, y, z, w = quat

        # 滚转角 (x 轴旋转)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)

        # 俯仰角 (y 轴旋转)
        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch_y = np.arcsin(t2)

        # 偏航角 (z 轴旋转)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        # 返回以弧度表示的滚转角、俯仰角和偏航角
        return np.array([roll_x, pitch_y, yaw_z])

    def get_obs(self, data):
        '''从 mujoco 数据结构中提取观测信息
        '''
        q = data.qpos.astype(np.double)
        dq = data.qvel.astype(np.double)
        quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
        r = R.from_quat(quat)
        v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # 在基坐标系下关节速度
        omega = data.sensor('angular-velocity').data.astype(np.double)
        gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
        return (q, dq, quat, v, omega, gvec)
    
    def get_observation(self, robot):
        '''从 pybullet 数据结构中提取观测信息
        '''
        q = []
        dq = []
        # 遍历所有关节并获取角度和速度
        for joint_index in self.joint_id:
            joint_state = p.getJointState(robot, joint_index)
            q.append(joint_state[0]) # 关节角，单位：弧度
            dq.append(joint_state[1]) # 关节速度，单位：弧度/秒
        quat = p.getBasePositionAndOrientation(robot)[1] # 四元数
        q = np.array(q).astype(np.double)
        dq = np.array(dq).astype(np.double)
        quat = np.array(quat).astype(np.double)
        r = R.from_quat(quat)
        linear_velocity, omega = p.getBaseVelocity(robot)
        v = r.apply(linear_velocity, inverse=True).astype(np.double) # 在基坐标系下
        gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
        return (q, dq, quat, v, omega, gvec)

    def apply_joint_actions(self, joint_torque):

        self.applied_joint_torques = joint_torque

        zero_gains = len(joint_torque) * (0., )
        p.setJointMotorControlArray(
            self.robot, 
            self.joint_id, 
            p.TORQUE_CONTROL, 
            forces=joint_torque, 
            positionGains=zero_gains, 
            velocityGains=zero_gains
        )

    def pd_control(self, target_q, q, kp, target_dq, dq, kd):
        '''根据位置命令计算扭矩
        '''
        # print("target_q:", (target_dq - dq))
        # print("KP:", kd)
        # print("ans:", (target_q - q) * kp + (target_dq - dq) * kd)
        return (target_q - q) * kp + (target_dq - dq) * kd

    def check_landing_stability(self, current_height, current_velocity):
        """检查机器人是否已经稳定落地"""
        if not self.is_landing:
            return False
        
        # 检查高度变化
        height_change = abs(current_height - self.last_height)
        # 检查速度大小
        velocity_magnitude = np.linalg.norm(current_velocity)
        
        # 更新上一次的值
        self.last_height = current_height
        self.last_velocity = current_velocity
        
        # 如果高度变化和速度都小于阈值，认为已经稳定
        if height_change < self.landing_height_threshold and velocity_magnitude < self.landing_velocity_threshold:
            elapsed_time = time.time() - self.landing_start_time
            if elapsed_time >= self.landing_stable_time:
                return True
        return False

    def run_mujoco(self, policy):
        """
        使用提供的策略和配置运行 Mujoco 仿真。

        Args:
            policy: 用于控制仿真的策略。

        Returns:
            None
        """
        self.model = mujoco.MjModel.from_xml_path(self.cfg.sim_config.mujoco_model_path)
        self.model.opt.timestep = self.cfg.sim_config.dt
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)
        viewer = mujoco.viewer.launch_passive(self.model, self.data)

        #读取初始关节角位置,qpos的1-9不包含0，0是bacelink
        self.joint_pos_back_button = self.get_obs(self.data)[0][7:16]
        #读取机体位置
        self.bacelink_pos = self.get_obs(self.data)[0][0:7]

        target_q = np.zeros((self.cfg.env.num_actions), dtype=np.double)
        action = np.zeros((self.cfg.env.num_actions), dtype=np.double)

        hist_obs = deque()
        for _ in range(self.cfg.env.frame_stack):
            hist_obs.append(np.zeros([1, self.cfg.env.num_single_obs], dtype=np.double))

        count_lowlevel = 0
        default_angle = np.zeros((self.cfg.env.num_actions), dtype=np.double)
        default_angle[0] = self.cfg.init_state.default_joint_angles['L_joint1']
        default_angle[1] = self.cfg.init_state.default_joint_angles['L_joint2']
        default_angle[2] = self.cfg.init_state.default_joint_angles['L_joint3']
        default_angle[3] = self.cfg.init_state.default_joint_angles['L_joint4']
        default_angle[4] = self.cfg.init_state.default_joint_angles['R_joint1']
        default_angle[5] = self.cfg.init_state.default_joint_angles['R_joint2']
        default_angle[6] = self.cfg.init_state.default_joint_angles['R_joint3']
        default_angle[7] = self.cfg.init_state.default_joint_angles['R_joint4']
        # default_angle[0] = 0
        # default_angle[1] = -0.4
        # default_angle[2] = 0.9
        # default_angle[3] = default_angle[1] + default_angle[2]
        # default_angle[4] = 0
        # default_angle[5] = -0.4
        # default_angle[6] = 0.9
        # default_angle[7] = default_angle[5] + default_angle[6]

        self.listener.start()

        for _ in tqdm(range(int(self .cfg.sim_config.sim_duration / self.cfg.sim_config.dt)), desc="Simulating..."):

            # 获取观测
            q, dq, quat, v, omega, gvec = self.get_obs(self.data)
            q = q[-self.cfg.env.num_actions:]
            dq = dq[-self.cfg.env.num_actions:]

            # 检查落地稳定性
            if self.is_landing and self.bottom == 1:
                current_height = self.data.qpos[2]
                current_velocity = self.data.qvel[:3]
                if self.check_landing_stability(current_height, current_velocity):
                    print("机器人已稳定落地，加载策略")
                    self.bottom = 2
                    self.is_landing = False
                    # 设置目标速度为0
                    self.target_vx = 0.0
                    self.target_vy = 0.0
                    self.target_dyaw = 0.0
                    self.current_vx = 0.0
                    self.current_vy = 0.0
                    self.current_dyaw = 0.0

            # 每执行若干次基础循环，更新一次控制（高频到低频的映射）
            if count_lowlevel % self.cfg.sim_config.decimation == 0:
                # 更新速度命令
                if self.is_ramping_velocity:
                    current_time = time.time()
                    elapsed_time = current_time - self.velocity_ramp_start_time
                    if elapsed_time < self.velocity_ramp_time:
                        # 线性插值计算当前速度
                        alpha = elapsed_time / self.velocity_ramp_time
                        self.current_vx = self.target_vx * alpha
                        self.current_vy = self.target_vy * alpha
                        self.current_dyaw = self.target_dyaw * alpha
                    else:
                        # 达到目标速度
                        self.current_vx = self.target_vx
                        self.current_vy = self.target_vy
                        self.current_dyaw = self.target_dyaw
                        self.is_ramping_velocity = False

                obs = np.zeros([1, self.cfg.env.num_single_obs], dtype=np.float32)
                eu_ang = self.quaternion_to_euler_array(quat)
                eu_ang[eu_ang > math.pi] -= 2 * math.pi
                # 更新观测信息，包括正弦函数、控制命令、关节位置、速度等
                obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * self.cfg.sim_config.dt / 0.5)
                obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * self.cfg.sim_config.dt / 0.5)
                print(f"phase: {count_lowlevel * self.cfg.sim_config.dt}")
                # obs[0, 2] = self.current_vx * self.cfg.normalization.obs_scales.lin_vel
                # obs[0, 3] = self.current_vy * self.cfg.normalization.obs_scales.lin_vel
                # obs[0, 4] = self.current_dyaw * self.cfg.normalization.obs_scales.ang_vel
                obs[0, 2] = self.cmd.vx * self.cfg.normalization.obs_scales.lin_vel
                obs[0, 3] = self.cmd.vy * self.cfg.normalization.obs_scales.lin_vel
                obs[0, 4] = self.cmd.dyaw * self.cfg.normalization.obs_scales.ang_vel
                obs[0, 5:13] = (q - default_angle) * self.cfg.normalization.obs_scales.dof_pos
                # obs[0, 5:13] = q * self.cfg.normalization.obs_scales.dof_pos
                obs[0, 13:21] = dq * self.cfg.normalization.obs_scales.dof_vel
                obs[0, 21:29] = action
                obs[0, 29:32] = omega
                obs[0, 32:35] = eu_ang

                # 限制观测值的范围
                obs = np.clip(obs, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)

                hist_obs.append(obs)
                hist_obs.popleft()

                policy_input = np.zeros([1, self.cfg.env.num_observations], dtype=np.float32)
                for i in range(self.cfg.env.frame_stack):
                    policy_input[0, i * self.cfg.env.num_single_obs : (i + 1) * self.cfg.env.num_single_obs] = hist_obs[i][0, :]

                # 根据策略产生动作
                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
                action = np.clip(action, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)

                # 根据bottom值，加载初始化或者加载强化学习文件
                if self.bottom == 0:
                    # 固定bacelink
                    self.data.qpos[0:7] = self.bacelink_pos 
                    leftpos = self.left_reset.Real_pos_vel(time_running_adjust = (count_lowlevel - self.loadcount) * self.cfg.sim_config.dt,
                                                           init_states = self.joint_pos_back_button[0:3],
                                                           target_state = default_angle
                                                           )
                    rightpos = self.right_reset.Real_pos_vel(time_running_adjust = (count_lowlevel - self.loadcount) * self.cfg.sim_config.dt, 
                                                             init_states = self.joint_pos_back_button[4:7],
                                                             target_state = default_angle
                                                             )
                    # 根据前3个关节计算脚踝的关节角
                    left_theta4 = leftpos[1][0] + leftpos[2][0]
                    right_theta4 = rightpos[1][0] + rightpos[2][0]
                    target_q = [rightpos[0][0], rightpos[1][0], rightpos[2][0], right_theta4, leftpos[0][0], leftpos[1][0], leftpos[2][0], left_theta4]
                # 添加重力
                elif self.bottom == 1:
                    self.model.opt.gravity[2] = -9.81
                # 加载强化学习文件
                elif self.bottom == 2:                 
                    add_q = action * self.cfg.control.action_scale
                    target_q = add_q + default_angle
                    # print(f"target_q is {target_q}")
                else:
                    #加载机器人
                    self.model.opt.gravity[2] = 0
                    target_q = self.joint_pos_back_button
                    # 下一次加载的时间，注意降频
                    self.loadcount = count_lowlevel + self.cfg.sim_config.decimation

            target_dq = np.zeros((self.cfg.env.num_actions), dtype=np.double)
            # 生成PD控制信号
            if self.bottom == 1:
                tau = self.pd_control(target_q, q, 400, target_dq, dq, 10)
                tau = np.clip(tau, -self.cfg.robot_config.tau_limit, self.cfg.robot_config.tau_limit)
                self.data.ctrl = tau
            if self.bottom == 0 or self.bottom == 2: 
                tau = self.pd_control(target_q, q, self.cfg.robot_config.kps, target_dq, dq, self.cfg.robot_config.kds)
                tau = np.clip(tau, -self.cfg.robot_config.tau_limit, self.cfg.robot_config.tau_limit)
                self.data.ctrl = tau
                print(f"tau is {tau}")
                # self.data.ctrl = target_q
                # print(f"target_q is {target_q}")

            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            count_lowlevel += 1

        viewer.close()
        self.listener.stop()

    def run_pybullet(self, policy):
        """
        使用提供的策略和配置运行 Pybullet 仿真。

        Args:
            policy: 用于控制仿真的策略。

        Returns:
            None
        """
        self.model = None # 这里的 model 是 None，表示没有使用 MuJoCo 模型
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robot_add = self.cfg.sim_config.pybullet_model_path
        p.setTimeStep(self.cfg.sim_config.dt)

        # 加载模型
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF(self.robot_add, 
                                [0, 0, 0.96], 
                                p.getQuaternionFromEuler([0, 0, 0]), 
                                useFixedBase=False
                                )
        
        # 保存关节id,将所有关节type为p.JOINT_REVOLUTE的关节id保存到列表中
        self.joint_id = []
        for joint_index in range(p.getNumJoints(self.robot)):
            joint_info = p.getJointInfo(self.robot, joint_index)
            if joint_info[2] == p.JOINT_REVOLUTE:   
                self.joint_id.append(joint_index)
        # for joint_index in range(p.getNumJoints(self.robot)):
        #     joint_info = p.getJointInfo(self.robot, joint_index)  # 获取关节信息
        #     joint_name = joint_info[1].decode('utf-8')  # 获取关节名称并解码为字符串
        #     print(f"Joint {joint_index}: Name = {joint_name}")

        self.joint_pos_back_button = self.get_observation(self.robot)[0]

        # 目标关节角初始化
        target_q = np.zeros((self.cfg.env.num_actions), dtype=np.double)
        action = np.zeros((self.cfg.env.num_actions), dtype=np.double)

        hist_obs = deque()
        for _ in range(self.cfg.env.frame_stack):
            hist_obs.append(np.zeros([1, self.cfg.env.num_single_obs], dtype=np.double))

        count_lowlevel = 0
        default_angle = np.zeros((self.cfg.env.num_actions), dtype=np.double)
        # default_angle[0] = self.cfg.init_state.default_joint_angles['L_joint1']
        # default_angle[1] = self.cfg.init_state.default_joint_angles['L_joint2']
        # default_angle[2] = self.cfg.init_state.default_joint_angles['L_joint3']
        # default_angle[3] = self.cfg.init_state.default_joint_angles['L_joint4']
        # default_angle[4] = self.cfg.init_state.default_joint_angles['R_joint1']
        # default_angle[5] = self.cfg.init_state.default_joint_angles['R_joint2']
        # default_angle[6] = self.cfg.init_state.default_joint_angles['R_joint3']
        # default_angle[7] = self.cfg.init_state.default_joint_angles['R_joint4']
        
        default_angle[0] = 0
        default_angle[1] = -0.4
        default_angle[2] = 0.9
        default_angle[3] = default_angle[1] + default_angle[2]
        default_angle[4] = 0
        default_angle[5] = -0.4
        default_angle[6] = 0.9
        default_angle[7] = default_angle[5] + default_angle[6]

        self.listener.start()
        print(f'joint is {self.joint_id}')
        for _ in tqdm(range(int(self .cfg.sim_config.sim_duration / self.cfg.sim_config.dt)), desc="Simulating..."):

            # 获取观测
            q, dq, quat, v, omega, gvec = self.get_observation(self.robot)

            # 检查落地稳定性
            if self.is_landing and self.bottom == 1:
                current_height = p.getBasePositionAndOrientation(self.robot)[0][2]
                current_velocity = np.array(p.getBaseVelocity(self.robot)[0])
                if self.check_landing_stability(current_height, current_velocity):
                    print("机器人已稳定落地，加载策略")
                    self.bottom = 2
                    self.is_landing = False
                    # 设置目标速度为0
                    self.target_vx = self.cmd.vx
                    self.target_vy = self.cmd.vy
                    self.target_dyaw = self.cmd.dyaw
                    self.current_vx = 0.0
                    self.current_vy = 0.0
                    self.current_dyaw = 0.0

            # 每执行若干次基础循环，更新一次控制（高频到低频的映射）
            if count_lowlevel % self.cfg.sim_config.decimation == 0:
                # 更新速度命令
                if self.is_ramping_velocity:
                    current_time = time.time()
                    elapsed_time = current_time - self.velocity_ramp_start_time
                    if elapsed_time < self.velocity_ramp_time:
                        # 线性插值计算当前速度
                        alpha = elapsed_time / self.velocity_ramp_time
                        self.current_vx = self.target_vx * alpha
                        self.current_vy = self.target_vy * alpha
                        self.current_dyaw = self.target_dyaw * alpha
                    else:
                        # 达到目标速度
                        self.current_vx = self.target_vx
                        self.current_vy = self.target_vy
                        self.current_dyaw = self.target_dyaw
                        self.is_ramping_velocity = False

                obs = np.zeros([1, self.cfg.env.num_single_obs], dtype=np.float32)
                eu_ang = self.quaternion_to_euler_array(quat)
                eu_ang[eu_ang > math.pi] -= 2 * math.pi

                # 更新观测信息，包括正弦函数、控制命令、关节位置、速度等
                obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * self.cfg.sim_config.dt / 0.64)
                obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * self.cfg.sim_config.dt / 0.64)
                obs[0, 2] = self.current_vx * self.cfg.normalization.obs_scales.lin_vel
                obs[0, 3] = self.current_vy * self.cfg.normalization.obs_scales.lin_vel
                obs[0, 4] = self.current_dyaw * self.cfg.normalization.obs_scales.ang_vel
                # obs[0, 2] = self.cmd.vx * self.cfg.normalization.obs_scales.lin_vel
                # obs[0, 3] = self.cmd.vy * self.cfg.normalization.obs_scales.lin_vel
                # obs[0, 4] = self.cmd.dyaw * self.cfg.normalization.obs_scales.ang_vel
                obs[0, 5:13] = (q - default_angle) * self.cfg.normalization.obs_scales.dof_pos
                obs[0, 13:21] = dq * self.cfg.normalization.obs_scales.dof_vel
                obs[0, 21:29] = action
                obs[0, 29:32] = omega
                obs[0, 32:35] = eu_ang

                # 限制观测值的范围
                obs = np.clip(obs, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)

                hist_obs.append(obs)
                hist_obs.popleft()

                policy_input = np.zeros([1, self.cfg.env.num_observations], dtype=np.float32)
                for i in range(self.cfg.env.frame_stack):
                    policy_input[0, i * self.cfg.env.num_single_obs : (i + 1) * self.cfg.env.num_single_obs] = hist_obs[i][0, :]

                # 根据策略产生动作
                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
                action = np.clip(action, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)

              # 根据bottom值，加载初始化或者加载强化学习文件
                if self.bottom == 0:
                    leftpos = self.left_reset.Real_pos_vel(time_running_adjust = (count_lowlevel - self.loadcount) * self.cfg.sim_config.dt,
                                                           init_states = self.joint_pos_back_button[0:3],
                                                           target_state = default_angle
                                                           )
                    rightpos = self.right_reset.Real_pos_vel(time_running_adjust = (count_lowlevel - self.loadcount) * self.cfg.sim_config.dt, 
                                                             init_states = self.joint_pos_back_button[4:7],
                                                             target_state = default_angle
                                                             )
                    # 根据前3个关节计算脚踝的关节角
                    left_theta4 = leftpos[1][0] + leftpos[2][0]
                    right_theta4 = rightpos[1][0] + rightpos[2][0]
                    target_q = [rightpos[0][0], rightpos[1][0], rightpos[2][0], right_theta4, leftpos[0][0], leftpos[1][0], leftpos[2][0], left_theta4]
                # 添加重力
                elif self.bottom == 1:
                    p.setGravity(0, 0, -9.8)
                # 加载强化学习文件
                elif self.bottom == 2:                 
                    add_q = action * self.cfg.control.action_scale
                    target_q = add_q + default_angle
                else:
                    #加载机器人
                    p.setGravity(0, 0, 0)
                    target_q = self.joint_pos_back_button
                    # 下一次加载的时间，注意降频
                    self.loadcount = count_lowlevel + self.cfg.sim_config.decimation
                    
            target_dq = np.zeros((self.cfg.env.num_actions), dtype=np.double)

            for idx,joint_index in enumerate(self.joint_id):
                p.setJointMotorControl2(bodyUniqueId=self.robot,
                                        jointIndex=joint_index,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=target_q[idx],
                                        )

            p.stepSimulation()
            time.sleep(self.cfg.sim_config.dt)
            count_lowlevel += 1

        p.disconnect()
        self.listener.stop()


    def on_press(self, key):
        try:
            if key == keyboard.Key.space:
                if self.model is None:
                    p.setGravity(0, 0, 0)
                else:
                    self.model.opt.gravity[2] = 0
                self.bottom = 0
                print(f"初始关节角是 is {self.joint_pos_back_button}")
                print("初始化")
                # 重置速度相关变量
                self.target_vx = 0.0
                self.target_vy = 0.0
                self.target_dyaw = 0.0
                self.current_vx = 0.0
                self.current_vy = 0.0
                self.current_dyaw = 0.0
                self.is_ramping_velocity = False
                self.is_landing = False
            elif key == keyboard.Key.down:
                print("开始落地")
                if self.model is None:
                    p.setGravity(0, 0, -9.8)
                    self.joint_pos_back_button = self.get_observation(self.robot)[0]
                else:
                    self.model.opt.gravity[2] = -9.81
                    self.joint_pos_back_button = self.data.qpos[7:15]
                print(f'当前关节角是 is {self.joint_pos_back_button}')
                self.bottom = 1  # 先只加载重力
                # 开始落地检测
                self.is_landing = True
                self.landing_start_time = time.time()
                # 初始化上一次的高度和速度
                if self.model is None:
                    self.last_height = p.getBasePositionAndOrientation(self.robot)[0][2]
                    self.last_velocity = np.array(p.getBaseVelocity(self.robot)[0])
                else:
                    self.last_height = self.data.qpos[2]
                    self.last_velocity = self.data.qvel[:3]
            elif key == keyboard.Key.enter:
                if self.model is None:
                    p.setGravity(0, 0, -9.8)
                else:
                    self.model.opt.gravity[2] = -9.81
                self.bottom = 2
                print("开始速度渐变")
                # 设置目标速度为命令速度
                self.target_vx = self.cmd.vx
                self.target_vy = self.cmd.vy
                self.target_dyaw = self.cmd.dyaw
                # 开始速度渐变
                self.velocity_ramp_start_time = time.time()
                self.is_ramping_velocity = True
        except Exception as e:
            print(f"出错：{e}")

    def on_release(self, key):
        if key == keyboard.Key.esc:
            print("退出仿真")
            return False

if __name__ == '__main__':
    '''
    空格初始化
    ↓加载重力
    enter加载强化学习文件
    '''
    import argparse

    ######################### 选择仿真引擎， 0是mujoco， 1是pybullet #####################################################
    sim_engine = 0
    ###################################################################################################################

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=True, default='/home/k205-2/humanoid-gym/logs/mcxRobot_ppo_sym/exported/policies/policy_1.pt',
                        help='Run to load from.')
    # parser.add_argument('--load_model', type=str, default='/home/k205-2/humanoid-gym/logs/mcxRobot_ppo_sym/exported/policies/policy_1.pt',
    #                     help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', default='plane', help='terrain or plane')
    args = parser.parse_args()

    class Sim2simCfg(mcxRobotCfg_sym):

        class sim_config:
            if args.terrain:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/assembled/mjcf/assembled.xml'
                pybullet_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/assembled/urdf/assembled.urdf'         
            else:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/assembled/mjcf/assembled.xml'
                pybullet_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/assembled/urdf/assembled.urdf'
            sim_duration = 60.0  # 模拟持续时间
            dt = 0.001  # 仿真时间步长
            decimation = 10  # 控制降频

        class robot_config:
            kps = np.array([120, 400, 400, 100, 120, 400, 400, 100], dtype=np.double)  # 比例增益
            kds = np.array([6., 6., 6., 10., 6., 6., 6., 10.], dtype=np.double)  # 微分增益
            tau_limit = np.array([18., 50., 50., 18., 18., 50., 50., 18.], dtype=np.double)  # 力矩限制
            # tau_limit = 800. * np.ones(8, dtype=np.double)  # 力矩限制
            

    # 加载训练好的策略模型
    policy = torch.jit.load(args.load_model)

    # 创建模拟器实例并运行仿真
    simulator = Simulator(Sim2simCfg())
    if sim_engine == 0:
        simulator.run_mujoco(policy)
    if sim_engine == 1:
        simulator.run_pybullet(policy)