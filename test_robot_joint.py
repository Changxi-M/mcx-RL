#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import time  # 用于控制动画速度
import pybullet as p  # PyBullet主模块
import pybullet_data
sys.path.append("/opt/openrobots/lib/python3.8/site-packages")
import pinocchio
from pinocchio.robot_wrapper import RobotWrapper
import matplotlib.pyplot as plt

# 初始化PyBullet可视化环境
physicsClient = p.connect(p.GUI)  # 连接图形界面
p.setGravity(0, 0, -9.81)  # 设置重力
p.setTimeStep(1/1000)  # 设置仿真步长
# 定义初始位姿（位置和四元数）
robotStartPos = [0, 0, 0.95]  # 初始位置
robotStartOrientation = pinocchio.Quaternion(1, 0, 0, 0)  # 初始姿态

# # ---------------------- 加载地面 ----------------------
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 设置搜索路径
p.loadURDF("plane.urdf")  # 加载地面

## ---------------------- 动态解析URDF和网格路径 ----------------------
# 无论Python文件在哪运行，直接指定URDF的绝对路径
# urdf_abs_path = "/home/k205-2/humanoid-gym/resources/robots/assembled/urdf/assembled.urdf"
urdf_abs_path = "/home/k205-2/humanoid-gym/resources/robots/mcx_test1/urdf/assembled.urdf"
dof_of_robot = 8
# urdf_abs_path = "/home/k205-2/humanoid-gym/resources/robots/hubot/urdf/hubot.urdf"
# dof_of_robot = 12

# 通过URDF路径推导meshes目录的绝对路径
urdf_dir = os.path.dirname(urdf_abs_path)  # URDF所在目录：.../urdf
meshes_abs_dir = os.path.normpath(os.path.join(urdf_dir, "../meshes"))  # 上一级目录的meshes

# 配置包路径映射：将package://assembled映射到meshes所在的父目录
package_dirs = [
    os.path.dirname(meshes_abs_dir),  # 对应resources/robots/assembled
    meshes_abs_dir                    # 备用直接映射meshes目录
]

# ---------------------- 路径验证 ----------------------
print("\n[路径诊断]")
print("URDF绝对路径:", urdf_abs_path)
print("推导的网格目录:", meshes_abs_dir)
print("包映射路径:", package_dirs)

# 示例验证L_Link4.STL路径
expected_mesh = os.path.join(meshes_abs_dir, "L_Link4.STL")
print("STL文件路径:", expected_mesh)
if not os.path.exists(expected_mesh):
    raise FileNotFoundError(f"网格文件不存在: {expected_mesh}")
else:
    print("网格文件验证成功")

# ---------------------- 在PyBullet中加载机器人模型 ----------------------
robotId = p.loadURDF(
    urdf_abs_path,
    basePosition=robotStartPos,
    baseOrientation=robotStartOrientation.coeffs().tolist(),  # [x,y,z,w]格式
    useFixedBase=True,  # 允许自由浮动基座
    flags=p.URDF_USE_INERTIA_FROM_FILE
)

print("\n[PyBullet 关节序号、名称及类型]")
for i in range(p.getNumJoints(robotId)):
    joint_info = p.getJointInfo(robotId, i)
    joint_name = joint_info[1].decode('utf-8')
    joint_type = joint_info[2]
    joint_type_str = {
        p.JOINT_REVOLUTE: "REVOLUTE",
        p.JOINT_PRISMATIC: "PRISMATIC",
        p.JOINT_SPHERICAL: "SPHERICAL",
        p.JOINT_PLANAR: "PLANAR",
        p.JOINT_FIXED: "FIXED"
    }.get(joint_type, str(joint_type))
    print(f"关节序号: {i}, 关节名称: {joint_name}, 类型: {joint_type_str}")

# ---------------------- 使用Pinocchio加载模型（修复路径问题）-----------------------
try:
    # 临时切换工作目录到URDF所在目录（确保相对路径../meshes有效）
    original_cwd = os.getcwd()
    os.chdir(os.path.dirname(urdf_abs_path))
    
    # 加载URDF（此时相对路径以URDF目录为基准）
    rrobot = RobotWrapper.BuildFromURDF(
        os.path.basename(urdf_abs_path),  # 直接使用文件名
        package_dirs=package_dirs,        # 映射到推导的目录
        root_joint=pinocchio.JointModelFreeFlyer()
    )
    rmodel = rrobot.model
    print("rmodel.nq:", rmodel.nq)
    print("rmodel.nv:", rmodel.nv)
    print("关节名:", rmodel.names)
except Exception as e:
    print(f"\n[致命错误] URDF加载失败: {str(e)}")
    print("排错建议：")
    print("1. 确认URDF中mesh路径为：<mesh filename=../meshes/...\"/>")
    
    sys.exit(1)

# 禁用默认关节控制（允许直接设置关节位置）
p.setJointMotorControlArray(
    robotId,
    jointIndices=range(p.getNumJoints(robotId)),
    controlMode=p.VELOCITY_CONTROL,
    forces=[0]*p.getNumJoints(robotId))
    

# ---------------------- 辅助函数：将Pinocchio配置转换为PyBullet控制 ----------------------
def apply_pinocchio_configuration(robotId, q):
    """将Pinocchio的q向量应用到PyBullet的机器人模型"""
    # 设置基座位姿（前7个元素：位置+四元数）
    p.resetBasePositionAndOrientation(
        robotId,
        posObj=q[:3].tolist(),  # 前3个元素是位置
        ornObj=q[3:7].tolist()  # Pinocchio使用(x,y,z,w)格式，PyBullet需要(w,x,y,z)
    )
    
    # 设置关节位置（剩余元素）
    joint_indices = []  # 需要排除固定关节
    for i in range(p.getNumJoints(robotId)):
        joint_info = p.getJointInfo(robotId, i)
        if joint_info[2] != p.JOINT_FIXED:  # 仅处理非固定关节
            joint_indices.append(i)
    
    # 确保关节数量匹配（根据URDF结构调整）
    assert len(joint_indices) == rmodel.nq - 7, "关节数量不匹配！"
    
    for idx, joint_idx in enumerate(joint_indices):
        p.resetJointState(
            robotId,
            joint_idx,
            q[7 + idx]  # 从q的第7个元素开始是关节位置
        )

# ---------------------- 初始配置 ----------------------
q0 = pinocchio.utils.zero(rmodel.nq)
q0[:3] = robotStartPos  # 初始位置
q0[3:7] = robotStartOrientation.coeffs()  # 初始姿态（四元数）


# q0[6] = 1  # 四元数w分量（自由浮动基座方向）
# q0[2] = 0.85 # 初始高度（根据需要调整）
apply_pinocchio_configuration(robotId, q0)

# ---------------------- 单关节测试循环 ----------------------
# print("显示各关节单独运动...")
# for i in range(rmodel.nq - 7):
#     q_test = pinocchio.utils.zero(rmodel.nq)
#     q_test[6] = 1  # 保持基座方向
#     q_test[2] = 0   # z坐标
#     q_test[7 + i] = 1  # 激活当前关节
    
#     apply_pinocchio_configuration(robotId, q_test)
#     p.stepSimulation()  # 更新物理模拟
#     time.sleep(2)     # 暂停观察

# ---------------------- 主步行动画循环 ----------------------
print("开始步行动画...")
left_foot_heights = []
right_foot_heights = []
time_list = []
# default_joint_angles = {  # = target angles [rad] when action = 0.0
#             'R_joint1': 0.,
#             'R_joint2': 0.,
#             'R_joint3': 0.3,
#             'R_joint4': -0.7,
#             'R_joint5': -0.4,
#             'R_joint6': 0.,
#             'L_joint1': 0.,
#             'L_joint2': 0.,
#             'L_joint3': -0.3,
#             'L_joint4': 0.7,
#             'L_joint5': 0.4,
#             'L_joint6': 0.,
            
#         }
default_joint_angles = {  # = target angles [rad] when action = 0.0
            'R_joint1': 0.,
            'R_joint2': -0.38,
            'R_joint3': 0.95,
            'R_joint4': 0.47,
            'L_joint1': 0.,
            'L_joint2': -0.38,
            'L_joint3': 0.95,
            'L_joint4': 0.47,
        }
try:
    # 获取PyBullet中左右脚link的index
    left_foot_index = None
    right_foot_index = None
    for i in range(p.getNumJoints(robotId)):
        name = p.getJointInfo(robotId, i)[12].decode()
        if name == "L_Link4":
            left_foot_index = i
        if name == "R_Link4":
            right_foot_index = i
    assert left_foot_index is not None and right_foot_index is not None, "找不到足端link"

    gait_period = 0.5   # 步态周期（秒）
    for i in range(20000):
        # 生成正弦波形关节参考位置（保持原逻辑）
        t = i * 0.001
        phase = (t % gait_period) / gait_period  # 归一化到[0,1)
        # print(f"phase = {phase}")
        # phase = i * 0.01
        sin_pos = np.sin(2 * np.pi * phase)
        sin_pos_l = sin_pos.copy()
        sin_pos_r = sin_pos.copy()

        ref_dof_pos = np.zeros(dof_of_robot)  # 假设有8个驱动关节
        scale_1 = 0.25
        scale_2 = 2 * scale_1
        
        
        # 左腿摆动相位处理
        # if sin_pos_l > 0:
        #     sin_pos_l = 0
        # 将正值设为0（对应PyTorch的 sin_pos_l[sin_pos_l > 0] = 0）
        sin_pos_l = np.where(sin_pos_l > 0, 0, sin_pos_l)
        # 将绝对值小于0.1的值设为0（对应PyTorch的 sin_pos_l[torch.abs(sin_pos_l) < 0.1] = 0）
        sin_pos_l = np.where(np.abs(sin_pos_l) < 0.1, 0, sin_pos_l)
        # ref_dof_pos[6] = default_joint_angles["L_joint1"]
        # ref_dof_pos[7] = default_joint_angles["L_joint2"]
        # ref_dof_pos[8] = sin_pos_l * scale_1 + default_joint_angles["L_joint3"]
        # ref_dof_pos[9] = -sin_pos_l * scale_2 + default_joint_angles["L_joint4"]
        # ref_dof_pos[10] = -sin_pos_l * scale_1 +  default_joint_angles["L_joint5"]
        # ref_dof_pos[11] = default_joint_angles["L_joint6"]
        ref_dof_pos[0] = default_joint_angles["L_joint1"]
        # ref_dof_pos[1] = default_joint_angles["L_joint2"]
        # ref_dof_pos[2] = default_joint_angles["L_joint3"]
        # ref_dof_pos[3] = default_joint_angles["L_joint4"]
        ref_dof_pos[1] = sin_pos_l * scale_1 + default_joint_angles["L_joint2"]
        ref_dof_pos[2] = -sin_pos_l * scale_2 + default_joint_angles["L_joint3"]
        ref_dof_pos[3] = -sin_pos_l * scale_1 + default_joint_angles["L_joint4"]
        
        # 右腿摆动相位处理
        
        # if sin_pos_r < 0:
        #     sin_pos_r = 0
        # 将负值设为0（对应PyTorch的 sin_pos_r[sin_pos_r < 0] = 0）
        sin_pos_r = np.where(sin_pos_r < 0, 0, sin_pos_r)
        # 将绝对值小于0.1的值设为0（对应PyTorch的 sin_pos_r[torch.abs(sin_pos_r) < 0.1] = 0）
        sin_pos_r = np.where(np.abs(sin_pos_r) < 0.1, 0, sin_pos_r)
        # ref_dof_pos[0] = default_joint_angles["R_joint1"]
        # ref_dof_pos[1] = default_joint_angles["R_joint2"]
        # ref_dof_pos[2] = sin_pos_r * scale_1 + default_joint_angles["R_joint3"]
        # ref_dof_pos[3] = -sin_pos_r * scale_2 + default_joint_angles["R_joint4"]
        # ref_dof_pos[4] = -sin_pos_r * scale_1 +  default_joint_angles["R_joint5"]
        # ref_dof_pos[5] = default_joint_angles["R_joint6"]
        ref_dof_pos[4] = default_joint_angles["R_joint1"]
        # ref_dof_pos[5] = default_joint_angles["R_joint2"]
        # ref_dof_pos[6] = default_joint_angles["R_joint3"]
        # ref_dof_pos[7] = default_joint_angles["R_joint4"]
        ref_dof_pos[5] = -sin_pos_r * scale_1 + default_joint_angles["R_joint2"]
        ref_dof_pos[6] = sin_pos_r * scale_2 + default_joint_angles["R_joint3"]
        ref_dof_pos[7] = sin_pos_r * scale_1 + default_joint_angles["R_joint4"]

        # 双支撑相位处理
        # if np.abs(sin_pos) < 0.1:
        #     ref_dof_pos[:] = 0
        
        # 构建完整配置向量
        q_motion = pinocchio.utils.zero(rmodel.nq)
        q_motion[6] = 1  # 保持基座方向
        q_motion[2] = 0   # z坐标
        q_motion[7:7+dof_of_robot] = ref_dof_pos  # 填入关节位置
        
        # 应用到PyBullet
        apply_pinocchio_configuration(robotId, q_motion)
    
        # 推进模拟并控制速度
        p.stepSimulation()
        time.sleep(1/240.)  # 与实际时间步长同步

        # 记录时间
        time_list.append(i * 1/240.)

        # 获取左右脚足端高度
        left_foot_state = p.getLinkState(robotId, left_foot_index)
        right_foot_state = p.getLinkState(robotId, right_foot_index)
        left_foot_heights.append(left_foot_state[0][2])
        right_foot_heights.append(right_foot_state[0][2])

        # 获取PyBullet中所有link的名字和index
        # for i in range(p.getNumJoints(robotId)):
        #     print(i, p.getJointInfo(robotId, i)[12].decode())
except KeyboardInterrupt:
    print("动画中断，退出循环。")
finally:
    if len(time_list) > 0 and len(left_foot_heights) > 0 and len(right_foot_heights) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(time_list, left_foot_heights, label="Left Foot Height", color='red')
        plt.plot(time_list, right_foot_heights, label="Right Foot Height", color='blue')
        plt.xlabel("Time (s)")
        plt.ylabel("Foot Height (m)")
        plt.title("Foot Height Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("没有足够的数据用于绘图！")

    print("time_list 长度:", len(time_list))
    print("left_foot_heights 长度:", len(left_foot_heights))
    print("right_foot_heights 长度:", len(right_foot_heights))

    # 断开连接
    p.disconnect()