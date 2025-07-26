import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import json
from datetime import datetime
from humanoid import LEGGED_GYM_ROOT_DIR
import matplotlib as mpl

# 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

def get_latest_log_dir():
    """获取最新的日志目录"""
    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'mcxRobot_ppo_sym_play')
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"未找到目录: {log_dir}，请确保logs/mcxRobot_ppo_sym_play目录存在")
    
    # 获取所有日志目录（排除plots目录）
    log_dirs = [d for d in os.listdir(log_dir) 
                if os.path.isdir(os.path.join(log_dir, d)) 
                and not d.startswith('.')]  # 排除隐藏目录
    
    if not log_dirs:
        raise FileNotFoundError(f"在{log_dir}中未找到日志目录，请确保已经运行过play.py并生成了数据文件")
    
    # 按修改时间排序，获取最新的目录
    latest_dir = max(log_dirs, key=lambda d: os.path.getmtime(os.path.join(log_dir, d)))
    print(f"使用最新的日志目录: {latest_dir}")
    return os.path.join(log_dir, latest_dir)

def plot_robot_data(data_dir=None, robot_idx=0):
    """绘制机器人数据图表"""
    if data_dir is None:
        data_dir = get_latest_log_dir()
    
    # 读取配置信息
    with open(os.path.join(data_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    
    # 读取CSV数据
    csv_path = os.path.join(data_dir, f'robot_{robot_idx}_data.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到数据文件: {csv_path}")
    
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({k: float(v) for k, v in row.items()})
    
    # 创建图表
    fig = plt.figure(figsize=(15, 12))
    
    # 1. 关节位置和速度
    ax1 = fig.add_subplot(321)
    time = [d['time_step'] * 0.01 for d in data]  # 控制频率为100Hz，每步0.01秒
    ax1.plot(time, [d['dof_pos_target'] for d in data], 'r--', label='Target Position')
    ax1.plot(time, [d['dof_pos'] for d in data], 'b-', label='Actual Position')
    ax1.set_title('Joint Position')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (rad)')
    ax1.grid(True)
    ax1.legend()
    
    ax2 = fig.add_subplot(322)
    ax2.plot(time, [d['dof_vel'] for d in data], 'g-', label='Joint Velocity')
    ax2.set_title('Joint Velocity')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (rad/s)')
    ax2.grid(True)
    ax2.legend()
    
    # 2. 基座速度
    ax3 = fig.add_subplot(323)
    ax3.plot(time, [d['command_x'] for d in data], 'r--', label='Command Velocity')
    ax3.plot(time, [d['base_vel_x'] for d in data], 'b-', label='Actual Velocity')
    ax3.set_title('Base X Velocity')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.grid(True)
    ax3.legend()
    
    ax4 = fig.add_subplot(324)
    ax4.plot(time, [d['base_vel_y'] for d in data], 'b-', label='Y Velocity')
    ax4.plot(time, [d['base_vel_z'] for d in data], 'g-', label='Z Velocity')
    ax4.set_title('Base Y/Z Velocity')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Velocity (m/s)')
    ax4.grid(True)
    ax4.legend()
    
    # 3. 足端轨迹
    ax5 = fig.add_subplot(325)
    ax5.plot([d['left_foot_pos_x'] for d in data], 
             [d['left_foot_pos_z'] for d in data], 
             'r-', label='Left Foot')
    ax5.plot([d['right_foot_pos_x'] for d in data], 
             [d['right_foot_pos_z'] for d in data], 
             'b-', label='Right Foot')
    ax5.set_title('Foot Trajectories (x-z plane)')
    ax5.set_xlabel('X Position (m)')
    ax5.set_ylabel('Z Position (m)')
    ax5.grid(True)
    ax5.legend()
    ax5.axis('equal')
    
    # 4. 足端高度随时间变化
    ax6 = fig.add_subplot(326)
    ax6.plot(time, [d['left_foot_pos_z'] for d in data], 'r-', label='Left Foot')
    ax6.plot(time, [d['right_foot_pos_z'] for d in data], 'b-', label='Right Foot')
    ax6.set_title('Foot Height Over Time')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Height (m)')
    ax6.grid(True)
    ax6.legend()
    
    plt.tight_layout()
    
    # 保存图表
    save_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'plots')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    timestamp = datetime.now().strftime('%b%d_%H-%M-%S')
    save_path = os.path.join(save_dir, f'robot_{robot_idx}_data_{timestamp}.png')
    plt.savefig(save_path)
    print(f"图表已保存到: {save_path}")
    
    plt.show()

def main():
    """主函数"""
    try:
        # 在这里修改参数
        data_dir = None  # 设置为None则自动使用最新的日志目录
        # data_dir = "/home/k205-2/humanoid-gym/logs/mcxRobot_ppo_sym_play/Feb20_10-30-00_run1"  # 或者指定具体路径
        robot_idx = 15  # 修改为您想要查看的机器人序号（0, 1, 2, ...）
        
        plot_robot_data(data_dir=data_dir, robot_idx=robot_idx)
    except Exception as e:
        print(f"错误: {str(e)}")
        print("请确保已经运行过play.py并生成了数据文件")

if __name__ == '__main__':
    main() 