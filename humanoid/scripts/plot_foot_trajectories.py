import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json
import csv
from humanoid import LEGGED_GYM_ROOT_DIR

def load_ref_trajectories():
    """加载参考轨迹数据"""
    try:
        data = np.load('ref_trajectories.npz')
        return {
            'time_points': data['time_points'],
            'foot_trajectories': data['foot_trajectories']  # [left_x, left_z, right_x, right_z]
        }
    except FileNotFoundError:
        print("警告：未找到参考轨迹数据文件 'ref_trajectories.npz'")
        return None

def load_actual_trajectories(log_dir):
    """加载实际轨迹数据"""
    # 获取最新的日志目录
    log_dirs = [d for d in os.listdir(log_dir) if d.startswith('Feb')]
    if not log_dirs:
        raise FileNotFoundError("未找到日志目录")
    latest_dir = max(log_dirs)
    run_dir = os.path.join(log_dir, latest_dir)
    
    # 读取配置信息
    with open(os.path.join(run_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    
    # 读取CSV数据
    robot_data = {}
    for robot_idx in range(config['num_robots']):
        csv_path = os.path.join(run_dir, f'robot_{robot_idx}_data.csv')
        if not os.path.exists(csv_path):
            continue
            
        data = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    'time_step': float(row['time_step']),
                    'foot_pos_x': float(row['foot_pos_x']),
                    'foot_pos_z': float(row['foot_pos_z'])
                })
        robot_data[robot_idx] = data
    
    return robot_data, config

def plot_trajectories(ref_data, actual_data, config):
    """绘制足端轨迹对比图"""
    # 创建图表
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 绘制x-z平面的轨迹图
    ax1 = fig.add_subplot(221)
    if ref_data is not None:
        # 绘制参考轨迹
        ax1.plot(ref_data['foot_trajectories'][:, 0], ref_data['foot_trajectories'][:, 1], 
                'r--', label='参考左足', alpha=0.7)
        ax1.plot(ref_data['foot_trajectories'][:, 2], ref_data['foot_trajectories'][:, 3], 
                'g--', label='参考右足', alpha=0.7)
    
    # 绘制实际轨迹
    for robot_idx, data in actual_data.items():
        foot_x = [d['foot_pos_x'] for d in data]
        foot_z = [d['foot_pos_z'] for d in data]
        ax1.plot(foot_x, foot_z, '-', label=f'机器人{robot_idx}实际轨迹', alpha=0.5)
    
    ax1.set_title('足端轨迹 (x-z平面)')
    ax1.set_xlabel('X位置 (m)')
    ax1.set_ylabel('Z位置 (m)')
    ax1.grid(True)
    ax1.legend()
    ax1.axis('equal')
    
    # 2. 绘制高度随时间的变化
    ax2 = fig.add_subplot(222)
    if ref_data is not None:
        time = ref_data['time_points']
        ax2.plot(time, ref_data['foot_trajectories'][:, 1], 'r--', label='参考左足高度', alpha=0.7)
        ax2.plot(time, ref_data['foot_trajectories'][:, 3], 'g--', label='参考右足高度', alpha=0.7)
    
    for robot_idx, data in actual_data.items():
        time = [d['time_step'] * 0.01 for d in data]  # 控制频率为100Hz，每步0.01秒
        foot_z = [d['foot_pos_z'] for d in data]
        ax2.plot(time, foot_z, '-', label=f'机器人{robot_idx}实际高度', alpha=0.5)
    
    ax2.set_title('足端高度随时间变化')
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('高度 (m)')
    ax2.grid(True)
    ax2.legend()
    
    # 3. 绘制x位置随时间的变化
    ax3 = fig.add_subplot(223)
    if ref_data is not None:
        time = ref_data['time_points']
        ax3.plot(time, ref_data['foot_trajectories'][:, 0], 'r--', label='参考左足x位置', alpha=0.7)
        ax3.plot(time, ref_data['foot_trajectories'][:, 2], 'g--', label='参考右足x位置', alpha=0.7)
    
    for robot_idx, data in actual_data.items():
        time = [d['time_step'] * 0.01 for d in data]  # 控制频率为100Hz，每步0.01秒
        foot_x = [d['foot_pos_x'] for d in data]
        ax3.plot(time, foot_x, '-', label=f'机器人{robot_idx}实际x位置', alpha=0.5)
    
    ax3.set_title('足端x位置随时间变化')
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('X位置 (m)')
    ax3.grid(True)
    ax3.legend()
    
    # 4. 绘制速度命令和实际速度
    ax4 = fig.add_subplot(224)
    for robot_idx, data in actual_data.items():
        time = [d['time_step'] * 0.01 for d in data]  # 控制频率为100Hz，每步0.01秒
        command_x = [config['command_speeds'][robot_idx]] * len(time)
        actual_x = np.gradient([d['foot_pos_x'] for d in data], time)
        ax4.plot(time, command_x, '--', label=f'机器人{robot_idx}命令速度', alpha=0.7)
        ax4.plot(time, actual_x, '-', label=f'机器人{robot_idx}实际速度', alpha=0.5)
    
    ax4.set_title('足端速度对比')
    ax4.set_xlabel('时间 (s)')
    ax4.set_ylabel('速度 (m/s)')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    
    # 保存图表
    save_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'foot_trajectories')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    timestamp = datetime.now().strftime('%b%d_%H-%M-%S')
    save_path = os.path.join(save_dir, f'foot_trajectories_{timestamp}.png')
    plt.savefig(save_path)
    print(f"图表已保存到: {save_path}")
    
    plt.show()

def main():
    # 加载参考轨迹数据
    ref_data = load_ref_trajectories()
    
    # 加载实际轨迹数据
    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs')
    actual_data, config = load_actual_trajectories(log_dir)
    
    # 绘制轨迹对比图
    plot_trajectories(ref_data, actual_data, config)

if __name__ == '__main__':
    main()