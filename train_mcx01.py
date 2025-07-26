#!/usr/bin/env python3
"""
MCX01 机器人训练脚本
用法: python train_mcx01.py
"""
import os
import sys
import subprocess
import argparse

sys.path.insert(0, os.path.abspath("."))

def main():
    parser = argparse.ArgumentParser(description="Humanoid 机器人训练/回放脚本（只需选择模式和是否断点续训）")
    parser.add_argument('--mode', choices=['train', 'play'], default='train', help='选择模式: train 或 play')
    parser.add_argument('--resume', action='store_true', default=False, help='从最近的 checkpoint 恢复训练')
    parser.add_argument('--use_tester', action='store_true', default=False, help='使用 tester 模式')
    args = parser.parse_args()

    print(f"🚀 选择模式: {args.mode}")
    print("是否断点续训: {}".format('是' if args.resume else '否'))
    print("=" * 50)
    
    # 默认参数
    task = 'mcxRobot_sym'
    experiment_name = 'mcxRobot_ppo_sym'
    run_name = 'v1'
    load_run = None  # 可设为 None 或指定字符串
    checkpoint = None  # 可设为 None 或指定整数
    headless = True
    num_envs = 4096
    seed = 16283
    max_iterations = 3001

    workspace_dir = os.getcwd()
    os.chdir(workspace_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{workspace_dir}"

    if args.mode == 'train':
        train_cmd = [
            "python",
            "humanoid/scripts/train.py",
            f"--task={task}",
            f"--experiment_name={experiment_name}",
            f"--run_name={run_name}",
            f"--num_envs={num_envs}",
            f"--seed={seed}",
            f"--max_iterations={max_iterations}",
            "--headless",
        ]
        if args.resume:
            train_cmd.append("--resume")
        if load_run:
            train_cmd.append(f"--load_run={load_run}")
        if checkpoint is not None:
            train_cmd.append(f"--checkpoint={checkpoint}")
        print(f"执行训练命令: {' '.join(train_cmd)}")
    else:
        train_cmd = [
            "python",
            "humanoid/scripts/play.py",
            f"--task={task}",
            f"--experiment_name={experiment_name}",
            f"--run_name={run_name}",
            
            # "--headless",
        ]
        if args.use_tester:
            train_cmd.append("--use_tester")
        print(f"执行回放命令: {' '.join(train_cmd)}")
        # sys.exit(0)
    print("=" * 50)
    try:
        result = subprocess.run(train_cmd, check=True, capture_output=False, env=env)
        print("✅ 运行完成！")
    except subprocess.CalledProcessError as e:
        print(f"❌ 运行失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  被用户中断")
        sys.exit(0)

if __name__ == "__main__":
    main()


#tensorboard --logdir /home/k205-2/engineai_rl_workspace/logs/mcx01_rough_ppo/default/mcx01_training_run