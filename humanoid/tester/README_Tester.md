# Humanoid Tester 功能使用说明

## 概述

Humanoid项目现在支持类似EngineAI的Tester系统，用于进行分段速度测试。这个系统允许你按顺序测试不同的运动模式，而不是同时测试多个机器人。

## 功能特点

### 1. 两种测试模式

- **多机器人同时测试模式**（默认）：同时测试多个机器人，每个机器人使用不同的速度
- **Tester分段测试模式**：按顺序测试不同的运动模式，每个模式运行固定步数

### 2. 支持的测试器类型

- `tester_normal_commands`: 正常命令测试
- `tester_zero_commands`: 零速度命令测试
- `tester_x_commands`: 仅X方向速度测试
- `tester_y_commands`: 仅Y方向速度测试
- `tester_yaw_commands`: 仅偏航角速度测试
- `tester_forward_commands`: 前进命令测试
- `tester_backward_commands`: 后退命令测试

## 使用方法

### 1. 使用Tester模式

```bash
python scripts/play.py --task XBotL_free --use_tester
```

### 2. 使用传统多机器人模式

```bash
python scripts/play.py --task XBotL_free
```

### 3. 自定义测试长度

```bash
python scripts/play.py --task XBotL_free --use_tester --test_length 400
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_tester` | flag | False | 启用Tester分段测试模式 |
| `--test_length` | int | 500 | 每个测试器的测试步数 |
| `--video` | flag | False | 录制视频 |
| `--headless` | flag | False | 无头模式（不显示GUI） |

## 测试流程

### Tester模式流程

1. **初始化**: 创建Tester实例，加载配置文件
2. **分段测试**: 按顺序运行每个测试器
   - 每个测试器运行 `test_length` 步
   - 自动切换测试器类型
   - 记录测试数据
3. **数据保存**: 将测试结果保存到日志目录

### 传统模式流程

1. **多机器人设置**: 创建多个机器人实例
2. **速度分配**: 为每个机器人分配不同的速度
3. **同时测试**: 所有机器人同时运行
4. **数据记录**: 记录每个机器人的状态数据

## 配置文件

Tester配置位于 `humanoid/tester/tester_config.yaml`：

```yaml
testers:
  tester_normal_commands:
    loggers:
      [ logger_type_base_vel,
        logger_type_position,
        logger_type_torque,
        logger_type_vel,
        logger_type_force ]
  tester_forward_commands:
    loggers:
      [ logger_type_base_vel,
        logger_type_position,
        logger_type_torque,
        logger_type_vel,
        logger_type_force ]
  # ... 其他测试器
```

## 输出文件

### Tester模式输出

- 日志目录: `logs/{experiment_name}_play/{timestamp}_{run_name}/`
- 数据文件: 每个测试器的数据保存在 `data/` 子目录中
- 视频文件: 如果启用视频录制，保存在 `videos/` 目录中

### 传统模式输出

- CSV文件: 每个机器人的详细数据
- 配置文件: `config.json` 包含测试配置信息
- 图表: 自动生成的状态图表

## 示例命令

### 基本Tester测试
```bash
python scripts/play.py --task XBotL_free --use_tester --run_name tester_test
```

### 带视频录制的Tester测试
```bash
python scripts/play.py --task XBotL_free --use_tester --video --run_name tester_video_test
```

### 自定义测试长度
```bash
python scripts/play.py --task XBotL_free --use_tester --test_length 300 --run_name short_test
```

### 无头模式测试
```bash
python scripts/play.py --task XBotL_free --use_tester --headless --run_name headless_test
```

## 故障排除

### 常见问题

1. **Tester创建失败**
   - 检查配置文件路径是否正确
   - 确保所有测试器类都已正确导入

2. **环境重置问题**
   - 在Tester模式下，环境重置由系统自动处理
   - 确保环境配置正确

3. **视频录制问题**
   - 确保启用了视频录制功能
   - 检查视频保存路径权限

### 调试模式

运行测试脚本验证Tester功能：
```bash
python scripts/test_tester.py
```

## 扩展功能

### 添加新的测试器

1. 在 `humanoid/tester/testers/` 目录下创建新的测试器类
2. 继承 `TesterTypeBase` 类
3. 实现 `set_commands()` 方法
4. 在配置文件中添加新测试器

### 自定义日志器

1. 在 `humanoid/tester/loggers/` 目录下创建新的日志器类
2. 继承 `LoggerBase` 类
3. 实现必要的数据记录方法
4. 在配置文件中引用新日志器

## 性能优化

- 使用 `--headless` 模式可以提高性能
- 调整 `--test_length` 参数以平衡测试覆盖率和运行时间
- 根据需要启用或禁用视频录制功能 