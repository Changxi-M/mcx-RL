import torch

def convert_to_visible_commands(commands, mins=[0.8, 0.2, 0.5]):
    """
    保证commands的每个分量绝对值不小于mins，且保持原有符号。
    commands: shape [N, 3] 或 [N, >=3]
    mins: 每个分量的最小绝对值
    """
    for i, threshold in enumerate(mins):
        sign = torch.where(
            torch.sign(commands[:, i]) == 0, 1, torch.sign(commands[:, i])
        )
        commands[:, i] = sign * torch.max(
            torch.abs(commands[:, i]), torch.ones_like(commands[:, i]) * threshold
        ) 