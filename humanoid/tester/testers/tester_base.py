from abc import ABC
import os

# 简化的类名转换函数
def instance_name_to_class_name(instance_name):
    """将实例名转换为类名"""
    return ''.join(word.capitalize() for word in instance_name.split('_'))

# 导入 humanoid 的 loggers
from humanoid.tester.loggers import *


class TesterTypeBase(ABC):
    def __init__(self, name, loggers, env, time, test_dir, extra_args):
        self.name = name
        self.env = env
        if self.__class__.__name__ != "TesterBase":
            self.test_dir = os.path.join(test_dir, name)
        self.loggers = []
        for key, value in loggers.items():
            logger_class = eval(instance_name_to_class_name(value))
            self.loggers.append(
                logger_class(key, env, time, os.path.join(test_dir, name), extra_args)
            )

    def set_commands(self):
        return None

    def start_record_video(self):
        self.env.start_recording_video()

    def end_and_save_recording_video(self):
        self.env.end_and_save_recording_video(self.name + ".mp4")
