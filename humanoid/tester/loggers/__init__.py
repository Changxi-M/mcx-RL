import os
from humanoid import LEGGED_GYM_ROOT_DIR

# 简化的模块导入函数
def get_module_path_from_files_in_dir(root_dir, current_dir, prefix=""):
    """简化的模块路径获取函数"""
    import_modules = {}
    if os.path.exists(current_dir):
        for file in os.listdir(current_dir):
            if file.endswith('.py') and not file.startswith('__') and file.startswith(prefix):
                module_name = file[:-3]
                module_path = f"humanoid.tester.loggers.{module_name}"
                import_modules[module_name] = module_path
    return import_modules

file_directory = os.path.dirname(os.path.abspath(__file__))
import_modules = get_module_path_from_files_in_dir(
    LEGGED_GYM_ROOT_DIR, file_directory, "logger_type"
)
for module_path in import_modules.values():
    exec(f"from {module_path} import *")
