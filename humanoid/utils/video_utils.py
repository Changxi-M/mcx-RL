import os
import cv2
import numpy as np
from isaacgym import gymapi
from datetime import datetime
from humanoid import LEGGED_GYM_ROOT_DIR  # 直接从humanoid包导入

class VideoRecorder:
    def __init__(self, env, train_cfg, args, video_type="2x2"):
        """
        初始化视频录制器
        :param env: Isaac Gym环境
        :param train_cfg: 训练配置
        :param args: 命令行参数
        :param video_type: 视频类型 ("2x2" 或 "single")
        """
        self.env = env
        self.train_cfg = train_cfg
        self.args = args
        self.video_type = video_type
        
        # 设置相机属性
        self.camera_properties = gymapi.CameraProperties()
        self.camera_properties.width = 1920
        self.camera_properties.height = 1080
        self.height = self.camera_properties.height
        self.width = self.camera_properties.width
        
        # 设置相机位置和旋转
        self.camera_offset = gymapi.Vec3(1.5, -1.5, 1.5)
        self.camera_rotation = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(-0.3, 0.2, 1), 
            np.deg2rad(135)
        )
        
        # 初始化相机句柄列表
        self.camera_handles = []
        
        # 创建视频目录
        self.video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')  # 使用导入的LEGGED_GYM_ROOT_DIR
        self.experiment_dir = os.path.join(self.video_dir, train_cfg.runner.experiment_name)
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 设置视频写入器
        self.setup_video_writer()
        
    def setup_video_writer(self):
        """设置视频写入器"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if self.video_type == "2x2":
            self.video_path = os.path.join(
                self.experiment_dir, 
                f"{timestamp}_2x2_{self.args.run_name}.mp4"
            )
            self.video_writer = cv2.VideoWriter(
                self.video_path, 
                cv2.VideoWriter_fourcc(*"mp4v"), 
                50.0, 
                (self.width*2, self.height*2)
            )
        else:
            self.video_path = os.path.join(
                self.experiment_dir, 
                f"{timestamp}_single_{self.args.run_name}.mp4"
            )
            self.video_writer = cv2.VideoWriter(
                self.video_path, 
                cv2.VideoWriter_fourcc(*"mp4v"), 
                50.0, 
                (self.width, self.height)
            )
        print(f"视频将保存到: {self.video_path}")
        
    def setup_cameras(self, robot_ids):
        """
        为指定的机器人设置相机
        :param robot_ids: 机器人ID列表
        """
        self.camera_handles = []
        for rid in robot_ids:
            try:
                h = self.env.gym.create_camera_sensor(self.env.envs[rid], self.camera_properties)
                actor_handle = self.env.gym.get_actor_handle(self.env.envs[rid], 0)
                body_handle = self.env.gym.get_actor_rigid_body_handle(self.env.envs[rid], actor_handle, 0)
                self.env.gym.attach_camera_to_body(
                    h, self.env.envs[rid], body_handle,
                    gymapi.Transform(self.camera_offset, self.camera_rotation),
                    gymapi.FOLLOW_POSITION
                )
                self.camera_handles.append(h)
            except Exception as e:
                print(f"相机设置失败: robot_id={rid}, 错误: {e}")
                self.camera_handles.append(None)
                
    def capture_robot_frame(self, robot_id, camera_handle, text_info=None):
        """
        捕获单个机器人的画面并添加文字信息
        :param robot_id: 机器人ID
        :param camera_handle: 相机句柄
        :param text_info: 要显示的文字信息字典
        :return: 处理后的图像
        """
        if camera_handle is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
        img = self.env.gym.get_camera_image(
            self.env.sim, 
            self.env.envs[robot_id], 
            camera_handle, 
            gymapi.IMAGE_COLOR
        )
        img = np.reshape(img, (self.height, self.width, 4))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        if text_info:
            y0 = 40
            dy = 50
            for i, (text, color) in enumerate(text_info.items()):
                cv2.putText(
                    img, text, (50, y0 + i*dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    color, 2, cv2.LINE_AA
                )
        
        return img[..., :3]
        
    def record_2x2_frame(self, robot_ids, text_info_list):
        """
        记录2x2拼接视频帧
        :param robot_ids: 4个机器人的ID列表
        :param text_info_list: 每个机器人对应的文字信息列表
        """
        frames = []
        for rid, camera_handle, text_info in zip(robot_ids, self.camera_handles, text_info_list):
            frame = self.capture_robot_frame(rid, camera_handle, text_info)
            frames.append(frame)
            
        # 确保有4个画面
        while len(frames) < 4:
            frames.append(np.zeros((self.height, self.width, 3), dtype=np.uint8))
            
        # 2x2拼接
        row1 = np.hstack([frames[0], frames[1]])
        row2 = np.hstack([frames[2], frames[3]])
        big_frame = np.vstack([row1, row2])
        self.video_writer.write(big_frame)
        
    def record_single_frame(self, robot_id, camera_handle, text_info=None):
        """
        记录单个机器人的视频帧
        :param robot_id: 机器人ID
        :param camera_handle: 相机句柄
        :param text_info: 要显示的文字信息
        """
        frame = self.capture_robot_frame(robot_id, camera_handle, text_info)
        self.video_writer.write(frame)
        
    def close(self):
        """关闭视频写入器"""
        if hasattr(self, 'video_writer'):
            self.video_writer.release()
            print(f"视频已保存到: {self.video_path}") 