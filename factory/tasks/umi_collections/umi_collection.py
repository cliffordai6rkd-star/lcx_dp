from hardware.base.camera import CameraBase
from hardware.sensors.cameras.realsense_camera import RealsenseCamera
from hardware.sensors.cameras.opencv_camera import OpencvCamera
from hardware.base.utils import object_class_check
from factory.tasks.inferences_tasks.utils.display import display_images
from teleop.pika_tracker.pika_tracker import PikaTracker
from sshkeyboard import listen_keyboard, stop_listening
import numpy as np
from dataset.lerobot.data_process import EpisodeWriter
import os, threading, time
import glog as log

class UmiCollection:
    def __init__(self, config):
        self._camera_classes = {
            "realsense_camera": RealsenseCamera,
            "opencv_camera": OpencvCamera, 
        }

        self._pika_config = config["pika_config"]
        self._pika_mode = config["pika_mode"]
        self._camera_infos = config["cameras"]
        self._use_fisheye = config.get(f'use_fisheye', True)
        self._camera_dict = {}
        self._record_frequency = config.get('collection_frequency', 30)
        task_name = config["task_name"]
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self._output_path = os.path.join(cur_dir, "../../../dataset/data", task_name)
        self.data_recorder = None
        self._image_visualization = config.get("image_visualization", True)
        self._enable_recording = False
        self._collection_running = True
    
    def _keyboard_on_press(self, key):
        if key == "r":
            self._enable_recording = not self._enable_recording
            if self._enable_recording and self.data_recorder is None:
                os.makedirs(self._output_path, exist_ok=True)
                log.info(f"{'='*15}Build data recoreder at {self._output_path}{'='*15}")
                self.data_recorder = EpisodeWriter(task_dir=self._output_path, 
                                                    rerun_log=False)
            if self._enable_recording: # start record a new episode
                # Record the episode data
                if not self.data_recorder.create_episode():
                    log.warn(f'Episode write failed to create a episode for recording data!!!!')
                else:
                    log.info(f"{'='*15}Data recorder started to write the episode data!!!!{'='*15}")
            else: # finish the episode write
                self.data_recorder.save_episode()
                time.sleep(0.5)
                log.info(f"{'='*15}Data recorder stoped recording the episode data!!!!{'='*15}")
        if key == 'q':
            log.info(f'Quit the umi collection process!!!')
            if self._enable_recording:
                self.data_recorder.save_episode()
            stop_listening()
            self._pika.close()
            for camera_name, camera in self._camera_dict:
                cam_obj:CameraBase = camera["object"]
                cam_obj.close()
                log.info(f'Closed the camera {camera_name}')
            self._collection_running = False
    
    def create_umi_system(self):
        self._pika = PikaTracker(self._pika_config)
        if not self._pika.initialize():
            raise ValueError(f'Pika tracker with {self._pika_config} failed to initialize!!!')
        
        # cameras
        for camera_info in self._camera_infos:
            camera_type = camera_info["type"]
            camera_cfg = camera_info["config"][camera_type]
            camera_name = camera_info["name"]
            if not self._use_fisheye:
                if 'fisheye' in camera_name:
                    continue
                
            if not object_class_check(self._camera_classes, camera_type):
                raise ValueError(f'camera type: {camera_type} is not supported!!!')
            camera_object:CameraBase = self._camera_classes[camera_type](camera_cfg)
            if not camera_object.initialize():
                raise ValueError(f'Camera {camera_type} {camera_name} with cfg: {camera_cfg} failed to initialize')
            self._camera_dict[camera_name] = dict(object=camera_object)
    
        # keyboard
        listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                        kwargs={"on_press": self._keyboard_on_press, 
                                                "until": None, "sequential": False,}, 
                                        daemon=True)
        listen_keyboard_thread.start()
    
    def collect_data(self):
        log.info(f'Umi collection data thread started!!!!')
        
        start_time = time.perf_counter()
        target_pose_key = ["left", "right", "head"]
        while self._collection_running:
            # images
            cur_colors = {}
            for cam_name, camera in self._camera_dict:
                cam_obj:CameraBase = camera["object"]
                cam_data = cam_obj.capture_all_data()
                image = cam_data["image"]
                cur_colors[cam_name] = image
                
            # display
            display_images(cur_colors, "UMI collection images")
            
            # pika poses
            success, poses, tools = self._pika.parse_data_2_robot_target()
            if success:
                ee_states = {}
                ee_tools = {}
                
                for pose_key, pose in poses.items():
                    if pose_key not in target_pose_key:
                        continue
                    pose = np.zeros(6)
                    if not isinstance(pose, list):
                        ee_states[pose_key] = pose.tolist()
                    
                for tool_key, tool in tools.items():
                    if tool_key not in ee_states:
                        raise ValueError(f'tool {tool_key} not in poses {list(ee_states.keys())}')
                    # give the gripper normalized value to the ee_tools
                    ee_tools[pose_key] = [tool[0]]
                
                # data write
                self.data_recorder.add_item(colors=cur_colors, 
                        ee_states=ee_states, tools=ee_tools)
            
            used_time = time.perf_counter() - start_time
            if used_time < (1.0 / self._record_frequency):
                time.sleep((1.0 / self._record_frequency) - used_time)
            elif used_time > 1.3 / self._record_frequency:
                log.warn(f'collect data is slow, actual: {1.0/used_time}Hz, expected: {self._record_frequency}Hz')
            start_time = time.perf_counter()
    
    
if __name__ == "__main__":
    from factory.utils import parse_args
    from hardware.base.utils import dynamic_load_yaml
    arguments = {
        "config": {
            "short_cut": "-c",
            "symbol": "--config",
            "type": str,
            "default": "teleop/config/franka_3d_mouse.yaml",
            "help": "Path to the config file"
        },
        "task": {
            "short_cut": "-t",
            "symbol": "--task",
            "type": int,
            "default": -1,
            "help": "Task ID to use (skip interactive selection)"
        }
    }
    args = parse_args("umi data collection", arguments)
    
    config = dynamic_load_yaml(args.config)
    umi_collection = UmiCollection(config)
    log.info(f'Created umi collection system')
    umi_collection.create_umi_system()
    
    umi_collection.collect_data()
    log.info(f'Finished umi collection process')
    