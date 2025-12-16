from hardware.base.camera import CameraBase
from hardware.sensors.cameras.realsense_camera import RealsenseCamera
from hardware.sensors.cameras.opencv_camera import OpencvCamera
from hardware.base.utils import object_class_check
from simulation.mujoco.mujoco_sim import MujocoSim
from factory.tasks.inferences_tasks.utils.display import display_images
from teleop.pika_tracker.pika_tracker import PikaTracker
from sshkeyboard import listen_keyboard, stop_listening
import numpy as np
from dataset.lerobot.data_process import EpisodeWriter
import os, threading, time
import glog as log
import yaml, cv2, copy

class UmiCollection:
    def __init__(self, config):
        self._camera_classes = {
            "realsense_camera": RealsenseCamera,
            "opencv_camera": OpencvCamera, 
        }
        
        self._use_mujoco = config.get(f'use_mujoco', False)
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
            for camera_name, camera in self._camera_dict.items():
                cam_obj:CameraBase = camera["object"]
                cam_obj.close()
                cv2.destroyAllWindows()
                log.info(f'Closed the camera {camera_name}')
            if self._use_mujoco:
                self._mujoco.close()
            self._collection_running = False
    
    def create_umi_system(self):
        if self._use_mujoco:
            mujoco_cfg = "simulation/config/mujoco_umi_cfg.yaml"
            cur_path = os.path.dirname(os.path.abspath(__file__))
            cfg_file = os.path.join(cur_path, "../../..", mujoco_cfg)
            with open(cfg_file, 'r') as stream:
                config = yaml.safe_load(stream)
            self._mujoco = MujocoSim(config["mujoco"])
        
        self._pika = PikaTracker(self._pika_config["pika_tracker"])
        if not self._pika.initialize():
            raise ValueError(f'Pika tracker with {self._pika_config["pika_tracker"]} failed to initialize!!!')
        
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
            log.info(f'Initializing {camera_type} {camera_name} {camera_info}')
            camera_object:CameraBase = self._camera_classes[camera_type](camera_cfg)
            if not camera_object.initialize():
                raise ValueError(f'Camera {camera_type} {camera_name} with cfg: {camera_cfg} failed to initialize')
            log.info(f'Successful created {camera_type} {camera_name}')
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
        target_pose_key = {"single":"targetR", "left": "targetL", 
                    "right": "targetR", "head": "targetH"}
        while self._collection_running:
            # images
            cur_colors = {}
            start = time.perf_counter()
            for cam_name, camera in self._camera_dict.items():
                cam_obj:CameraBase = camera["object"]
                cam_data = cam_obj.capture_all_data()
                if 'image' in cam_data:
                    image = cam_data["image"]
                    color_name = cam_name + '_color'
                    cur_colors[color_name] = {"data": image, "time_stamp": cam_data['time_stamp']}
                    # display_image_dicts[color_name] = image
                else:
                    raise ValueError(f'{cam_name} do not get color image data')
            img_read_time = time.perf_counter() - start
            # display
            start = time.perf_counter()
            display_images(cur_colors, "UMI collection images", attributes="data")
            img_display_time = time.perf_counter() - start
            img_total_time = img_display_time + img_read_time
            
            # pika poses
            success, poses, tools = self._pika.parse_data_2_robot_target(self._pika_mode)
            if success:
                ee_states = {}
                ee_tools = {}
                
                for pose_key, pose in poses.items():
                    if pose_key not in target_pose_key:
                        continue
                    if self._use_mujoco:
                        mocap = target_pose_key[pose_key]
                        visual_pose = copy.deepcopy(pose)
                        if self._pika_mode == "absolute_delta":
                            if 'left' in pose_key:
                                visual_pose[1] += 0.3
                                visual_pose[2] += 0.45
                            elif 'right' in pose_key:
                                visual_pose[1] -= 0.3
                                visual_pose[2] += 0.45
                            else: visual_pose[2] += 0.95
                        self._mujoco.set_target_mocap_pose(mocap, visual_pose)
                    if not isinstance(pose, list):
                        ee_states[pose_key] = dict(pose=pose.tolist(), time_stamp=time.perf_counter())
                    
                for tool_key, tool in tools.items():
                    if tool_key not in ee_states:
                        raise ValueError(f'tool {tool_key} not in poses {list(ee_states.keys())}')
                    # give the gripper normalized value to the ee_tools
                    ee_tools[tool_key] = dict(position=float(tool[0]), time_stamp=time.perf_counter())
                    # log.info(f'tool posi: {tool[0]}')
                    
                # data write
                if self._enable_recording:
                    self.data_recorder.add_item(colors=cur_colors, 
                            ee_states=ee_states, tools=ee_tools)
            
            used_time = time.perf_counter() - start_time
            start_time = time.perf_counter()
            if used_time < (1.0 / self._record_frequency):
                sleep_time = (1.0 / self._record_frequency) - used_time
                time.sleep(0.92*sleep_time)
                # log.info(f'used time: {used_time}, sleep_time: {sleep_time}')
            elif used_time > 1.3 / self._record_frequency:
                #  log.warn(f'collect data is slow, actual: {1.0/used_time:.2f}Hz, expected: {self._record_frequency:.2f}Hz)
                log.warn(f'collect data is slow, actual: {1.0/used_time:.2f}Hz, expected: {self._record_frequency:.2f}Hz, img read time: {1.0/img_read_time:.2f}HZ, img display time {1.0/img_display_time:.2f}Hz img read time: {1.0/img_read_time:.2f}HZ, img total time {1.0/img_total_time:.2f}Hz')
    
    
if __name__ == "__main__":
    from factory.utils import parse_args
    from hardware.base.utils import dynamic_load_yaml
    arguments = {
        "config": {
            "short_cut": "-c",
            "symbol": "--config",
            "type": str,
            "default": "factory/tasks/umi_collections/config/umi_collection_left_only_cfg.yaml",
            "help": "Path to the config file"
        },
    }
    args = parse_args("umi data collection", arguments)
    
    config = dynamic_load_yaml(args.config)
    umi_collection = UmiCollection(config)
    log.info(f'Created umi collection system')
    umi_collection.create_umi_system()
    
    umi_collection.collect_data()
    log.info(f'Finished umi collection process')
    