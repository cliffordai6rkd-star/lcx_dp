from teleop.base.teleoperation_base import TeleoperationDeviceBase
from pika.sense import Sense
from pika.tracker.vive_tracker import ViveTracker
from pika.tracker.vive_tracker import PoseData
from hardware.base.utils import convert_homo_2_7D_pose, negate_pose, transform_pose, transform_quat, pose_diff
import glog as log
import numpy as np
import time, threading, math
# from sshkeyboard import listen_keyboard, stop_listening
from pynput import keyboard
from typing import Dict, Tuple, Optional
import copy

class PikaTracker(TeleoperationDeviceBase):
    _tracker: dict[str, Sense]
    
    T_ROBOT_TRACKER = np.eye(4)
    # T_TRACKER_ROBOT = np.array([[0, 1, 0, 0],
    #                        [-1, 0, 0, 0],
    #                        [0, 0, 1, 0],
    #                        [ 0, 0, 0, 1]])
    T_TRACKER_ROBOT = np.array([[-1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, 1, 0],
                           [ 0, 0, 0, 1]])
    # T_TRACKER_ROBOT = np.eye(4)
    
    # init pose for absolute delta pose calculation, left, right, head
    INIT_TARGET_POSE = [[0,0,0,0,0,0,1], [0,0,0,0,0,0,1], [0,0,0,0,0,0,1]]
    
    def __init__(self, config):
        self._serial_port: dict = config.get("serial_ports")
        self._tracker = None
        self._sense = {}
        for key, port in self._serial_port.items():
            log.info(f'{key} port: {port}')
            if port is not None:
                sense = Sense(port=port)
                self._sense[key] = sense
            else: self._sense[key] = None
        self._rotation_enabled = config["rotation_enabled"]
        for key, rot_enable in self._rotation_enabled.items():
            if not rot_enable: continue
            assert len(rot_enable) == 3, f"{key} rotation enabled need to have three elemets indicating rpy but get {len(rot_enable) == 3}"

        # Vive Tracker configuration
        self._vive_config_path = config.get("vive_config_path", None)
        self._vive_lh_config = config.get("vive_lh_config", None)
        self._vive_args = config.get("vive_args", None)
        self._tracker_detection_attempts = config.get('num_attempts', 10)
        
        # Device configuration
        self._device_id: dict[str, str] = config["target_device"]  # Default tracker ID
        self._tracker_offset: dict[str, bool] = config.get("tracker_pose_offset", None)
        self._readed_device_id = None
        self._output_left = config.get("output_left", True)
        self._output_right = config.get("output_right", True)
        if self._output_left and self._output_right:
            self._index = {"left": 0, "right": 1}
        elif not self._output_left: 
            self._index = {"single": 1}
        elif not self._output_right:
            self._index = {"single": 0}
        self._position_scale = config.get("position_scale", 1.0)
        # head info
        self._head_info = config.get('head', None)
        if self._head_info: self._index["head"] = 2
        self.T_TRACKER_HEAD = config.get("trakcer_head_trans", None)
        if self.T_TRACKER_HEAD is None:
            self.T_TRACKER_HEAD = np.eye(4)
        else: 
            self.T_TRACKER_HEAD = np.array(self.T_TRACKER_HEAD)
            log.info(f'T tracker head: {self.T_TRACKER_HEAD}')

        super().__init__(config)
        self._init_tracker_robot_axis_alignment()
       
        self._device_enabled = False
        self._reset_pose_counter = 0
        
    
    def _update_tracker_uid(self):
        vive_tracker_device_names = self._tracker.get_devices()
        if len(vive_tracker_device_names) == 0:
            log.error("Failed to initialize Vive Tracker")

        if not self._readed_device_id or len(self._readed_device_id) < len(vive_tracker_device_names):
            self._readed_device_id = vive_tracker_device_names
            return True
        else: return False
    
    def initialize(self) -> bool:
        """Initialize the Pika Sense device and Vive Tracker."""
        if self._is_initialized:
            return True
        
        # sense connection
        for key, sense in self._sense.items():
            if not sense: 
                log.info(f"{key} pika Sense device is not used, skip sense connection")
                continue
            if not sense.connect():
                raise ValueError(f"Failed to connect to Pika Sense device for {key} with serial id: {self._serial_port[key]}")
        
        # tracker initialization
        pose_offset = None
        if self._head_info:
            pose_offset = {}
            pose_offset[self._head_info["device"]] = False
            # device id only contain hand infos
            for key, cur_uid in self._device_id.items():
                if cur_uid:
                    cur_pose_offset = True if not self._tracker_offset or \
                        key not in self._tracker_offset else self._tracker_offset[key]
                    pose_offset[cur_uid] = cur_pose_offset
        self._tracker = ViveTracker(pose_offset=pose_offset, use_uid=True)
        if not self._tracker.connect():
            raise ValueError(f"Failed to connect to HTC vive tracker!!!")
        
        # Initialize device reading requirement
        get_right_device = {'left': False, 'right': False}
        for key, tracker_uid in self._device_id.items():
            if not tracker_uid:
                get_right_device[key] = True
                continue
        
        # pika tracker uid check
        log.info(f'Trying to find the trackers in {self._tracker_detection_attempts} times!')
        for j in range(self._tracker_detection_attempts):
            time.sleep(1.2)
            
            res = self._update_tracker_uid()
            
            # go to next iter without update of device id
            if res: 
                log.info(f'{j}th readed device uid: {self._readed_device_id}')
            else: continue
            
            for key, device_uid in self._device_id.items():
                if device_uid and device_uid in self._readed_device_id:
                    get_right_device[key] = True
            
            # early stop
            if get_right_device["left"] and get_right_device["right"]:
                break
                
        if not get_right_device["left"] or not get_right_device["right"]:
            raise ValueError(f'Failed to get all tracker device id: {get_right_device}')
        
        # head tracker extra check
        if self._head_info:
            get_head_tracker = False
            for j in range(int(self._tracker_detection_attempts / 2)):
                time.sleep(0.5)
                self._update_tracker_uid()
                if self._head_info["device"] in self._readed_device_id:
                    get_head_tracker = True
                    break
            if not get_head_tracker:
                raise ValueError(f'Could not get the head tracker device uid {self._head_info["device"]}')
            
        # keyboard listening for update init pose
        self._key_pressed = False
        self._keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press
        )
        self._keyboard_listener.start()
        log.info("PikaTracker keyboard listener started")
        
        # Wait for tracker data to stabilize
        time.sleep(1.0)
        self._tracker_robot_trans = convert_homo_2_7D_pose(self.T_TRACKER_ROBOT)
        self._tracker_head_trans = convert_homo_2_7D_pose(self.T_TRACKER_HEAD)
        self._robot_tracker_trans = negate_pose(self._tracker_robot_trans)
        self._head_tracker_trans = negate_pose(self._tracker_head_trans)
        log.info(f"PikaTracker initialized successfully with device: {self._serial_port} and device name: {self._device_id}")
        return True
    
    def _update_pose(self, pose, key, tracker_pose_data):
        pose[key] = np.zeros(7)
        pose[key][:3] = tracker_pose_data.position
        pose[key][3:] = tracker_pose_data.rotation  # [qx, qy, qz, qw]
    
    def read_data(self):
        start = time.perf_counter()
        tracker_pose = self._tracker.get_pose()
        pose_time = time.perf_counter() - start
        if not tracker_pose:
            return None, None
            
        pose_quat = {}
        all_pose_flag = {"left": False, "right": False}
        if not self._output_left:
            all_pose_flag["left"] = True
        if not self._output_right:
            all_pose_flag["right"] = True
        for device_name, cur_pose in tracker_pose.items():
            if all_pose_flag["left"] and all_pose_flag["right"]:
                break
            
            if self._output_left and self._device_id["left"] == device_name:
                self._update_pose(pose_quat, "left", cur_pose)
                all_pose_flag["left"] = True
            if self._output_right and self._device_id["right"] == device_name:
                self._update_pose(pose_quat, "right", cur_pose)
                all_pose_flag["right"] = True
        # could not read left & right pose
        if not all(list(all_pose_flag.values())):
            return None, None
        
        start = time.perf_counter()
        tool_data = {}
        for key, sense in self._sense.items():
            if not sense:
                tool_data[key] = np.array([-1.0, 0.0])
                continue
            
            encoder = sense.get_encoder_data()["rad"]
            distance = self._change_motor_rad_to_gripper_distance(encoder)
            normalized_value = distance / 90.0
            normalized_value = np.clip(normalized_value, 0.0, 1.0)
            command = sense.get_command_state()
            tool_data[key] = np.array([normalized_value, command])
        tool_time = time.perf_counter() - start
        # log.info(f'pose time: {pose_time*1000:.1f}ms tool time: {tool_time*1000:.1f}ms')
            
        if not self._output_left and self._output_right:
            pose_quat = dict(single=pose_quat["right"])
            tool_data = dict(single=tool_data["right"])
        if not self._output_right and self._output_left:
            pose_quat = dict(single=pose_quat["left"])
            tool_data = dict(single=tool_data["left"])
        
         # head pose 
        if self._head_info:
            if self._head_info["device"] in tracker_pose:
                cur_pose = tracker_pose[self._head_info["device"]]
                self._update_pose(pose_quat, "head", cur_pose)
            else:
                return None, None
        return pose_quat, tool_data

    def _press_i(self):
        pose, _ = self.read_data()
        if pose is None:
            log.warn("Failed to read pose data for update init pose!!!")
            return
        for i, (key, cur_pose) in enumerate(pose.items()):
            cur_pose = self._process_raw_pose(cur_pose, key)
            self.INIT_TARGET_POSE[self._index[key]] = cur_pose
        self._device_enabled = True
        self._key_pressed = True 
        log.info("init pose is updated!!!")
        
    def _press_u(self):
        self._device_enabled = False
        log.info(f'Pika disabled!!!')

    def _on_key_press(self, key):
        try:
            if key.char == 'i' and not self._device_enabled:
                self._press_i()
                log.info(f'Pika enabled!!!')
            elif key.char == 'u' and self._device_enabled:
                self._press_u()
                log.info(f'Pika disabled!!!')
        except AttributeError:
              # ignore some hotkeys
              pass
          
    def _process_raw_pose(self, pose, key):
        start = time.perf_counter()
        # res_pose = np.zeros(7)
        res_pose = pose
        if key != "head":
            # tracker_robot_trans = convert_homo_2_7D_pose(self.T_TRACKER_ROBOT)
            tracker_robot_trans = self._tracker_robot_trans
            robot_tracker_trans = self._robot_tracker_trans
        else:
            # tracker_robot_trans = convert_homo_2_7D_pose(self.T_TRACKER_HEAD)
            tracker_robot_trans = self._tracker_head_trans
            robot_tracker_trans = self._head_tracker_trans
        res_pose = transform_pose(transform_pose(robot_tracker_trans, pose), tracker_robot_trans)
        basis_change_time = time.perf_counter() - start
        
        start1 = time.perf_counter()
        tracker_offset = True if not self._tracker_offset or \
            key not in self._tracker_offset else self._tracker_offset[key]
        if key != "head" and tracker_offset:
            # read pose init sync to x forward y left z up 
            static_rot_offset = np.array([0, 0, 1, 0])
            res_pose = self.apply_rotation_offset(res_pose, static_rot_offset)
        rot_offset_time = time.perf_counter() - start1
        
        total_time = time.perf_counter()-start
        # log.info(f'{key} pose porcess time: {(total_time)*1000.0:.2f}ms {basis_change_time/total_time*100:.2f}% {rot_offset_time/total_time*100:.2f}%')
        return res_pose

    def parse_data_2_robot_target(self, mode: str) -> Tuple[bool, Optional[Dict], Optional[Dict]]:
        """Parse Vive Tracker pose data to robot target format."""
        start = time.perf_counter()
        if 'absolute' not in mode:
            log.warn('The pika tracker only supports absolute pose related teleoperation')
            raise ValueError(f'{mode} mode is not supported for pika tracker')
        
        if not self._is_initialized:
            log.warn("Device not initialized")
            return False, None, None
        
        # Get pose data from Vive Tracker, all dict
        pose_quat, tool_data = self.read_data()
        read_time = time.perf_counter() - start
        # log.info(f'raw read time {read_time*1000:.1f}ms')
        if pose_quat is None:
            log.warn(f"No pose data available for device: {self._serial_port}")
            return False, None, None
                
        # Prepare robot target dict
        pose_target = {} 
        tool_target = {}
        start0 = time.perf_counter(); sub_process = 0
        for key, value in pose_quat.items():
            # basis change
            start1 = time.perf_counter()
            cur_pose = self._process_raw_pose(value, key)
            sub_process += time.perf_counter() - start1
            
            # @TODO: zyx, select rotation
            # for i in range(3):
            #     if self._rotation_enabled and not self._rotation_enabled[i]:
            #         if mode == "absolute":
            #             cur_pose[3+i] = self._init_pose_rot[self._index[key]][i]
            #         else:
            #             cur_pose[3+i] = self
            
            if mode == "absolute_delta" and self._device_enabled:
                cur_pose = self._get_diff_trans(cur_pose, self._index[key])
                cur_pose[:3] *= self._position_scale
            pose_target[key] = cur_pose
            # skip the tool target for hear
            if "head" in key:
                continue
            tool_target[key] = np.hstack((tool_data[key], copy.deepcopy(self._key_pressed)))
        process_time = time.perf_counter() - start0
        
        if self._key_pressed :
            if self._reset_pose_counter > 15:
                self._key_pressed = False
                self._reset_pose_counter = 0
            else: self._reset_pose_counter += 1

        total_time = time.perf_counter() - start
        # log.info(f'total data parse: {total_time*1000:.2f}ms read {read_time*1000:.2f}ms process {process_time*1000:.2f}ms sub process: {sub_process*1000:.2f}ms')
        if mode == "absolute" or (mode == "absolute_delta" and self._device_enabled):
            return True, pose_target, tool_target
        else: return False, None, None
            
    def print_data(self):
        data = self.parse_data_2_robot_target("absolute")
        log.info(f'pose: {data[1]}')
    
    def close(self):
        """Clean up resources and disconnect devices."""
        if not self._is_initialized:
            return 
        
        self._device_enabled = False
        self._tracker.disconnect()
        for key, sense in self._sense.items():
            if sense:
                sense.disconnect()
        if self._keyboard_listener:
            self._keyboard_listener.stop()
        log.info("PikaTracker disconnected successfully")
        return True
    
    def apply_rotation_offset(self, pose, rot_offset):
        # new_pose = copy.deepcopy(pose)
        # apply rotation offset
        pose[3:] = transform_quat(pose[3:], rot_offset)
        return pose
    
    def _get_diff_trans(self, cur_pose, index):
        diff_pose = pose_diff(cur_pose, self.INIT_TARGET_POSE[index])
        return diff_pose
    
    def _change_motor_rad_to_gripper_distance(self, angle_rad):
        distance = (self._get_distance(angle_rad) - self._get_distance(0)) * 2.0
        return distance
            
    def _get_distance(self, angle):
        angle = (180.0 - 43.99) / 180.0 * math.pi - angle
        height = 0.0325 * math.sin(angle)
        width_d = 0.0325 * math.cos(angle)
        width = math.sqrt(0.058**2 - (height - 0.01456)**2) + width_d
        return width*1000
    