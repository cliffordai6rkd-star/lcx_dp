from hardware.communication.servo_pika_img_interface import G1UmiClient
from hardware.base.tool_base import ToolBase, ToolControlMode
from hardware.base.utils import ToolState, ToolType
import numpy as np
import glog as log
import copy, threading, time
from functools import partial

"""
    duo tool: left and right pika in one class
"""
class ZmqPika(ToolBase):
    _tool_type: ToolType = ToolType.GRIPPER
    def __init__(self, config):
        self._server_ip = config["ip"]
        self._ctrl_port = config.get("port", 5555)
        self._update_frequency = config.get("update_frequency", 100.0)  # Hz
        self._max_distance = 90.0  # mm
        self._min_distance = 0.0   # mm
        
        self._zmq_interface = G1UmiClient(self._server_ip, 
            self._ctrl_port, img_endpoint=None, require_control=True)
        time.sleep(1.0)
        
        self._gripper_state_updated = False
        self._zmq_lock = threading.Lock()
        self._lock = threading.Lock()
        self._thread_running = False
        self._update_thread = None
        
        super().__init__(config)
        
    def initialize(self):
        if self._is_initialized:
            return True
        
        init_command = self._current_position_scaled * self._max_distance
        for i in range(3):
            res = self._zmq_interface.set_all_gripper_commands(init_command, init_command)
            if not res:
                return False
            cur_position = self._zmq_interface.get_all_gripper_positions()
            log.info(f'ZMQ Pika gripper trying to connect with current positions: {cur_position}')
            
        # thread starting
        self._state = {"left": ToolState(), "right": ToolState()}
        self._gripper_ok = {"left": None, "right": None}
        self._gripper_check_lock = threading.Lock()
        self._thread_running = True
        self._update_thread = threading.Thread(target=self.update_loop, daemon=True)
        self._update_thread.start()
        while not self._gripper_state_updated:
            time.sleep(0.001)
            
        log.info(f'ZMQ Pika grippers initialized!!!!')
        return True
    
    def recover(self):
        # @TODO: recover
        return 
    
    def update_loop(self):
        log.info(f'Started zmq pika grippers state updating loop!')
        
        expected_dt = 1.0 / self._update_frequency
        last_read_time = time.perf_counter()
        counter = 0
        while self._thread_running:
            with self._zmq_lock:
                current_position = self._zmq_interface.get_all_gripper_positions()
            with self._lock:
                self._state["left"]._position = current_position[0]
                self._state["left"]._time_stamp = time.perf_counter()
                self._state["right"]._position = current_position[1]
                self._state["right"]._time_stamp = time.perf_counter()
            if not self._gripper_state_updated: self._gripper_state_updated = True
            
            dt = time.perf_counter() - last_read_time
            last_read_time = time.perf_counter()
            if dt < expected_dt:
                sleep_time = expected_dt - dt
                time.sleep(0.92*sleep_time)
            elif dt > 1.3* expected_dt:
                counter += 1
                if counter % 600 == 0:
                    log.warn(f'ZMQ PIKA gripper update slow, real: {1.0/dt:.2f}, expected: {self._update_frequency}')
                    counter = 0
                
        log.info(f'ZMQ Pika gripper update thread stopped!!!')
    
    def set_hardware_command(self, command):
        pass

    def set_single_command(self, command, key):
        if not self._is_initialized:
            return 

        command = command * self._max_distance
        with self._zmq_lock:
            self._zmq_interface.set_gripper_command(command, key)
        
    def set_tool_command(self, target):
        assert len(target) == 2, "ZMQ Pika gripper need to have two targets"
        if isinstance(target, dict):
            new_command = [target["left"], target["right"]]
        else: new_command = target
        
        keys = ["left", "right"]
        for i, command in enumerate(new_command):
            with self._gripper_check_lock:
                gripper_ok = self._gripper_ok[keys[i]]
                gripper_ok = True if gripper_ok is None or not gripper_ok.is_alive() else False
            if not gripper_ok:
                log.debug(f"ZMQ gripper {keys[i]} is currently working on other command, please wait for some time to set new command") 
                continue

            new_command[i] = np.clip(command, 0, 1)
            cur_value = self._state[keys[i]]._position / self._max_distance
            if self._control_mode == ToolControlMode.BINARY:
                # Binary mode: extract value -> threshold judgment -> 0.0 or 1.0
                target = self._apply_binary_threshold(new_command[i])
                self._gripper_ok[keys[i]] = False
                success = self.set_single_command(target, keys[i])
                self._gripper_ok[keys[i]] = True
            elif self._control_mode == ToolControlMode.INCREMENTAL:
                thread = self._handle_gripper_incremental_command(new_command[i], 
                    cur_value=cur_value, func=partial(self.set_single_command, key=keys[i]))
                with self._gripper_check_lock:
                    self._gripper_ok[keys[i]] = thread
            else:
                raise ValueError(f"Unsupported control mode: {self._control_mode}")

    def get_tool_state(self) -> ToolState:
        """Get current tool state in thread-safe manner"""
        if not self._gripper_state_updated:
            return None
        
        with self._lock:
            return copy.deepcopy(self._state)
    
    def stop_tool(self):
        """Stop the gripper and clean up resources"""
        # Stop update thread
        self._thread_running = False
        if self._update_thread is not None and self._update_thread.is_alive():
            self._update_thread.join(timeout=1.0)
        
        # Disable motor and disconnect
        if hasattr(self, '_zmq_interface'):
            self._zmq_interface.close()
        
        log.info(f"ZMQ Pika Gripper stopped successfully")
    
    
    def get_tool_type_dict(self):
        """Return tool type dictionary for framework compatibility"""
        return {'left': self._tool_type, 'right': self._tool_type}
    