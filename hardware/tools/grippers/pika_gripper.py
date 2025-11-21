from hardware.base.tool_base import ToolBase
from hardware.base.utils import ToolState, ToolType

# Try to import pika gripper, fall back to mock if not available
# try:
#     from pika.gripper import Gripper
# except (ImportError, ModuleNotFoundError):
#     import glog as log
#     log.warning("pika.gripper not available, using mock implementation")
#     from hardware.mocks.mock_pika import Gripper

from dependencies.pika_sdk.pika.gripper import Gripper

import threading
import time
import copy
import numpy as np
import glog as log

class PikaGripper(ToolBase):
    _tool_type: ToolType = ToolType.GRIPPER
    def __init__(self, config):
        self._serial_port = config["serial_port"]
        self._max_distance = 90.0  # mm
        self._min_distance = 0.0   # mm
        self._update_frequency = config.get("update_frequency", 100.0)  # Hz
        self._temp_threshold = config.get("temp_threshold", 60) 
        self._current_threshold = config.get("current_threshold", 100) 
        
        # Pika gripper instance
        self._pika_gripper = Gripper(self._serial_port)
        
        # Thread management for state updates
        self._thread_running = False
        self._update_thread = None
        self._lock = threading.Lock()
        
        # Initialize ToolBase after setting required attributes
        super().__init__(config)
        self.set_hardware_command(self._current_position_scaled)
        self._state._position = \
            self._current_position_scaled * (self._max_distance - self._min_distance)\
            + self._min_distance
        log.info(f'init posi: {self._state._position}')

        # State management
        self._state._tool_type = self._tool_type
        
    def initialize(self) -> bool:
        """Initialize the Pika Gripper connection"""
        if self._is_initialized:
            log.warn("Pika Gripper already initialized")
            return True
        
        try:
            # Connect to the gripper
            if not self._pika_gripper.connect():
                log.error(f"Failed to connect to Pika Gripper on {self._serial_port}")
                return False
            
            # Enable the motor
            if not self._pika_gripper.enable():
                log.error("Failed to enable Pika Gripper motor")
                return False
            
            # Move to open position (safe initialization)
            time.sleep(0.2)  # Wait for connection to stabilize

            success = self._pika_gripper.set_gripper_distance(self._max_distance)
            self._state._position = self._max_distance
            
            # Start state update thread
            self._gripper_state_updated = False
            self._thread_running = True
            self._update_thread = threading.Thread(target=self.update_state, daemon=True)
            self._update_thread.start()
            
            while not self._gripper_state_updated:
                time.sleep(0.001)
            
            if success:
                log.info(f"Pika Gripper initialized successfully on {self._serial_port}")
                return True
            else:
                log.error("Failed to move Pika Gripper to initial position")
                return False
                
        except Exception as e:
            log.error(f"Exception during Pika Gripper initialization: {e}")
            return False
        
    def recover(self):
        # @TODO: recover
        return 
    
    def _get_motor_status(self):
        if not self._is_initialized:
            return None, None
        
        temp = self._pika_gripper.get_motor_temp()
        current = self._pika_gripper.get_motor_current()
        return (temp, current)
    
    def check_is_over_current_and_temp(self, temp, current):
        if temp > self._temp_threshold or current > self._current_threshold:
            return True, temp, current
        else: return False, None, None
    
    def set_hardware_command(self, command):
        if not self._is_initialized:
            log.warn(f'Pika Gripper is not initialized!')
            return
        
        # Normalize and clamp target
        target = np.clip(command, 0.0, 1.0)
        
        # Map [0,1] to [min_distance, max_distance] mm
        target_distance = self._min_distance + target * (self._max_distance - self._min_distance)
        
        # Avoid continuous identical commands (debounce)
        if np.isclose(target_distance, self._state._position, rtol=0.001):
            return
        
        success = self._pika_gripper.set_gripper_distance(target_distance)
        if not success:
            log.warn(f"Failed to move Pika Gripper to {target_distance:.1f}mm")
        else:
            time.sleep(0.12)
            current_position = self._pika_gripper.get_gripper_distance()
            is_grasped = current_position > target_distance + 2
            # log.info(f'check : {is_grasped}, current: {current_position}, target: {target_distance} diff: {current_position - target_distance}')
            with self._lock:
                self._state._is_grasped = is_grasped
                # self._state._position = target_distance
        
    def get_tool_state(self) -> ToolState:
        """Get current tool state in thread-safe manner"""
        if not self._gripper_state_updated:
            return None
        
        with self._lock:
            return copy.deepcopy(self._state)
    
    def update_state(self):
        """Background thread for updating gripper state"""
        log.info(f"Starting Pika Gripper state update thread for {self._serial_port}")
        
        last_read_time = time.time()
        dt_target = 1.0 / self._update_frequency
        # max_time = 0
        while self._thread_running:
            try:
                # Read current gripper state
                start = time.perf_counter()
                current_distance = self._pika_gripper.get_gripper_distance()
                distance_reading_time = time.perf_counter() - start
                # avoid position sudden change
                with self._lock: 
                    last_position = self._state._position
                
                if abs(current_distance - last_position) > 0.5 * (self._max_distance - self._min_distance):
                    # log.warn(f'Detected sudden change of gripper position: {last_position}, {current_distance}')
                    continue
                start = time.perf_counter()
                motor_temp, motor_current = self._get_motor_status()
                distance_reading_time += time.perf_counter() - start
                # log.info(f'cur dist: {current_distance}')
                self._gripper_state_updated = True
                
                # Update state in thread-safe manner
                with self._lock:
                    self._state._position = current_distance
                    self._state._time_stamp = time.perf_counter()
                    # Determine grasp state based on current (high current = grasping)
                over_temp, temp, current = self.check_is_over_current_and_temp(motor_temp, motor_current)
                if over_temp:
                    # log.warn(f'Pika gripper {self._serial_port} is over temp, temp: {temp}, current: {current}')
                    self._state._is_grasped = True
                
            except Exception as e:
                log.warn(f"Error reading Pika Gripper state: {e}")
            
            # Timing control
            dt = time.time() - last_read_time
            last_read_time = time.time()
            # if dt > max_time: max_time = dt
            # log.info(f'pika gripper read state dt: {1.0 / dt} Hz, max freq: {1.0/max_time} hz')
            # log.info(f'pika gripper read state dt: {1.0 / dt} Hz')
            if dt < dt_target:
                sleep_time = dt_target - dt
                time.sleep(sleep_time)
            # elif dt > 1.2 * dt_target:
            #     log.warn(f"Pika Gripper {self._serial_port} update thread running slow: {1.0 / dt}hz, \
            #         expected: {self._update_frequency}hz, reading time: {1.0/distance_reading_time}hz")
            
        log.info(f"Pika Gripper {self._serial_port} state update thread stopped")
    
    def stop_tool(self):
        """Stop the gripper and clean up resources"""
        # Stop update thread
        self._thread_running = False
        if self._update_thread is not None and self._update_thread.is_alive():
            self._update_thread.join(timeout=1.0)
        
        # Disable motor and disconnect
        if hasattr(self, '_pika_gripper'):
            self._pika_gripper.disable()
            self._pika_gripper.disconnect()
        
        log.info(f"Pika Gripper {self._serial_port} stopped successfully")
            
    def get_tool_type_dict(self):
        """Return tool type dictionary for framework compatibility"""
        return {'single': self._tool_type}
    