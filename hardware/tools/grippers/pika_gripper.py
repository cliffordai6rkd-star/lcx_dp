from hardware.base.tool_base import ToolBase
from hardware.base.utils import ToolState, ToolType
from pika.gripper import Gripper
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
        self._gripper_step_value = config.get("step_velue", 5) # 5mm
        self._temp_threshold = config.get("temp_threshold", 60) 
        self._current_threshold = config.get("current_threshold", 100) 
        
        # Pika gripper instance
        self._pika_gripper = Gripper(self._serial_port)
        
        # State management
        self._state._tool_type = self._tool_type
        self._last_command = 0.0
        self._gripper_idle = True
        
        # Thread management for state updates
        self._thread_running = False
        self._update_thread = None
        self._lock = threading.Lock()
        
        # Initialize ToolBase after setting required attributes
        super().__init__(config)
        
        
    def initialize(self) -> bool:
        """Initialize the Pika Gripper connection"""
        if self._is_initialized:
            log.warning("Pika Gripper already initialized")
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
            
            # Start state update thread
            self._thread_running = True
            self._update_thread = threading.Thread(target=self.update_state, daemon=True)
            self._update_thread.start()
            self._gripper_state_updated = False
            
            while not self._gripper_state_updated:
                time.sleep(0.001)
            success = self._pika_gripper.set_gripper_distance(self._max_distance)
            
            if success:
                log.info(f"Pika Gripper initialized successfully on {self._serial_port}")
                return True
            else:
                log.error("Failed to move Pika Gripper to initial position")
                return False
                
        except Exception as e:
            log.error(f"Exception during Pika Gripper initialization: {e}")
            return False
    
    def get_motor_status(self):
        if not self._is_initialized:
            return None
        
        temp = self._pika_gripper.get_motor_temp()
        current = self._pika_gripper.get_motor_current()
        return (temp, current)
    
    def set_tool_command(self, target):
        """Set gripper command with target range [0,1]
        
        Args:
            target: float in range [0,1] where 0=closed, 1=fully open
        """
        if not self._gripper_idle:
            log.debug("Pika Gripper busy, ignoring command")
            return
        
        if not self._is_initialized:
            log.warn(f'Pika Gripper is not initialized!')
            return
        
        # Normalize and clamp target
        target = np.clip(target, 0.0, 1.0)
        
        # Map [0,1] to [min_distance, max_distance] mm
        target_distance = self._min_distance + target * (self._max_distance - self._min_distance)
        
        # Avoid continuous identical commands (debounce)
        if np.isclose(target_distance, self._last_command, rtol=0.001):
            return
        
        def grasp_task():
            """Non-blocking gripper control task"""
            try:
                self._gripper_idle = False
                current_distance = self._pika_gripper.get_distance()
                while not np.isclose(current_distance, target_distance):
                    current_distance += self._gripper_step_value
                    success = self._pika_gripper.set_gripper_distance(current_distance)

                    if not success:
                        log.warning(f"Failed to move Pika Gripper to {current_distance:.1f}mm for {target_distance}")
                        break
                    else:
                        with self._lock:
                            self._state._position = target_distance
                        self._last_command = target_distance
                        log.debug(f"Pika Gripper moved to {target_distance:.1f}mm")
                    
                    temp, current = self.get_motor_status()
                    if temp > self._temp_threshold or current > self._current_threshold:
                        with self._lock:
                            self._state._is_grasped = True
                        log.warning(f'Grasp task has overloaded temp or motor current')
                        break
            except Exception as e:
                log.error(f"Exception in Pika Gripper grasp task: {e}")
            finally:
                with self._lock:
                    if self._state._position >= target_distance or np.isclose(self._state._position):
                        self._state._is_grasped = True 
                    else: self._state._is_grasped = False
                self._gripper_idle = True
        
        # Execute command in separate thread to avoid blocking
        gripper_thread = threading.Thread(target=grasp_task, daemon=True)
        gripper_thread.start()
    
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
        
        while self._thread_running:
            try:
                # Read current gripper state
                current_distance = self._pika_gripper.get_gripper_distance()
                current_current = self._pika_gripper.get_motor_current()
                self._gripper_state_updated = True
                
                # Update state in thread-safe manner
                with self._lock:
                    self._state._position = current_distance
                    # Determine grasp state based on current (high current = grasping)
                    self._state._is_grasped = (abs(current_current) > self._current_threshold)  # Threshold for grasp detection
                
            except Exception as e:
                log.warning(f"Error reading Pika Gripper state: {e}")
            
            # Timing control
            dt = time.time() - last_read_time
            if dt < dt_target:
                sleep_time = dt_target - dt
                time.sleep(sleep_time)
            elif dt > 1.3 * dt_target:
                log.warning(f"Pika Gripper {self._serial_port} update thread running slow: {dt:.3f}s")
            
            last_read_time = time.time()
        
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
    