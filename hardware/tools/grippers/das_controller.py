import argparse
import copy
import struct
import threading
import time

import glog as log
import numpy as np

from hardware.base.tool_base import ToolBase
from hardware.base.utils import ToolState, ToolType

import sys, os
gen_py_sdk_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../dependencies/gen_con_sdk_python_release"
)
sys.path.insert(0, gen_py_sdk_path)
from dependencies.gen_con_sdk_python_release.gen_controller_sdk_python import DataBus


class DasController(ToolBase):
    _tool_type: ToolType = ToolType.GRIPPER

    def __init__(self, config):
        self._serial_port = config.get("serial_port", None)
        self._update_frequency = config.get("update_frequency", 50.0)
        self._max_distance = 0.103  # meters
        self._min_distance = 0.0    # meters
        self._grasp_threshold = config.get("grasp_threshold", 0.002)  # meters

        self._databus = None
        self._lock = threading.Lock()
        self._gripper_state_updated = False
        self._target_distance = None
        self._last_command_distance = None

        super().__init__(config)
        self._state._tool_type = self._tool_type

    def initialize(self) -> bool:
        if self._is_initialized:
            log.warn("DasController already initialized")
            return True

        if not self._serial_port:
            log.error("DasController serial_port not set in config")
            raise ValueError("DasController requires a valid serial_port")
            return False

        try:
            self._databus = DataBus(
                tty_port=self._serial_port,
                baudrate=self._config.get("baudrate", 921600),
                encoder_freq=self._update_frequency,
                encoder_callback=self._encoder_callback,
            )
        except Exception as e:
            log.error(f"Failed to open DasController serial port {self._serial_port}: {e}")
            raise ValueError(f"Failed to open serial port: {self._serial_port}") from e
            return False

        # Send initial target based on current scaled position
        init_target = float(np.clip(self._current_position_scaled, 0.0, 1.0))
        init_distance = self._min_distance + init_target * (self._max_distance - self._min_distance)
        try:
            self._databus.set_target_distance(init_distance)
            self._target_distance = init_distance
            self._last_command_distance = init_distance
        except Exception as e:
            log.error(f"Failed to set initial DasController distance: {e}")
            return False

        # Wait for first encoder update
        start_time = time.perf_counter()
        while not self._gripper_state_updated:
            time.sleep(0.001)
            if time.perf_counter() - start_time > 2.0:
                log.warn("DasController did not receive encoder data within timeout")
                break

        log.info(f"DasController initialized successfully on {self._serial_port}")
        return True

    def recover(self):
        # @TODO: recover
        return

    def _encoder_callback(self, record_data: bytes):
        try:
            distance = struct.unpack(">f", record_data)[0]
        except Exception as e:
            log.warn(f"DasController encoder callback parse error: {e}")
            return

        with self._lock:
            self._state._position = distance
            self._state._time_stamp = time.perf_counter()
            if self._target_distance is not None:
                self._state._is_grasped = distance > (self._target_distance + self._grasp_threshold)

        if not self._gripper_state_updated:
            self._gripper_state_updated = True

    def set_hardware_command(self, command):
        if not self._is_initialized:
            log.warn("DasController is not initialized")
            return

        target = float(np.clip(command, 0.0, 1.0))
        target_distance = self._min_distance + target * (self._max_distance - self._min_distance)

        if self._last_command_distance is not None and np.isclose(
            target_distance, self._last_command_distance, rtol=0.001, atol=1e-4
        ):
            return

        try:
            self._databus.set_target_distance(target_distance)
            self._target_distance = target_distance
            self._last_command_distance = target_distance
        except Exception as e:
            log.warn(f"Failed to move DasController to {target_distance:.4f}m: {e}")

    def get_tool_state(self) -> ToolState:
        if not self._gripper_state_updated:
            return None

        with self._lock:
            return copy.deepcopy(self._state)

    def stop_tool(self):
        if self._databus is not None:
            self._databus.stop()
        log.info(f"DasController {self._serial_port} stopped successfully")

    def get_tool_type_dict(self):
        return {'single': self._tool_type}

def main():
    def _run_target(controller: DasController, target_scaled: float):
        controller.set_tool_command(target_scaled)
        state = controller.get_tool_state()
        if state is not None:
            log.info(
                f"DasController state: pos={state._position:.4f}m "
                f"target={target_scaled*controller._max_distance:.4f}m "
                f"raw target={target_scaled}"
            )
        
        
    config = {
        "serial_port": "/dev/ttyUSB0",
        # "baudrate": 
        "update_frequency": 200,
        "grasp_threshold": 0.01, # 1cm
        "control_mode": "binary",
        "binary_threshold": 0.5,
        "step_size": 0.6
    }

    try:
        controller = DasController(config)
    except ValueError as e:
        log.error(f"DasController init failed: {e}")
        return 1

    target_a = 0.2; target_b = 1.0
    counter = 0
    while True:
        if counter == 0:
            _run_target(controller, target_a)
        elif counter == 500: 
            _run_target(controller, target_b)
        elif counter == 1000: counter = -1
            
        time.sleep(0.001)
        counter += 1 

if __name__ == "__main__":
    raise SystemExit(main())
