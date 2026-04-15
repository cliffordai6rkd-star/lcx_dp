"""Mock implementation of pika gripper for simulation-only mode"""
import numpy as np
import time
import glog as log


class MockGripper:
    """Mock Pika gripper"""
    def __init__(self, serial_port):
        self._serial_port = serial_port
        self._distance = 90.0  # Start at max distance (open)
        self._motor_temp = 25.0  # Room temperature
        self._motor_current = 10.0  # Low current
        self._connected = False
        self._enabled = False
        log.info(f"Mock Pika Gripper created for port: {serial_port}")

    def connect(self):
        """Mock connect to gripper"""
        self._connected = True
        log.info(f"Mock Pika Gripper connected on {self._serial_port}")
        return True

    def disconnect(self):
        """Mock disconnect from gripper"""
        self._connected = False
        log.info(f"Mock Pika Gripper disconnected from {self._serial_port}")

    def enable(self):
        """Mock enable motor"""
        self._enabled = True
        log.info(f"Mock Pika Gripper motor enabled on {self._serial_port}")
        return True

    def disable(self):
        """Mock disable motor"""
        self._enabled = False
        log.info(f"Mock Pika Gripper motor disabled on {self._serial_port}")

    def set_gripper_distance(self, distance):
        """Mock set gripper distance in mm"""
        if not self._connected or not self._enabled:
            log.warning(f"Mock Pika Gripper not ready: connected={self._connected}, enabled={self._enabled}")
            return False

        self._distance = np.clip(distance, 0.0, 90.0)
        # Simulate slight current increase when moving
        self._motor_current = 15.0 if abs(self._distance - distance) > 5 else 10.0
        log.debug(f"Mock Pika Gripper set distance to {self._distance:.1f}mm")
        return True

    def get_gripper_distance(self):
        """Mock get current gripper distance in mm"""
        return self._distance

    def get_motor_temp(self):
        """Mock get motor temperature"""
        # Simulate slight temperature variation
        return self._motor_temp + np.random.uniform(-1, 1)

    def get_motor_current(self):
        """Mock get motor current"""
        # Simulate slight current variation
        return self._motor_current + np.random.uniform(-2, 2)


# Module-level exports
Gripper = MockGripper
