"""Mock implementation of NetFT (ATI Force/Torque sensor) for simulation-only mode"""
import numpy as np
import time
import glog as log


class MockSensor:
    """Mock ATI Force/Torque sensor"""
    def __init__(self, ip):
        self._ip = ip
        self._bias = np.zeros(6)
        self._streaming = False
        log.info(f"Mock ATI F/T sensor created for IP: {ip}")

    def zero(self):
        """Zero the sensor"""
        self._bias = self.getMeasurement()
        log.info(f"Mock ATI F/T sensor zeroed at IP {self._ip}")

    def startStreaming(self):
        """Start streaming data"""
        self._streaming = True
        log.info(f"Mock ATI F/T sensor started streaming at IP {self._ip}")

    def stopStreaming(self):
        """Stop streaming data"""
        self._streaming = False
        log.info(f"Mock ATI F/T sensor stopped streaming at IP {self._ip}")

    def getMeasurement(self):
        """Get a single measurement (blocking)"""
        # Return mock F/T data with small random noise
        # Format: [Fx, Fy, Fz, Tx, Ty, Tz] in micro units
        base_forces = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        noise = np.random.uniform(-1000, 1000, 6)  # Small noise in micro units
        return base_forces + noise

    def measurement(self):
        """Get current measurement from stream"""
        if not self._streaming:
            log.warning(f"Mock ATI F/T sensor at {self._ip} not streaming")
            self.startStreaming()

        return self.getMeasurement()


# Module-level exports
Sensor = MockSensor
