"""Mock implementation of unitree_sdk2py for simulation-only mode"""
import numpy as np
import glog as log


class MockChannelSubscriber:
    """Mock channel subscriber"""
    def __init__(self, topic):
        self._topic = topic
        log.debug(f"Mock Unitree subscriber created for topic: {topic}")

    def Init(self, callback, *args):
        """Initialize subscriber"""
        log.debug(f"Mock Unitree subscriber initialized for {self._topic}")


class MockChannelPublisher:
    """Mock channel publisher"""
    def __init__(self, topic):
        self._topic = topic
        log.debug(f"Mock Unitree publisher created for topic: {topic}")

    def Init(self):
        """Initialize publisher"""
        log.debug(f"Mock Unitree publisher initialized for {self._topic}")

    def Write(self, data):
        """Publish data"""
        log.debug(f"Mock Unitree publisher writing to {self._topic}")


def MockChannelFactoryInitialize(domain_id=0, interface=""):
    """Mock channel factory initialization"""
    log.info(f"Mock Unitree SDK initialized with domain_id={domain_id}, interface={interface}")


# Mock RTC module
class MockRTC:
    """Mock RTC module"""
    class MockMotionSwitcher:
        """Mock motion switcher"""
        def __init__(self):
            pass

        def SelectMode(self, mode):
            log.debug(f"Mock motion switcher mode: {mode}")

        def ReleaseMode(self):
            log.debug("Mock motion switcher released")

    class MockLowCmd:
        """Mock low-level command"""
        def __init__(self):
            self.mode_pr = 0
            self.mode_machine = 0
            self.motor_cmd = [self.MockMotorCmd() for _ in range(30)]

        class MockMotorCmd:
            """Mock motor command"""
            def __init__(self):
                self.q = 0.0
                self.dq = 0.0
                self.kp = 0.0
                self.kd = 0.0
                self.tau = 0.0


# Mock IDL modules
class MockIDL:
    """Mock IDL module"""
    class MockMotorState:
        """Mock motor state"""
        def __init__(self):
            self.q = 0.0
            self.dq = 0.0
            self.ddq = 0.0
            self.tau_est = 0.0

    class MockLowState:
        """Mock low state"""
        def __init__(self):
            self.motor_state = [MockIDL.MockMotorState() for _ in range(30)]
            self.imu_state = self.MockIMUState()

        class MockIMUState:
            """Mock IMU state"""
            def __init__(self):
                self.quaternion = [1.0, 0.0, 0.0, 0.0]
                self.gyroscope = [0.0, 0.0, 0.0]
                self.accelerometer = [0.0, 0.0, 9.81]


# Mock utils module
class MockUtils:
    """Mock utils module"""
    @staticmethod
    def QueryServiceStatus(service_name):
        """Query service status - always return active in mock"""
        log.debug(f"Mock query service status: {service_name}")
        return True


# Module-level exports
class core:
    """Mock core module"""
    class channel:
        """Mock channel module"""
        ChannelSubscriber = MockChannelSubscriber
        ChannelPublisher = MockChannelPublisher
        ChannelFactoryInitialize = MockChannelFactoryInitialize


class rtc:
    """Mock RTC module"""
    class RTC:
        """Mock RTC class"""
        MotionSwitcher = MockRTC.MockMotionSwitcher
        LowCmd = MockRTC.MockLowCmd


class idl:
    """Mock IDL module"""
    class unitree_go:
        """Mock unitree_go module"""
        class msg:
            """Mock msg module"""
            class dds_:
                """Mock dds_ module"""
                LowState_ = MockIDL.MockLowState


class utils:
    """Mock utils module"""
    class thread:
        """Mock thread module"""
        RecurrentThread = None  # Not needed for mock

    QueryServiceStatus = MockUtils.QueryServiceStatus
