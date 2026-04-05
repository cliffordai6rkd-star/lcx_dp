"""Mock implementation of pyrealsense2 for simulation-only mode"""
import numpy as np
import glog as log


class MockStream:
    """Mock stream types"""
    color = 0
    depth = 1
    accel = 2
    gyro = 3


class MockFormat:
    """Mock format types"""
    bgr8 = 0
    z16 = 1
    motion_xyz32f = 2


class MockAlign:
    """Mock align class"""
    def __init__(self, align_to):
        self._align_to = align_to

    def process(self, frames):
        """Return the same frames"""
        return frames


class MockMotionData:
    """Mock motion data"""
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class MockMotionFrame:
    """Mock motion frame"""
    def __init__(self):
        self._data = MockMotionData()

    def get_motion_data(self):
        return self._data


class MockFrame:
    """Mock frame"""
    def __init__(self, frame_type):
        self._type = frame_type
        self._data = None

    def get_data(self):
        # Return mock image data
        if self._type == MockStream.color:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        elif self._type == MockStream.depth:
            return np.zeros((480, 640), dtype=np.uint16)
        return None

    def as_motion_frame(self):
        return MockMotionFrame()

    def get_timestamp(self):
        import time
        return time.time()


class MockFrames:
    """Mock frames collection"""
    def __init__(self):
        self._frames = {}

    def get_color_frame(self):
        return MockFrame(MockStream.color)

    def get_depth_frame(self):
        return MockFrame(MockStream.depth)

    def first_or_default(self, stream_type):
        return MockFrame(stream_type)


class MockIntrinsics:
    """Mock camera intrinsics"""
    def __init__(self):
        self.width = 640
        self.height = 480
        self.ppx = 320.0
        self.ppy = 240.0
        self.fx = 600.0
        self.fy = 600.0
        self.model = 0
        self.coeffs = [0, 0, 0, 0, 0]


class MockVideoStreamProfile:
    """Mock video stream profile"""
    def get_intrinsics(self):
        return MockIntrinsics()


class MockStreamProfile:
    """Mock stream profile"""
    def as_video_stream_profile(self):
        return MockVideoStreamProfile()


class MockDepthSensor:
    """Mock depth sensor"""
    def get_depth_scale(self):
        return 0.001  # 1mm scale


class MockDevice:
    """Mock RealSense device"""
    def first_depth_sensor(self):
        return MockDepthSensor()


class MockProfile:
    """Mock pipeline profile"""
    def get_device(self):
        return MockDevice()

    def get_stream(self, stream_type):
        return MockStreamProfile()


class MockConfig:
    """Mock pipeline configuration"""
    def enable_device(self, serial_number):
        log.info(f"Mock RealSense: enabled device {serial_number}")

    def enable_stream(self, stream_type, width=None, height=None, format_type=None, fps=None):
        log.debug(f"Mock RealSense: enabled stream {stream_type}")


class MockPipeline:
    """Mock RealSense pipeline"""
    def __init__(self):
        self._running = False

    def start(self, config):
        """Start the pipeline"""
        self._running = True
        log.info("Mock RealSense pipeline started")
        return MockProfile()

    def stop(self):
        """Stop the pipeline"""
        self._running = False
        log.info("Mock RealSense pipeline stopped")

    def wait_for_frames(self):
        """Return mock frames"""
        import time
        time.sleep(0.001)  # Simulate frame capture delay
        return MockFrames()


# Module-level functions
def pipeline():
    """Create a mock pipeline"""
    return MockPipeline()


def config():
    """Create a mock config"""
    return MockConfig()


def align(align_to):
    """Create a mock align object"""
    return MockAlign(align_to)


# Module-level constants
stream = MockStream()
format = MockFormat()
