from teleop.base.teleoperation_base import TeleoperationDeviceBase
from pika.sense import Sense
import glog as log

class PikaTracker(TeleoperationDeviceBase):
    def __init__(self, config):
        super().__init__(config)
        self._serial_port = config["serial_port"]
        self._tracker = Sense(self._serial_port)
        
    def initialize(self):
        return super().initialize()
    
    def parse_data_2_robot_target(self, mode):
        if not 'absolute' in mode:
            log.warn(f'The pika tracker only supports for absolute pose related teleoperation')
            return False, None, None

        