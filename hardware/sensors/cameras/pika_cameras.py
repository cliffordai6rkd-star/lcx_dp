from hardware.base.camera import CameraBase
from pika.gripper import Gripper

class PikaCameras(CameraBase):
    def __init__(self, config):
        super().__init__(config)
        
    def initialize(self):
        return super().initialize()
    
    def update_thread(self):
        pass
    
    def close(self):
        return super().close()
    
    
