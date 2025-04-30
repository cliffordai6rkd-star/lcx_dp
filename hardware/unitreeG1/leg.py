from typing import Text, Mapping, Any
from hardware.base.leg import LegBase

import glog as log
import time,math
import numpy as np
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

Kp = [
    60, 60, 60, 100, 40, 40,      # legs
    60, 60, 60, 100, 40, 40,      # legs
    60, 40, 40,                   # waist
    40, 40, 40, 40,  40, 40, 40,  # arms
    40, 40, 40, 40,  40, 40, 40   # arms
]

Kd = [
    1, 1, 1, 2, 1, 1,     # legs
    1, 1, 1, 2, 1, 1,     # legs
    1, 1, 1,              # waist
    1, 1, 1, 1, 1, 1, 1,  # arms
    1, 1, 1, 1, 1, 1, 1   # arms 
]

class Mode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints

class Leg(LegBase):
  
  def __init__(self, config: Mapping[Text, Any], isLeft: bool = True):
    super().__init__()

    self.jointIndices=[]
    if isLeft:
        self.jointIndices = [0,1,2,3,4,5]
    else:
        self.jointIndices = [6,7,8,9,10,11]

  def print_state(self, low_state: LowState_):
      for idx in self.jointIndices:
          print(f"motor_state[{idx}]: {low_state.motor_state[idx]}")

  def get_model(self):
      pass
  
  def get_state(self):
      pass
  


 
