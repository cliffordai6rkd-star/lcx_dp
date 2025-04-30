from typing import Text, Mapping, Any

import time
from hardware.unitreeG1.arm import Arm
from hardware.unitreeG1.leg import Leg
from hardware.unitreeG1.waist import Waist

from hardware.base.robot import Robot

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

import numpy as np

import glog as log

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
class Agent(Robot):
    def __init__(self,  config: Mapping[Text, Any]):
        log.info(f"network_interface: {config['network_interface']}")

        self.update_mode_machine_ = False
        self.mode_machine_ = 0
        self.low_state = None
        self.time_ = 0.0
        self.control_dt_ = config['control_dt']  # [2ms]
        self.duration_ = config['duration']    # [3 s]
        self.counter_ = 0
        self.mode_pr_ = Mode.PR
        self.mode_machine_ = 0

        ChannelFactoryInitialize(0, config['network_interface'])

        # create publisher #
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()
        # create subscriber # 
        self.lowstate_subscriber_ = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber_.Init(self.LowStateHandler, 10)

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()
        status, result = self.msc.CheckMode()
        while result['name']:
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

        crc = CRC()
        low_cmd: LowCmd_ = unitree_hg_msg_dds__LowCmd_()
        self.low_cmd = low_cmd
        self.crc = crc
        self._arm_left = Arm(config['arm'])
        self._arm_right = Arm(config['arm'], False)

        self._leg_left = Leg(config['leg'])
        self._leg_right = Leg(config['leg'], False)

        self._waist = Waist(config['waist']) 

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg

        if self.update_mode_machine_ == False:
            self.mode_machine_ = self.low_state.mode_machine
            self.update_mode_machine_ = True

    def print_state(self):
        log.info(f"IMU State: {self.low_state.imu_state}")
        log.info(f"Version: {self.low_state.version}")
        log.info(f"Mode PR: {self.low_state.mode_pr}")
        log.info(f"Mode Machine: {self.low_state.mode_machine}")
        log.info(f"Tick: {self.low_state.tick}")
        log.info(f"Wireless Remote: {self.low_state.wireless_remote}")
        log.info(f"Reserve: {self.low_state.reserve}")
        log.info(f"CRC: {self.low_state.crc}")
        log.info('------------------left arm--------------------------')
        self._arm_left.print_state(self.low_state)
        log.info('------------------right arm--------------------------')
        self._arm_right.print_state(self.low_state)
        log.info('------------------left leg--------------------------')
        self._leg_left.print_state(self.low_state)
        log.info('------------------right leg--------------------------')
        self._leg_right.print_state(self.low_state)
        log.info('------------------waist--------------------------')
        self._waist.print_state(self.low_state)

    def arm_left(self) -> Arm:
        return self._arm_left
    
    def arm_right(self) -> Arm:
        return self._arm_right
    
    def Start(self):
      self.lowCmdWriteThreadPtr = RecurrentThread(
          interval=self.control_dt_, target=self.LowCmdWrite, name="control"
      )
      while self.update_mode_machine_ == False:
          time.sleep(1)
      if self.update_mode_machine_ == True:
          self.lowCmdWriteThreadPtr.Start()
  
    def LowCmdWrite(self):
        self.time_ += self.control_dt_
        # [Stage 1]: set robot to zero posture
        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine_
        ratio = np.clip(self.time_ / self.duration_, 0.0, 1.0)
        self._arm_left.LowCmdUpdate(self.low_cmd, self.low_state, ratio)
        self._arm_right.LowCmdUpdate(self.low_cmd, self.low_state, ratio)
        
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)

    def PreMove(self):
        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine_

        ratio = np.clip(self.time_ / self.duration_, 0.0, 1.0)
        self._arm_left.LowCmdUpdate(self.low_cmd, self.low_state, ratio)
        self._arm_right.LowCmdUpdate(self.low_cmd, self.low_state, ratio)

    def LowCmdApply(self):
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)
    

