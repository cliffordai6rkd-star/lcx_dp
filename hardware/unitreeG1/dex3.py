from hardware.base.hand import HandBase
import glog as log
import time
import sys

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

import numpy as np

G1_NUM_MOTOR = 29

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

class G1JointIndex:
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11
    WaistYaw = 12
    WaistRoll = 13        # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistA = 13           # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistPitch = 14       # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistB = 14           # NOTE: INVALID for g1 23dof/29dof with waist locked
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20   # NOTE: INVALID for g1 23dof
    LeftWristYaw = 21     # NOTE: INVALID for g1 23dof
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27  # NOTE: INVALID for g1 23dof
    RightWristYaw = 28    # NOTE: INVALID for g1 23dof

class Mode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints

class Dex3(HandBase):
    def HandStateHandler(self, msg: HandState_):
        self.hand_state = msg

        if self.update_mode_machine_ == False:
            self.mode_machine_ = self.hand_state.mode_machine
            self.update_mode_machine_ = True
        
        self.counter_ +=1
        if (self.counter_ % 500 == 0) :
            self.counter_ = 0
            print(f"Motor State: {self.hand_state.motor_state}")
            print(f"Press Sensor State: {self.hand_state.press_sensor_state}")
            print(f"Power Voltage: {self.hand_state.power_v}")
            print(f"Power Current: {self.hand_state.power_a}")
            print(f"System Voltage: {self.hand_state.system_v}")
            print(f"Device Voltage: {self.hand_state.device_v}")
            print(f"Error: {self.hand_state.error}")
            print(f"Reserve: {self.hand_state.reserve}")

    def __init__(self, isLeft: bool = True):
        super().__init__()
        if isLeft:
            self._hand_id = 0
            self._dds_namespace = "rt/dex3/left"
            self._sub_namespace = "rt/lf/dex3/left/state"
        else:
            self._hand_id = 1
            self._dds_namespace = "rt/dex3/right"
            self._sub_namespace = "rt/lf/dex3/right/state"

        self.time_ = 0.0
        self.control_dt_ = 0.002  # [2ms]
        self.duration_ = 3.0    # [3 s]
        self.counter_ = 0
        self.mode_pr_ = Mode.PR
        self.mode_machine_ = 0
        self.hand_cmd = unitree_hg_msg_dds__HandCmd_()  
        self.hand_state = None 
        self.update_mode_machine_ = False
        self.crc = CRC()

        # create publisher #
        self.handcmd_publisher_ = ChannelPublisher(self._dds_namespace, HandCmd_)
        self.handcmd_publisher_.Init()

        # create subscriber # 
        self.handstate_subscriber = ChannelSubscriber(self._sub_namespace, HandState_)
        self.handstate_subscriber.Init(self.HandStateHandler, 10)

    def HandCmdWrite(self):
        self.time_ += self.control_dt_

        # if self.time_ < self.duration_ :
        #     # [Stage 1]: set robot to zero posture
        #     for i in range(G1_NUM_MOTOR):
        #         ratio = np.clip(self.time_ / self.duration_, 0.0, 1.0)
        #         self.hand_cmd.mode_pr = Mode.PR
        #         self.hand_cmd.mode_machine = self.mode_machine_
        #         self.hand_cmd.motor_cmd[i].mode =  1 # 1:Enable, 0:Disable
        #         self.hand_cmd.motor_cmd[i].tau = 0. 
        #         self.hand_cmd.motor_cmd[i].q = (1.0 - ratio) * self.low_state.motor_state[i].q 
        #         self.hand_cmd.motor_cmd[i].dq = 0. 
        #         self.hand_cmd.motor_cmd[i].kp = Kp[i] 
        #         self.hand_cmd.motor_cmd[i].kd = Kd[i]

        # elif self.time_ < self.duration_ * 2 :
        #     # [Stage 2]: swing ankle using PR mode
        #     max_P = np.pi * 30.0 / 180.0
        #     max_R = np.pi * 10.0 / 180.0
        #     t = self.time_ - self.duration_
        #     L_P_des = max_P * np.sin(2.0 * np.pi * t)
        #     L_R_des = max_R * np.sin(2.0 * np.pi * t)
        #     R_P_des = max_P * np.sin(2.0 * np.pi * t)
        #     R_R_des = -max_R * np.sin(2.0 * np.pi * t)

        #     self.hand_cmd.mode_pr = Mode.PR
        #     self.hand_cmd.mode_machine = self.mode_machine_
        #     self.hand_cmd.motor_cmd[G1JointIndex.LeftAnklePitch].q = L_P_des
        #     self.hand_cmd.motor_cmd[G1JointIndex.LeftAnkleRoll].q = L_R_des
        #     self.hand_cmd.motor_cmd[G1JointIndex.RightAnklePitch].q = R_P_des
        #     self.hand_cmd.motor_cmd[G1JointIndex.RightAnkleRoll].q = R_R_des

        # else :
        #     # [Stage 3]: swing ankle using AB mode
        #     max_A = np.pi * 30.0 / 180.0
        #     max_B = np.pi * 10.0 / 180.0
        #     t = self.time_ - self.duration_ * 2
        #     L_A_des = max_A * np.sin(2.0 * np.pi * t)
        #     L_B_des = max_B * np.sin(2.0 * np.pi * t + np.pi)
        #     R_A_des = -max_A * np.sin(2.0 * np.pi * t)
        #     R_B_des = -max_B * np.sin(2.0 * np.pi * t + np.pi)

        #     self.hand_cmd.mode_pr = Mode.AB
        #     self.hand_cmd.mode_machine = self.mode_machine_
        #     self.hand_cmd.motor_cmd[G1JointIndex.LeftAnkleA].q = L_A_des
        #     self.hand_cmd.motor_cmd[G1JointIndex.LeftAnkleB].q = L_B_des
        #     self.hand_cmd.motor_cmd[G1JointIndex.RightAnkleA].q = R_A_des
        #     self.hand_cmd.motor_cmd[G1JointIndex.RightAnkleB].q = R_B_des
            
        #     max_WristYaw = np.pi * 30.0 / 180.0
        #     L_WristYaw_des = max_WristYaw * np.sin(2.0 * np.pi * t)
        #     R_WristYaw_des = max_WristYaw * np.sin(2.0 * np.pi * t)
        #     self.hand_cmd.motor_cmd[G1JointIndex.LeftWristRoll].q = L_WristYaw_des
        #     self.hand_cmd.motor_cmd[G1JointIndex.RightWristRoll].q = R_WristYaw_des
    

        self.hand_cmd.crc = self.crc.Crc(self.hand_cmd)
        self.handcmd_publisher_.Write(self.hand_cmd)

    def Start(self):
        self.handCmdWriteThreadPtr = RecurrentThread(
            interval=self.control_dt_, target=self.HandCmdWrite, name="control"
        )
        while self.update_mode_machine_ == False:
            time.sleep(1)

        if self.update_mode_machine_ == True:
            self.handCmdWriteThreadPtr.Start()