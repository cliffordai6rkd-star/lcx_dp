import time
import numpy as np
import glog as log
from collections import deque
from hardware.base.utils import Buffer
from hardware.base.arm import ArmBase
from hardware.unitreeG1.consts import G1_JOINTS_KP, G1_JOINTS_KD, Mode, G1JointIndex, G1_WRIST_MOTORS, G1_WEAK_MOTORS

# Try to import unitree_sdk2py, fall back to mock if not available
try:
    from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_, unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.utils.thread import RecurrentThread
    from unitree_sdk2py.utils.crc import CRC
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
except (ImportError, ModuleNotFoundError):
    log.warning("unitree_sdk2py not available, using mock implementation")
    from hardware.mocks.mock_unitree_sdk2py import core, rtc, idl, utils
    ChannelSubscriber = core.channel.ChannelSubscriber
    ChannelPublisher = core.channel.ChannelPublisher
    ChannelFactoryInitialize = core.channel.ChannelFactoryInitialize
    # Create mock classes for the other imports
    class MockLowCmd:
        def __init__(self):
            self.motor_cmd = [type('obj', (object,), {'q': 0, 'dq': 0, 'tau': 0, 'kp': 0, 'kd': 0, 'mode': 0})() for _ in range(35)]
            self.crc = 0
    class MockLowState:
        def __init__(self):
            self.motor_state = [type('obj', (object,), {'q': 0, 'dq': 0, 'ddq': 0, 'tau_est': 0})() for _ in range(35)]
            self.mode_machine = 0
            self.imu_state = type('obj', (object,), {'rpy': [0, 0, 0]})()
    unitree_hg_msg_dds__LowCmd_ = MockLowCmd
    unitree_hg_msg_dds__LowState_ = MockLowState
    LowCmd_ = MockLowCmd
    LowState_ = MockLowState
    class MockRecurrentThread:
        def __init__(self, interval, target, name):
            pass
        def Start(self):
            pass
    RecurrentThread = MockRecurrentThread
    class MockCRC:
        def Crc(self, cmd):
            return 0
    CRC = MockCRC
    class MockMotionSwitcherClient:
        def SetTimeout(self, timeout):
            pass
        def Init(self):
            pass
        def CheckMode(self):
            return True, {'name': ''}
        def ReleaseMode(self):
            pass
    MotionSwitcherClient = MockMotionSwitcherClient

class UnitreeG1(ArmBase):
    def __init__(self, config):
        self._network_interface = config["network_interface"]
        self._enable_low_level = config.get(f'enable_low_level', False)
        self._control_frequency = config.get("control_frequency", 500)
        self._update_frequency = config.get("update_frequency", 800)
        self._ankle_mode = config.get('ankle_mode', "pr")
        if self._ankle_mode == "pr":
            self._ankle_mode = Mode.PR
        else: self._ankle_mode = Mode.AB
        self._control_mode = config.get("control_mode", "position")
        self._command_buffer = Buffer(20, 30)
        self._zero_finished = True
        
        # dds related 
        self.counter = 0
        ChannelFactoryInitialize(0, self._network_interface) # dds channel initialization
        self._msc = MotionSwitcherClient()
        pub_topic = "rt/lowcmd" if self._enable_low_level else "rt/arm_sdk"
        self._lowcmd_publisher = ChannelPublisher(pub_topic, LowCmd_)
        self._lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._update_mode_machine = False
        self._mode_machine = 0 # G1 型號
        self._low_state_updated = False
        self._low_cmd = unitree_hg_msg_dds__LowCmd_()
        self._low_state = unitree_hg_msg_dds__LowState_()
        self._crc = CRC()
        
        self._arm_joints = [
          G1JointIndex.LeftShoulderPitch,  G1JointIndex.LeftShoulderRoll,
          G1JointIndex.LeftShoulderYaw,    G1JointIndex.LeftElbow,
          G1JointIndex.LeftWristRoll,      G1JointIndex.LeftWristPitch,
          G1JointIndex.LeftWristYaw,
          G1JointIndex.RightShoulderPitch, G1JointIndex.RightShoulderRoll,
          G1JointIndex.RightShoulderYaw,   G1JointIndex.RightElbow,
          G1JointIndex.RightWristRoll,     G1JointIndex.RightWristPitch,
          G1JointIndex.RightWristYaw,]
        self._waist_joints = [G1JointIndex.WaistYaw, G1JointIndex.WaistRoll,
                        G1JointIndex.WaistPitch]
        self._leg_joints = [
            G1JointIndex.LeftHipPitch, G1JointIndex.LeftHipRoll, G1JointIndex.LeftHipYaw,
            G1JointIndex.LeftKnee, G1JointIndex.LeftAnklePitch, G1JointIndex.LeftAnkleRoll,
            G1JointIndex.RightHipPitch, G1JointIndex.RightHipRoll, G1JointIndex.RightHipYaw,
            G1JointIndex.RightKnee, G1JointIndex.RightAnklePitch, G1JointIndex.RightAnkleRoll,
        ]
        
        self._robot_id = self._arm_joints
        if self._enable_low_level:
            self._robot_id = self._leg_joints + self._waist_joints + self._robot_id
        total_dof = len(self._robot_id)
        self._joint_states.set_state_dof(total_dof)
        
        self._kp_high = 300.0
        self._kd_high = 3.0
        self._kp_low = 80.0
        self._kd_low = 3.0
        self._kp_wrist = 40.0
        self._kd_wrist = 1.5

        super().__init__(config)
        
    def initialize(self):
        if self._is_initialized:
            return True
        
        self._msc.SetTimeout(5.0)
        self._msc.Init()
        
        status, result = self._msc.CheckMode()
        while result['name']:
            self._msc.ReleaseMode()
            status, result = self._msc.CheckMode()
            time.sleep(1)
        log.info(f'Passed the check mode of motion service client for unitree G1')
        
        # dds sub and pub
        self._lowcmd_publisher.Init()
        self._lowstate_subscriber.Init(self._LowStateHandler, 10)
        
        # low command writer thread
        self._low_state_updated = False
        while not self._low_state_updated:
            time.sleep(0.001)
        log.info(f'Get the low state from the unitree g1')
        # update command for not used joints
        for jid in (self._arm_joints + self._waist_joints + self._leg_joints):
            self._low_cmd.motor_cmd[jid].mode = 1
            if jid in self._arm_joints:
                if jid in G1_WRIST_MOTORS:
                    self._low_cmd.motor_cmd[jid].kp = self._kp_wrist
                    self._low_cmd.motor_cmd[jid].kd = self._kd_wrist
                else:
                    self._low_cmd.motor_cmd[jid].kp = self._kp_low
                    self._low_cmd.motor_cmd[jid].kd = self._kd_low
            else:
                if jid in G1_WEAK_MOTORS:
                    self._low_cmd.motor_cmd[jid].kp = self._kp_low
                    self._low_cmd.motor_cmd[jid].kd = self._kd_low
                else:
                    self._low_cmd.motor_cmd[jid].kp = self._kp_high
                    self._low_cmd.motor_cmd[jid].kd = self._kd_high
            self._low_cmd.motor_cmd[jid].q = self._low_state.motor_state[jid].q
        log.info(f'Unitree G1 low command initialized to current state!!')
        
        self._last_write_time = time.perf_counter()
        self._low_command_writer_thread = RecurrentThread(
            interval=1.0/self._control_frequency, target=self._LowCommandWriter, name="control"
        )
        self._low_command_writer_thread.Start()
        
        log.info(f'Unitree G1 with dds network {self._network_interface} successfully initialized!')
        return True
    
    def update_arm_states(self):
        if not self._is_initialized:
            log.warn(f'Unitree g1 is still not initialized for update joint state')
            
        with self._lock:
            for id, joint_id in enumerate(self._robot_id):
                self._joint_states._positions[id] = self._low_state.motor_state[joint_id].q
                self._joint_states._velocities[id] = self._low_state.motor_state[joint_id].dq
                self._joint_states._accelerations[id] = self._low_state.motor_state[joint_id].ddq
                self._joint_states._torques[id] = self._low_state.motor_state[joint_id].tau_est
            self._joint_states._time_stamp = time.perf_counter()
    
    def set_joint_command(self, mode, command):
        if not self._is_initialized:
            log.warn(f'Unitree g1 is still not initialized for setting joint command')
        
        if not self._zero_finished:
            return 
        
        if isinstance(mode, list):
            self._control_mode = mode[0]
        else: self._control_mode = mode
        
        g1_command = np.zeros(30)
        for id, joint_id in enumerate(self._robot_id):
            g1_command[joint_id] = command[id]
        self._command_buffer.push_data(g1_command, time.perf_counter())
    
    def close(self):
        self._lowcmd_publisher.Close()
        self._lowstate_subscriber.Close()
    
    def move_to_start(self):
        self._command_buffer.clear()
        self._control_mode = "position"
        self._command_buffer.push_data(np.zeros(30), time.perf_counter())
        self._zero_finished = False
        while not self._zero_finished:
            time.sleep(0.001)
    
    def _LowStateHandler(self, msg: LowState_):
        self._low_state = msg
        self._low_state_updated = True
        self.update_arm_states()

        if self._update_mode_machine == False:
            self._mode_machine = self.low_state.mode_machine
            self._update_mode_machine = True
        
        self.counter +=1
        if (self.counter % 500 == 0) :
            self.counter = 0
            log.info(self._low_state.imu_state.rpy)
    
    def _LowCommandWriter(self):
        control_time_duration = time.perf_counter() - self._last_write_time
        self._last_write_time = time.perf_counter()
        
        sucess, command, _ = self._command_buffer.pop_data()
        if not sucess:
            return 
        
        if control_time_duration < 1.0 / self._control_frequency:
            self._low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q =  1 # 1:Enable arm_sdk, 0:Disable arm_sdk
            for i,joint in enumerate(self._robot_id):
                self._low_cmd.motor_cmd[joint].mode = 1 # 1 for enable 0 for disable
                self._low_cmd.motor_cmd[joint].tau = 0. 
                self._low_cmd.motor_cmd[joint].q = 0
                self._low_cmd.motor_cmd[joint].dq = 0. 
                # self._low_cmd.motor_cmd[joint].kp = self._kp 
                # self._low_cmd.motor_cmd[joint].kd = self._kd
                if self._control_mode == "position":
                    self._low_cmd.motor_cmd[joint].q = command[joint]
                elif self._control_mode == "torque":
                    self._low_cmd.motor_cmd[joint].tau = command[joint]
                else:
                    raise ValueError(f'The unitree g1 motor do not support the mode {self._control_mode}')
        elif control_time_duration < 1.5 / self._control_frequency:
            log.warn(f'control frequency is slow for unitree g1 writing, expected: {self._control_frequency}, actual: {1.0 / control_time_duration}')
        elif control_time_duration > 10 / self._update_frequency:
            # release the control and quit, @TODO:
            pass
        
        self._low_cmd.crc = self._crc.Crc(self._low_cmd)
        self._lowcmd_publisher.Write(self._low_cmd)
        if not self._zero_finished:
            self._zero_finished = True
            
