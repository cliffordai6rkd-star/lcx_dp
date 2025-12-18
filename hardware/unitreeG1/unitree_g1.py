import time, threading, copy
import numpy as np
import glog as log
from collections import deque
from scipy.interpolate import interp1d
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

DEBUG = False

def make_minjerk(q0: np.ndarray, qT: np.ndarray, T: float):
    q0 = np.asarray(q0, dtype=float).reshape(-1)
    qT = np.asarray(qT, dtype=float).reshape(-1)
    if q0.shape != qT.shape:
        raise ValueError("q0 and qT must have same shape")
    if T <= 0:
        raise ValueError("T must be > 0")

    dq = (qT - q0)

    def _u(t):
        t = np.asarray(t, dtype=float)
        return np.clip(t / T, 0.0, 1.0)

    def q(t):
        u = _u(t)
        s = 10*u**3 - 15*u**4 + 6*u**5
        return q0 + dq * s[..., None] if np.ndim(u) else q0 + dq * s

    return q

class UnitreeG1(ArmBase):
    def __init__(self, config):
        self._network_interface = config["network_interface"]
        self._enable_low_level = config.get(f'enable_low_level', False)
        self._control_frequency = config.get("control_frequency", 500)
        self._update_frequency = config.get("update_frequency", 500)
        self._ankle_mode = config.get('ankle_mode', "pr")
        self._actuate_motors = config.get("actuate_motors", True)
        log.info(f'G1 actuate motors: {self._actuate_motors}')
        if self._ankle_mode == "pr":
            self._ankle_mode = Mode.PR
        else: self._ankle_mode = Mode.AB
        self._control_mode = config.get("control_mode", "position")
        self._move_to_start_time = config.get("reset_time", 1.5)
        self._num_command = 60
        self._command_buffer = Buffer(20, self._num_command)
        self._zero_finished = True
        
        # dds related 
        self.counter = 0
        ChannelFactoryInitialize(0, self._network_interface) # dds channel initialization
        self._msc = MotionSwitcherClient()
        # pub_topic = "rt/lowcmd" if self._enable_low_level else "rt/arm_sdk"
        pub_topic = "rt/lowcmd" 
        self._lowcmd_publisher = ChannelPublisher(pub_topic, LowCmd_)
        self._lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._update_mode_machine = False
        self._mode_machine = 0 # G1 型號
        self._low_state_updated = False
        self._low_cmd = unitree_hg_msg_dds__LowCmd_()
        self._low_cmd.mode_pr = self._ankle_mode
        self._low_state = unitree_hg_msg_dds__LowState_()
        self._crc = CRC()
        self._thread_running = True
        
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
        if self._ankle_mode == Mode.PR:
            left_ankle = [G1JointIndex.LeftAnklePitch, G1JointIndex.LeftAnkleRoll]
            right_ankle = [G1JointIndex.RightAnklePitch, G1JointIndex.RightAnkleRoll]
        else: 
            left_ankle = [G1JointIndex.LeftAnkleA, G1JointIndex.LeftAnkleB]
            right_ankle = [G1JointIndex.RightAnkleA, G1JointIndex.RightAnkleB]
        self._leg_joints = [
            G1JointIndex.LeftHipPitch, G1JointIndex.LeftHipRoll, G1JointIndex.LeftHipYaw,
            G1JointIndex.LeftKnee, left_ankle[0], left_ankle[1],
            G1JointIndex.RightHipPitch, G1JointIndex.RightHipRoll, G1JointIndex.RightHipYaw,
            G1JointIndex.RightKnee, right_ankle[0], right_ankle[1],
        ]
        
        self._robot_id = self._arm_joints
        if self._enable_low_level:
            self._robot_id = self._leg_joints + self._waist_joints + self._robot_id
        
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
        total_dof = len(self._robot_id)
        self._joint_states.set_state_dof(total_dof)
        
        # dds sub and pub
        self._last_read_time = time.perf_counter()
        self._lowcmd_publisher.Init()
        # self._lowstate_subscriber.Init(self._LowStateHandler, 10)
        self._lowstate_subscriber.Init()

        # low state read thread
        self._subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self._subscribe_thread.start()
        self._low_state_updated = False
        while not self._low_state_updated:
            time.sleep(0.001)
        self._low_cmd.mode_machine = self._mode_machine
        log.info(f'Get the low state from the unitree g1 with the version {self._mode_machine}')
        # update command for not used joints
        update_init_position = True if self._init_joint_positions is None else False
        if update_init_position: self._init_joint_positions = np.zeros(len(self._robot_id))
        for i, jid in enumerate(self._arm_joints + self._waist_joints + self._leg_joints):
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
            if update_init_position:
                self._init_joint_positions[i] = self._low_state.motor_state[jid].q
        log.info(f'Unitree G1 low command initialized to current state!!')
        
        # low command writer thread
        self._last_write_time = None
        # self._low_command_writer_thread = RecurrentThread(
        #     interval=0.1/self._control_frequency, target=self._LowCommandWriter, name="control"
        # )
        # self._low_command_writer_thread.Start()
        self._publish_thread = threading.Thread(target=self._ctrl_motor_state, daemon=True)
        self._publish_thread.start()
        while self._last_write_time is None:
            time.sleep(0.001)

        self._is_initialized = True
        self.move_to_start()
        time.sleep(1.0)

        log.info(f'Unitree G1 with dds network {self._network_interface} successfully initialized!')
        return True
    
    def update_arm_states(self):
        if not self._low_state_updated:
            log.warn(f'Unitree g1 low state is still not updated with {self._low_state_updated}')
            return 
            
        with self._lock:
            for id, joint_id in enumerate(self._robot_id):
                # log.info(f'{self._joint_states._positions} {id}: joint id: {joint_id}, {self._low_state.motor_state[joint_id]}')
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
        
        g1_command = np.zeros(self._num_command)
        for id, joint_id in enumerate(self._robot_id):
            g1_command[joint_id] = command[id]

        # with forward tau
        if len(command) > len(self._robot_id):
            tau_command = command[-len(self._robot_id):]
            tau = np.zeros(30)
            tau[joint_id] = tau_command[id]
            g1_command[-30:] = tau
            g1_command[-1] = 1.5
        self._command_buffer.push_data(g1_command, time.perf_counter())
    
    def close(self):
        self._thread_running = False
        if hasattr(self, "_subscribe_thread"):
            if self._subscribe_thread.is_alive():
                self._subscribe_thread.join()

        if hasattr(self, "_publish_thread"):
            if self._publish_thread.is_alive():
                self._publish_thread.join()
                
        self._lowcmd_publisher.Close()
        self._lowstate_subscriber.Close()
    
    def move_to_start(self, joint_commands = None):
        self._command_buffer.clear()
        self._control_mode = "position"
        with self._lock:
            current_joint_positions = copy.deepcopy(self._joint_states._positions)
        if joint_commands:
           target_command = joint_commands
        else:
            if self._init_joint_positions is None:
                target_command = np.zeros_like(current_joint_positions)
            else: target_command = self._init_joint_positions
        
        # smoothing to the target
        q_func = make_minjerk(current_joint_positions, target_command, self._move_to_start_time)
        t = 0.0; stop = False
        while not stop:
            t += 1.2 / self._control_frequency
            smoothed_q = q_func(t)
            if t >= self._move_to_start_time:
                smoothed_q = target_command
                stop = True
            self.set_joint_command(self._control_mode, smoothed_q)
            time.sleep(0.001)
        
        self._zero_finished = False
        while not self._zero_finished:
            time.sleep(0.001)
    
    def _subscribe_motor_state(self):
        log.info(f'Started unitree g1 state update loop with!!!!')
        
        self._last_read_time = time.perf_counter()
        target_dt = 1.0 / self._update_frequency
        counter = 0
        while self._thread_running:
            start = time.perf_counter()
            self._low_state = self._lowstate_subscriber.Read()
            read_time = time.perf_counter() - start
            self._low_state_updated = True
            self.update_arm_states()

            if self._update_mode_machine == False:
                self._mode_machine = self._low_state.mode_machine
                self._update_mode_machine = True

            used_time = time.perf_counter() - self._last_read_time
            self._last_read_time = time.perf_counter()
            if used_time < target_dt:
                sleep_time = target_dt - used_time
                # log.info(f'sleep time: {sleep_time*1000:.1f}ms')
                time.sleep(sleep_time)
            elif used_time > 1.3 * target_dt:
                counter += 1
                if counter %1000 == 0:
                    log.info(f'Unitree G1 state update slow, expected: {target_dt*1000:.1f}ms actual: {used_time*1000:.1f}ms read: {read_time*1000:.1f}ms')
                    counter = 0        
        log.info(f'Stopped unitree g1 state update loop!!!')

    def _ctrl_motor_state(self):
        log.info(f'Started unitree g1 ctrl loop!!!!')
        
        target_dt = 1.0 / self._control_frequency    
        counter = 0
        while self._thread_running:
            if self._last_write_time is None: self._last_write_time = time.perf_counter()
            
            # start = time.perf_counter()
            sucess = False
            if self._command_buffer.size() > 0:
                sucess = True; command =  self._command_buffer._data[0]
            
            if sucess:
                self._low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q =  1.0 # 1:Enable arm_sdk, 0:Disable arm_sdk
                tau = np.zeros(30)
                if command[-1] > 1:
                    tau = command[-30:]
                for i, joint in enumerate(self._robot_id):
                    self._low_cmd.motor_cmd[joint].mode = 1 # 1 for enable 0 for disable
                    self._low_cmd.motor_cmd[joint].tau = 0. 
                    self._low_cmd.motor_cmd[joint].q = 0
                    self._low_cmd.motor_cmd[joint].dq = 0. 
                    # self._low_cmd.motor_cmd[joint].kp = self._kp 
                    # self._low_cmd.motor_cmd[joint].kd = self._kd
                    if self._control_mode == "position":
                        self._low_cmd.motor_cmd[joint].q = command[joint]
                        self._low_cmd.motor_cmd[joint].tau = tau[joint]
                    elif self._control_mode == "torque":
                        self._low_cmd.motor_cmd[joint].tau = command[joint]
                    else:
                        raise ValueError(f'The unitree g1 motor do not support the mode {self._control_mode}')
                _, command, _ = self._command_buffer.pop_data()
            # other_time = time.perf_counter() - start

            start = time.perf_counter()
            if self._actuate_motors:
                self._low_cmd.crc = self._crc.Crc(self._low_cmd)
                self._lowcmd_publisher.Write(self._low_cmd)
            write_time = time.perf_counter() - start
            
            if not self._zero_finished:
                self._zero_finished = True
            
            dt = time.perf_counter() - self._last_write_time
            self._last_write_time = time.perf_counter()
            if dt < target_dt:
                sleep_time = target_dt - dt
                time.sleep(sleep_time)
            elif dt > 1.35*target_dt:
                counter += 1
                if counter %1000 == 0:
                    log.warn(f'Unitree G1 control frequency is slow for writing, expected: {self._control_frequency}, actual: {1.0/dt:.2f} write {1.0/write_time:.2f}Hz')
                    counter = 0
            elif dt > 10*target_dt:
                # release the control and quit, @TODO:
                log.warn(f'Too slow control freq for unitree g1')
                return 
            
        log.info(f'Stopped unitree g1 ctrl loop!!!!')
