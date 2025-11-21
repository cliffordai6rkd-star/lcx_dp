from __future__ import annotations

from hardware.base.arm import ArmBase
import numpy as np

# Try to import panda_py, fall back to mock if not available
try:
    from panda_py import controllers
    # import panda_py.constants
    from panda_py import Panda
except (ImportError, FileNotFoundError):
    import glog as log
    log.warning("panda_py not available, using mock implementation")
    from hardware.mocks.mock_panda_py import controllers, Panda

import threading
import time
from scipy.spatial.transform import Rotation as R
import glog as log

class Fr3Arm(ArmBase):
    def __init__(self, config):
        self._ip = config["ip"]
        log.info(f'Fr3 arm with ip {self._ip} is initializing!!!')
        self._fr3_robot = Panda(self._ip)
        self._damping = config.get("damping", None)
        self._stiffness = config.get("stiffness", None)
        self._filter_coefficient = config.get("filter_coeff", None)
        self._collision_behaviour = config.get("collision_behaviour", None)
        self._control_mode = None
        self._need_recover = False
        self._controller_lock = threading.Lock()
        self._thread = threading.Thread(target=self.update_robot_state_thread)
        self._last_velocities = np.zeros(7)
        self._fr3_state = None
        self._fr3_state_update_flag = False
        self._thread_running = True
        self._recovery_occurred = False  # Track if recovery happened
        
        # init
        super().__init__(config)
    
    def update_robot_state_thread(self):
        read_frequency = 500
        read_dt = 1.0 / read_frequency 
        
        last_read_time = time.perf_counter()
        while self._thread_running:
            dt = time.perf_counter() - last_read_time
            last_read_time = time.perf_counter()
            
            # diff acceleration
            with self._controller_lock:
                self._fr3_state = self._fr3_robot.get_state()
            state_reading_time = time.perf_counter() - last_read_time
            self._fr3_state_update_flag = True
            self._joint_states._accelerations = (np.array(self._fr3_state.dq) - self._last_velocities) / dt
            self._last_velocities = np.array(self._fr3_state.dq)

            # update state
            with self._lock:
                self.update_arm_states()
            
            # @TODO: check why the reading thread is slow
            if dt < read_dt:
                time.sleep(read_dt - dt)
            # elif dt > 2.0 / read_frequency:
            #     log.warn(f"Reading fr3 robot {self._ip} state is slower than the read frequency "
            #         f"{read_frequency}Hz, actual: {1.0 / dt}Hz, reading freq: {1.0/state_reading_time}Hz")
        log.info(f'Fr3 with ip {self._ip} stopped its thread!!!')
            
    def initialize(self, restart_controller=False):
        # Stop any existing controller to avoid concurrent operation errors
        # if self._control_mode is not None:
        #     self._fr3_robot.stop_controller()
        
        if self._control_mode is None:
            self._panda_py_controller = controllers.JointPosition()
            self._control_mode = "position"

        # pandapy low level controller parameters
        if not self._damping is None and \
            not isinstance(self._panda_py_controller, controllers.PureTorque): 
            self._panda_py_controller.set_damping(self._damping)
        if not self._stiffness is None and \
            not isinstance(self._panda_py_controller, controllers.PureTorque):
            self._panda_py_controller.set_stiffness(self._stiffness)
        if not self._filter_coefficient is None and \
            not isinstance(self._panda_py_controller, controllers.IntegratedVelocity) and \
            not isinstance(self._panda_py_controller, controllers.PureTorque):
            self._panda_py_controller.set_filter(self._filter_coefficient)
        # controller start
        flag = "initialized" if not self._is_initialized else "updated"
        if not self._is_initialized or restart_controller:
            with self._controller_lock:
                self._fr3_robot.start_controller(self._panda_py_controller) 
            log.info(f'Started fr3 controller for {flag}')

        if not self._is_initialized:
            if self._collision_behaviour is not None:
                self.set_collision_threshold(self._collision_behaviour["torque_min"],
                                             self._collision_behaviour["torque_max"],
                                             self._collision_behaviour["force_min"],
                                             self._collision_behaviour["force_max"])
            self._thread.start()
            
        self._fr3_state_update_flag = False
        while not self._fr3_state_update_flag:
            time.sleep(0.001)
        log.info(f'Fr3 robot with ip {self._ip} is successfully {flag}!!!')
        return True
    
    def close(self):
        self._thread_running = False

        if hasattr(self, '_thread') and self._thread.is_alive():
            self._thread.join(timeout=2.0) # 給予2秒的等待時間

        self._fr3_robot.stop_controller()
        log.info(f'Fr3 robot with ip {self._ip} is closed!!')        
                   
    def update_arm_states(self):
        if not self._fr3_state_update_flag:
            log.warn(f'The fr3 state is still not ready for robot state to update!')
            return 
        
        self._joint_states._positions = np.array(self._fr3_state.q)
        # log.info(f'posi: {self._joint_states._positions}')
        self._joint_states._velocities = np.array(self._fr3_state.dq)
        self._joint_states._torques = np.array(self._fr3_state.tau_J)
        self._joint_states._time_stamp = time.perf_counter()
        
        # tcp_pose = np.array(self._fr3_state.O_T_EE)
        # tcp_pose = np.reshape(tcp_pose, (4, 4))
        # self._tcp_pose[:3] = tcp_pose[:3, 3]
        # self._tcp_pose[3:] = R.from_matrix(tcp_pose[:3, :3]).as_quat()
    
    def set_joint_command(self, mode, command):
        if self._need_recover:
            log.warn(f'Fr3 with {self._ip} is still in the recover mode!!')
            return False

        if not self._is_initialized:
            log.warn(f'Fr3 with {self._ip} is still waiting to be initialized!!')
            return False
        
        # controller setting or controller change
        if self._control_mode != mode:
            if mode == 'position':
                self._panda_py_controller = controllers.JointPosition()
            elif mode == 'velocity':
                self._panda_py_controller = controllers.IntegratedVelocity()
            elif mode == 'torque':
                self._panda_py_controller = controllers.PureTorque()
            else:
                log.warn(f"Unsupported control mode: {mode}")
                return False
            log.info(f're-initialize in set joint command with mode switching from {self._control_mode} {type(self._control_mode)} to {mode} {type(mode)}')
            self._control_mode = mode
            self._is_initialized = self.initialize(True)
            if not self._is_initialized:
                raise ValueError(f'Failed to reset controller for fr3 with ip {self._ip}')

        # set command, checking
        if len(command) != self._dof:
            log.warn(f"the command dimension does not match with the arm dof: "
                    f"expect: {self._dof}, get: {len(command)}")
            return False
        
        # checking error state
        if self.recover():
            log.warn(f'The robot falls in error state and finished recover!!!')
            time.sleep(0.3)
            self._fr3_state_update_flag = False
            while not self._fr3_state_update_flag:
                time.sleep(0.001)
            log.info(f'Successfully finished the reset procedures!!!!')
            # time.sleep(0.5)
            self._need_recover = False
            return False
        
        # set command
        with self._controller_lock:
            if mode == 'position':
                self.set_joint_position(command)
            elif mode == 'velocity':
                self.set_joint_velocity(command)
            elif mode == 'torque':
                self.set_joint_torque(command)
            else:
                log.warn(f'fr3 did not support control mode {mode}')
                return False
        return True
            
    def set_joint_position(self, position):
        self._panda_py_controller.set_control(position)
    
    def set_joint_velocity(self, velocity):
        self._panda_py_controller.set_control(velocity)
    
    def set_joint_torque(self, torque):
        self._panda_py_controller.set_control(torque)

    def set_teaching_mode(self, enable_taching: bool):
        self._fr3_robot.teaching_mode(enable_taching)

    def get_ee_pose(self):
        """
            return the end effector pose in the homogenous format
        """
        pose = np.array(self._fr3_robot.get_pose())
        return pose
    
    def stop_controller(self):
        """
            stop the controller
        """
        with self._controller_lock:
            self._fr3_robot.stop_controller()
        self._control_mode = None
        
    def recover(self) -> bool:
        """
            recover the robot from error state
            return: if robot need to recover then return true else false
        """
        # check whether need to recover
        try:
            with self._controller_lock:
                self._fr3_robot.raise_error()
            return False
        except:
            self._need_recover = True
            log.info(f'stop controller in recover')
            self.stop_controller()
            # Only acquire lock for the actual recover call
            self._controller_lock.acquire()
            try:
                self._fr3_robot.recover()
            finally:
                self._controller_lock.release()
            log.info(f'Initialize in recover mode')
            self.initialize(True)
            self._recovery_occurred = True  # Mark that recovery happened
            return True
    
    def move_to_start(self, joint_commands = None):
        if self._init_joint_positions is None \
            and joint_commands is None:
            # Stop controller before move_to_start to avoid concurrent operation
            log.info(f'stop controller in move2start')
            self.stop_controller()
            
            # Only acquire lock for the actual move_to_start call
            self._controller_lock.acquire()
            try:
                self._fr3_robot.move_to_start()
            finally:
                self._controller_lock.release()
        else:
            if joint_commands is not None:
                if len(joint_commands) != 7:
                    log.warn(F'The reset joint position len is not 7 but got {len(joint_commands)}')
                    return 
                command = joint_commands
            else:
                command = self._init_joint_positions
            self.set_joint_command('position', command)

    def check_and_clear_recovery_flag(self) -> bool:
        """
        Check if recovery occurred and clear the flag
        Returns:
            True if recovery occurred since last check
        """
        if self._recovery_occurred:
            self._recovery_occurred = False
            return True
        return False
    
    def set_collision_threshold(self, torque_min: float | list[float],
                                torque_max: float | list[float],
                                force_min: float | list[float],
                                force_max: float | list[float]):
        if not isinstance(torque_min, list):
            torque_min = [torque_min] * 7
        elif len(torque_min) != 7:
            log.warn(f'The collision threshold for fr3 {self._ip} failed to be updated!')
            return False
        if not isinstance(torque_max, list):
            torque_max = [torque_max] * 7
        elif len(torque_max) != 7:
            log.warn(f'The collision threshold for fr3 {self._ip} failed to be updated!')
            return False
        
        if not isinstance(force_min, list):
            force_min = [force_min] * 6
        elif len(force_min) != 6:
            log.warn(f'The collision threshold for fr3 {self._ip} failed to be updated!')
            return False
        if not isinstance(force_max, list):
            force_max = [force_max] * 6
        elif len(force_max) != 6:
            log.warn(f'The collision threshold for fr3 {self._ip} failed to be updated!')
            return False
        
        self._fr3_robot.get_robot().set_collision_behavior(torque_min, torque_max,
            torque_min, torque_max, force_min, force_max, force_min, force_max)
        

if __name__ == '__main__':
    import yaml
    import os
    config = None
    cur_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(cur_path, 'config', 'fr3_cfg.yaml')
    log.info(f'cfg file name: {cfg_file}')
    with open(cfg_file, 'r') as stream:
        config = yaml.safe_load(stream)
    log.info(f'yaml data: {config}')
    # fr3 = Fr3Arm(config['fr3'])
    
    # test_mode = "position"
    # test_times = 10
    # counter = 0
    # joint_id = 6
    # test_value = 0.1
    # init_state = fr3.get_joint_states()
    # while counter <= test_times:
    #     joint_state = fr3.get_joint_states()
    #     # log.info(f'joint acc: {joint_state._accelerations}')
    #     log.info(f'tor: {joint_state._torques}')
    #     # fr3.log.info_state()
    #     if test_mode == "position":
    #         joint_command = init_state._positions
    #         if counter % 2 == 0:
    #             joint_command[joint_id] +=  test_value
    #         else:
    #             joint_command[joint_id] -= test_value
    #         init_state._positions = joint_command
    #     elif test_mode == "torque":
    #         joint_command = init_state._torques
    #         if counter % 2 == 0:
    #             joint_command[joint_id] = test_value
    #         else:
    #             joint_command[joint_id] = -test_value
    #     else:
    #         raise ValueError(f"Not support for mode {test_mode}")
        
    #     log.info(f'command: {joint_command}, mode: {test_mode}')
    #     fr3.set_joint_command(test_mode, joint_command)
        
    #     counter += 1     
    #     time.sleep(0.1)
        
    # read test
    fr3 = Panda('192.168.1.206')
    test_nums = 50000
    total_time = 0.0
    for i in range(test_nums):
        start = time.perf_counter()
        state = fr3.get_state()
        used_time = time.perf_counter() - start
        total_time += used_time
        log.info(f'state read used time {used_time * 1000}ms, freq: {1.0 / used_time}')
    log.info(f'avg freq: {test_nums / total_time}Hz')
    log.info('Exit the testing of fr3 arm!!!!')
    