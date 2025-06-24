import sys
sys.path.append("../../dependencies/libfranka-python/franka_bindings")
from franka_bindings import (Robot, ControllerMode, JointPositions, JointVelocities, Torques)

from hardware.base.arm import ArmBase
from hardware.fr3.gripper import Gripper
import glog as log
import numpy as np

import time, os
from typing import Text, Mapping, Any, Callable, Sequence, Union
from data_types import se3
from threading import Thread, Lock

USE_KDL = False
if USE_KDL:
    from motion.kinematics_model import KinematicsModel
else:
    from motion.kinematics import PinocchioKinematicsModel as KinematicsModel

from motion import trajectory_planner, trajectory_executor
from tools import file_utils
import numpy as np
import threading, queue

kJointPositionStartData = np.array([
    0.0,
    -np.pi / 4,
    0.0,
    -3 * np.pi / 4,
    0.0,
    np.pi / 2,
    np.pi / 4,
])

def set_default_behavior(robot):
    robot.set_collision_behavior(
        [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0], [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        # [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0], [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        [20.0, 20.0, 20.0, 20.0, 20.0, 20.0], [20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        # [10.0, 10.0, 10.0, 10.0, 10.0, 10.0], [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    )
    robot.set_joint_impedance([3000, 3000, 3000, 2500, 2500, 2000, 2000])
    robot.set_cartesian_impedance([3000, 3000, 3000, 300, 300, 300])

# class TrajectoryGenerator():
    
#     def __init__(self, initial_position, desired_position):
#         self.done = False
#         run_time = self.calcRunTimeAdvanced(initial_position,desired_position)
#         D = desired_position - initial_position
#         T = run_time
#         # 系数公式推导自边界条件：
#         #   t=0: pos=initial, vel=0, acc=0
#         #   t=T: pos=desired, vel=0, acc=0
#         a3 = 10 * D / (T**3)
#         a4 = -15 * D / (T**4)
#         a5 = 6 * D / (T**5)

#         self.points = []
#         elapsed_time = 0
#         while elapsed_time < run_time:
#             # try:
#             #     _, duration = self.read()
#             # except Exception as e:
#             #     print(f"Error reading state: {e}")
#             #     break

#             elapsed_time += 0.01
#             t = min(elapsed_time, run_time)  # 确保不超过总时间
            
#             # 五次多项式位置计算
#             self.points.append(initial_position + a3 * t**3 + a4 * t**4 + a5 * t**5)
        
#         log.info(f"{len(self.points)} points generated!")

#     def calcRunTimeAdvanced(self, src_jp, tar_jp, 
#                        max_vel_per_joint=np.array([2.62,2.62,2.62,2.62,5.26,4.18,5.26]),  # 每个关节的最大速度数组
#                        max_acc_per_joint=np.array([10,10,10,10,10,10,10,])): # 每个关节的最大加速度数组
#         delta_q = np.abs(tar_jp - src_jp)
#         joint_times = []
        
#         for i in range(len(delta_q)):
#             if delta_q[i] < 1e-6:
#                 continue
                
#             t_vel = delta_q[i] / max_vel_per_joint[i]
#             t_acc = np.sqrt(delta_q[i] / max_acc_per_joint[i])
#             joint_times.append(max(t_vel, t_acc))
        
#         return max(joint_times) * 15 if joint_times else 0.0
#     def get_next_point(self):
#         if len(self.points)>0:
#             ret = self.points[0]
#             self.points.pop(0)
#         else:
#             raise Exception("get_next_point FATAL")
#         return ret
        

#     def is_done(self):
#         return len(self.points) == 0

class Arm(ArmBase):
    def __init__(self, config: Mapping[Text, Any]):
        super().__init__()
        self.trajectory_generator = None
        self.robot_in_error = False

        self.qs_prev = None
        self.config = config
        robot = Robot(config['ip'])
        log.info(f"Robot instance created with IP: {config['ip']}")
        self._gripper = Gripper(config=config['gripper'])
        log.info(f"Gripper instance created with IP: {config['ip']}")

        urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', config['urdf_path']))
        base_link = config['base_link']
        end_link = config['end_link']
        joint_names = config['joint_names']

        try:
            if USE_KDL:
                self.kinematics = KinematicsModel(urdf=file_utils.read_file(urdf_path), base_link=base_link, ee_link=end_link, joint_names=joint_names)
            else:
                self.kinematics = KinematicsModel(urdf_path=urdf_path, base_link=base_link, end_effector_link=end_link)

        except Exception as e:
            log.error(f"Failed to load URDF: {e}")

        trajectory_args = {'rate': config['control_rate'],
                       'time_func': time.time,
                       'logging': config.get('log_trajectory', False)}
        if config['control_mode'] == 'position':
            trajectory_args['pos_setter'] = self.set_joint_positions
        elif config['control_mode'] == 'velocity':
            trajectory_args['vel_setter'] = self.set_joint_velocities
        else:
            raise ValueError('control_mode should be one of position or velocity.')
        self.trajectory_executor = trajectory_executor.OpenTrajectoryExecutor(
        **trajectory_args)

        # Set collision behavior
        lower_torque_thresholds = [20.0] * 7  # Nm
        upper_torque_thresholds = [40.0] * 7  # Nm
        lower_force_thresholds = [10.0] * 6   # N (linear) and Nm (angular)
        upper_force_thresholds = [20.0] * 6   # N (linear) and Nm (angular)
        
        # lower_torque_thresholds = [100.0] * 7  # Nm
        # upper_torque_thresholds = [100.0] * 7  # Nm
        # lower_force_thresholds = [100.0] * 6   # N (linear) and Nm (angular)
        # upper_force_thresholds = [100.0] * 6   # N (linear) and Nm (angular)


        robot.set_collision_behavior(
            lower_torque_thresholds,
            upper_torque_thresholds,
            lower_force_thresholds,
            upper_force_thresholds
        )

        # set_default_behavior(robot)

        self.robot = robot

        self.flange_t_tcp = np.eye(4)

        self.tcp_t_flange = se3.Transform(matrix=self.flange_t_tcp).inverse()

        mode = config['control_mode']
        # if 'velocity' == mode:
        #     self.controller = robot.start_joint_velocity_control(ControllerMode.JointImpedance)
        #     log.info(f'cur mode == {mode}')
        # # elif 'impedance' == mode:
        # #     self.controller = controllers.CartesianImpedance()
        # elif 'position' == mode:
        #     self.controller = robot.start_joint_position_control(ControllerMode.JointImpedance)
        #     log.info(f'cur mode == {mode}')
        
        # # elif 'torque' == mode:
        # #     self.controller = controllers.AppliedTorque()
        # # elif 'force' == mode:
        # #     self.controller = controllers.AppliedForce()
        # # elif 'pid_force' == mode:
        # #     self.controller = controllers.Force()
        # else:
        #     raise NotImplementedError

        self.mode = mode
        # log.info(f"controller {self.controller} created!")
        # self.motion_finished = True
        # self.points_lock = threading.Lock() 
        # self.points_queue = queue.Queue(maxsize=50)
        # self.duration = 0
        # self.robot_in_error = False
        # self._lock = threading.Lock()
        # time.sleep(0.1)
        # control_thread = threading.Thread(target=self.control_loop)
        # control_thread.daemon = True
        # control_thread.start()

        
        # time.sleep(0.1)

    def get_robot_in_error(self):
        """Safely get the robot_in_error flag."""
        with self._lock:
            return self.robot_in_error
    def set_robot_in_error(self, error_state: bool):
        """Safely set the robot_in_error flag."""
        with self._lock:
            self.robot_in_error = error_state
            log.warning(f"Robot in error state: {self.robot_in_error}")

    def control_loop(self):
        pass
        # with self._lock:
        #     self.state, self.duration = self.controller.readOnce()


        # _setState(robot_state);
        # franka::Torques tau = franka::Torques({0, 0, 0, 0, 0, 0, 0});
        # if (current_controller_) {
        # current_controller_->setTime(current_controller_->getTime() +
        #                             duration.toSec());
        # tau = current_controller_->step(robot_state, duration);
        # }
        # // Virtual joint walls
        # Array7d tau_virtual_wall, tau_saturated, tau_clipped;
        # virtual_walls_->computeTorque(robot_state.q, robot_state.dq,
        #                             tau_virtual_wall);
        # for (int i = 0; i < 7; i++) {
        # tau.tau_J[i] += tau_virtual_wall[i];
        # }
        # tau_saturated = saturateTorqueRate(tau.tau_J, robot_state.tau_J_d);
        # tau_clipped = clipTorques(tau_saturated);
        # tau.tau_J = tau_clipped;

    def get_joint_positions(self):
        # state, _ = self.controller.readOnce()
        # return np.array(list(state.q))
        # start_time = time.perf_counter_ns()  # 记录起始时间
        log.info(f"readOnce --")
        state, _ = self.controller.readOnce()
        log.info(f"readOnce ++")
        # end_time = time.perf_counter_ns()    # 记录结束时间
        
        # latency_ns = end_time - start_time   # 计算函数执行时间
        # log.info(f"Function latency: {latency_ns / 1e6:.3f} ms")
        
        return np.array(state.q)  # 优化：避免list转换，直接np.array(state.q)
    
    def get_flange_pose(self) -> np.ndarray:
        """Gets the pose of the flange.
        """
        # log.info(f"flange pose: {self.kinematics.fk(self.get_joint_positions())}")
        x: np.ndarray = self.get_joint_positions()
        x = np.concatenate([x, np.array([0, 0])])
        
        return self.kinematics.fk(x)
    
    def set_joint_positions(self, q: np.ndarray, motion_finished = False):
        self.qs_prev = q

        jp = JointPositions(q)
        jp.motion_finished = motion_finished
        
        # log.info(f"set_jp({q})")
        # with self.points__lock:
        #     self.points_queue.put(jp)
        #     log.info(f"self.points_queue.put(jp)")
        # time.sleep(0.001)
        # log.info(f"self.points_queue.put")

        self.controller.writeOnce(jp)
            # time.sleep(self.duration)
        # log.info(f"===={self.controller}, {q}")

    def calcRunTimeAdvanced(self, src_jp, tar_jp, 
                       max_vel_per_joint=np.array([2.62,2.62,2.62,2.62,5.26,4.18,5.26]),  # 每个关节的最大速度数组
                       max_acc_per_joint=np.array([10,10,10,10,10,10,10,])): # 每个关节的最大加速度数组
        delta_q = np.abs(tar_jp - src_jp)
        joint_times = []
        
        for i in range(len(delta_q)):
            if delta_q[i] < 1e-6:
                continue
                
            t_vel = delta_q[i] / max_vel_per_joint[i]
            t_acc = np.sqrt(delta_q[i] / max_acc_per_joint[i])
            joint_times.append(max(t_vel, t_acc))
        
        return max(joint_times) * 15 if joint_times else 0.0
    
    # def move_to_joint_target(self, desired_position):
    #     """设置一个新的运动目标，并创建一个新的轨迹生成器"""
    #     robot_state, _ = self.read()
    #     initial_position = robot_state.q_d if hasattr(robot_state, 'q_d') else robot_state.q
    #     self.trajectory_generator = TrajectoryGenerator(initial_position, desired_position) # 假设5秒
    #     log.info(f"Starting new trajectory to {desired_position}")

    def move_to_joint_target(self, desired_position):
        # self.robot.stop()
        # time.sleep(1)

        self.controller = None
        # self.robot.
        # time.sleep(1)
        if 'position' == self.mode:
            self.controller = self.robot.start_joint_position_control(ControllerMode.JointImpedance)
            log.info(f'cur mode == {self.mode}')
            
        
        robot_state, duration = self.read()
        initial_position = robot_state.q_d if hasattr(robot_state, 'q_d') else robot_state.q
        run_time = self.calcRunTimeAdvanced(initial_position, desired_position)  # 总运动时间
        log.info(f"run time == {run_time}")
        elapsed_time = 0.0

        if np.allclose(initial_position, desired_position, rtol=1e-3, atol=1e-3):
            log.warn(f"Already in pose!")
            return
        # 计算五次多项式系数
        D = desired_position - initial_position
        T = run_time
        # 系数公式推导自边界条件：
        #   t=0: pos=initial, vel=0, acc=0
        #   t=T: pos=desired, vel=0, acc=0
        a3 = 10 * D / (T**3)
        a4 = -15 * D / (T**4)
        a5 = 6 * D / (T**5)

        while elapsed_time < run_time:
            try:
                _, duration = self.read()
            except Exception as e:
                print(f"Error reading state: {e}")
                break

            elapsed_time += duration
            # with self._lock:
            #     elapsed_time += self.duration
            t = min(elapsed_time, run_time)  # 确保不超过总时间
            
            # 五次多项式位置计算
            jp = initial_position + a3 * t**3 + a4 * t**4 + a5 * t**5
            
            try:
                self.set_joint_positions(jp)
                log.info(f"{duration}")
            except Exception as e:
                print(f"Error writing joint positions: {e}")
                continue
        
        # 最终确保到达目标位置
        try:
            self.set_joint_positions(desired_position)
        except Exception as e:
            print(f"Final position error: {e}")

        # self.robot.stop()

    # def move_thru_joint_targets(self, targets: Sequence[np.ndarray],blocking: bool = True) -> bool:
    #     cur_jp = self.get_joint_positions()
    #     log.info('before constructing trajectory from ' +
    #                 ','.join(map(str, cur_jp)) + ' to ' +
    #                 ','.join(map(str, targets[-1])))
    #     trajectory = trajectory_planner.TimeOptimalTrajectoryWrapper(
    #     [cur_jp] + targets, self.config['max_deviation'],
    #     self.config['jvel_limit'], self.config['jacc_limit'])
    #     log.info(f'after constructing trajectory: {trajectory}')
    #     return self.execute_trajectory(
    #     trajectory, timeout=trajectory.get_duration() + 0.5, blocking=blocking)
    
    # def execute_trajectory(
    #     self, trajectory: trajectory_planner.TimeOptimalTrajectoryWrapper,
    #     timeout: float = None, blocking: bool = True) -> bool:
    #     self.trajectory_executor.follow_trajectory(trajectory)
    #     if blocking:
    #         return self.wait_for_trajectory_done(timeout)
    #     else:
    #         return True
        
    # def log_trajectory(self, timestamp: float):
    #     pass
    #     # raise NotImplementedError("log_trajectory not implemented")
    #     # self.time_log.append(timestamp)
    #     # self.jp_log.append(self.arm.get_jp())
    #     # try:
    #     #     self.jv_log.append(self.arm.get_jv())
    #     # except NotImplementedError as e:
    #     #     log.error(e)

    # def wait_for_trajectory_done(self, timeout: float = None) -> bool:
    #     result = self.trajectory_executor.wait(
    #     timeout=timeout, callback=self.log_trajectory)
    #     self.hold_joints()
    #     return result
    
    # def hold_joints(self) -> None:
    #     """Holds current joint position with zero velocity.
    #     """
    #     # TODO: check if this is correct
    #     # cur_jp = self.get_joint_angles()
    #     # self.set_joint_angles(cur_jp)
    #     if self.qs_prev is not None:
    #         self.set_joint_positions(self.qs_prev)

    # def move_to_joint_target(self, target: np.ndarray, blocking: bool = True) -> bool:
    #     """Gets current jp and directly moves to target.

    #     Using the two waypoint trajectory.
    #     """
    #     return self.move_thru_joint_targets([target], blocking)

    def get_joint_target_from_pose(self, target: se3.Transform,
                                start: np.ndarray = None):
        """Gets joint configuration via IK.
        """
        flange_target = target * self.tcp_t_flange
        return self.ik(flange_target._matrix)
    
    def move_to_pose(self, target: se3.Transform) -> bool:
        """Moves to the target that specifies TCP pose in base frame.
        """
        return self.move_to_joint_target(
        self.get_joint_target_from_pose(target))
      
    def move_to_start(self):
        self.move_to_joint_target(kJointPositionStartData)

    def get_joint_velocities(self):
        state, _ = self.controller.readOnce()
        return list(state.dq)
    
    def set_joint_velocities(self, dq: JointVelocities):
        self.controller.writeOnce(dq)

    # TODO: this not work now, we need to use the flange pose
    # def get_tcp_pose(self):
    #     state, _ = self.controller.readOnce()
    #     return np.array(state.O_T_EE).reshape((4, 4)).T

    def get_tcp_pose(self) -> np.ndarray:
        return self.get_flange_pose() @ self.flange_t_tcp
    
    def stop(self):
        self.robot.stop()

    def get_duration(self):
        _, duration = self.controller.readOnce()
        return duration.to_sec()
    def read(self):
        # start_time = time.perf_counter_ns()  # 记录起始时间
        # with self._lock:
        robot_state, duration = self.controller.readOnce()
        # end_time = time.perf_counter_ns()    # 记录结束时间
        
        # latency_ns = end_time - start_time   # 计算函数执行时间
        # log.info(f"Function latency: {latency_ns / 1e6:.3f} ms")
        # log.info(f"Elapsed time: {duration.to_sec():.2f} seconds")
        # log.info(f"Joint Positions (q): {robot_state.q}")
        # log.info(f"Desired Joint Positions (q_d): {robot_state.q_d}")
        # log.info(f"Joint Velocities (dq): {robot_state.dq}")
        # log.info(f"Desired Joint Velocities (dq_d): {robot_state.dq_d}")
        # log.info(f"Joint Torques (tau_J): {robot_state.tau_J}")
        # log.info(f"Desired Joint Torques (tau_J_d): {robot_state.tau_J_d}")
        # log.info(f"End-Effector Pose (O_T_EE): {robot_state.O_T_EE}")
        # log.info(f"Desired End-Effector Pose (O_T_EE_d): {robot_state.O_T_EE_d}")
        # log.info(f"End-Effector Force-Torque (F_T_EE): {robot_state.F_T_EE}")
        # log.info(f"End-Effector to Kinematic Frame (EE_T_K): {robot_state.EE_T_K}")

        return robot_state, duration.to_sec()
    
    def get_gripper(self):
        return self._gripper
    
    def ik(self, pose):
        if not USE_KDL:
            return self.kinematics.ik(pose)[:7]
        else:
            return self.kinematics.ik(se3.Transform(matrix=pose))

    def fk(self, jp):
        return self.kinematics.fk(jp)

    def get_state(self):
        robot_state, duration = self.controller.readOnce()
        return robot_state

    def print_state(self):
        robot_state, duration = self.controller.readOnce()
        log.info(f"Elapsed time: {duration.to_sec():.2f} seconds")
        log.info(f"Joint Positions (q): {robot_state.q}")
        log.info(f"Desired Joint Positions (q_d): {robot_state.q_d}")
        log.info(f"Joint Velocities (dq): {robot_state.dq}")
        log.info(f"Desired Joint Velocities (dq_d): {robot_state.dq_d}")
        log.info(f"Joint Torques (tau_J): {robot_state.tau_J}")
        log.info(f"Desired Joint Torques (tau_J_d): {robot_state.tau_J_d}")
        log.info(f"End-Effector Pose (O_T_EE): {robot_state.O_T_EE}")
        log.info(f"Desired End-Effector Pose (O_T_EE_d): {robot_state.O_T_EE_d}")
        log.info(f"End-Effector Force-Torque (F_T_EE): {robot_state.F_T_EE}")
        log.info(f"End-Effector to Kinematic Frame (EE_T_K): {robot_state.EE_T_K}")

        self._gripper.print_state()
        # time.sleep(2)
        pass
        
