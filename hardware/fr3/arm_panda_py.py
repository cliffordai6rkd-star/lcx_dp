# from panda_py import controllers, libfranka, Panda
from typing import Text, Mapping, Any, Callable, Sequence, Union
import os
from hardware.base.arm import ArmBase
import glog as log
import numpy as np
# from panda_py.constants import *

import panda_py
import panda_py.controllers as controllers
import panda_py.constants
Panda = panda_py.Panda
# controllers = panda_py.controllers
# import panda_py.Panda as Panda
# import panda_py.controllers as controllers
from motion.kinematics import PinocchioKinematicsModel as KinematicsModel
from scipy.spatial.transform import Rotation as R
import time
class Arm(ArmBase):
    def __init__(self, ip:str, mode, config: Mapping[Text, Any]):
        super().__init__()
        print(f"=====")
        self.instance = Panda(ip)
        print(f"Robot IP: {ip}")
        self.model = self.instance.get_model()
        if 'velocity' == mode:
            self.controller = controllers.IntegratedVelocity()
        elif 'impedance' == mode:
            self.controller = controllers.CartesianImpedance()
        elif 'position' == mode:
            self.controller = controllers.JointPosition()
        elif 'torque' == mode:
            self.controller = controllers.AppliedTorque()
        elif 'force' == mode:
            self.controller = controllers.AppliedForce()
        elif 'pid_force' == mode:
            self.controller = controllers.Force()
        else:
            raise NotImplementedError
        
        urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', config['urdf_path']))
        base_link = config['base_link']
        end_link = config['end_link']
        self.kinematics = KinematicsModel(urdf_path=urdf_path, base_link=base_link, end_effector_link=end_link)
        self.flange_t_tcp = np.eye(4)
        self.tcp_t_flange = np.linalg.inv(self.flange_t_tcp)
        self.arr_t=[]
        self.arr_r=[]

    def get_model(self):
        return self.model

    def create_context(self, frequency=1e3, max_runtime=1):
        return self.instance.create_context(frequency=frequency, max_runtime=max_runtime)

    def start_controller(self):
        self.instance.start_controller(self.controller)

    # def get_spatial_mass_matrix(self):
    #     return self.model.mass(robot_state=self.instance.get_state())
    def get_flange_pose(self) -> np.ndarray:
        """Gets the pose of the flange.
        """
        # log.info(f"flange pose: {self.kinematics.fk(self.get_joint_positions())}")
        x: np.ndarray = np.array(self.get_state().q)
        x = np.concatenate([x, np.array([0, 0])])
        
        return self.kinematics.fk(x)
    def get_tcp_pose(self) -> np.ndarray:
        return self.get_flange_pose() @ self.flange_t_tcp
    def get_ee_orientation(self):
        """
        Current end-effector orientation in robot base frame.
        """
        orientation = self.instance.get_orientation()
        log.info(f"End Effector Orientation: {orientation}")
        return orientation

    def get_ee_position(self):
        """
        Current end-effector position in robot base frame.
        """
        position = self.instance.get_position()
        log.info(f"End Effector Position: {position}")
        return position
    
    def get_tcp_pose(self):
        """
        Current end-effector pose (position and orientation) in robot base frame.
        """
        pose = self.instance.get_pose()
        log.info(f"End Effector Pose: {pose}")
        return pose

    def get_controller_time(self):
        return self.controller.get_time()
    
    def set_impedance_control(self, position, orientation):
        """
        Set the Cartesian impedance control with position and orientation.
        """
        if not isinstance(self.controller, controllers.CartesianImpedance):
            raise TypeError("Controller is not in impedance mode.")
        self.controller.set_control(position, orientation)
    
    def set_impedance_damping_ratio(self, damping):
        """
        Set the damping ratio for the Cartesian impedance controller.
        """
        if not isinstance(self.controller, controllers.CartesianImpedance):
            raise TypeError("Controller is not in impedance mode.")
        self.controller.set_damping_ratio(damping)
    
    def set_impedance_filter(self, filter_coeff: float):
        """
        Set the filter coefficient for the Cartesian impedance controller.
        """
        if not isinstance(self.controller, controllers.CartesianImpedance):
            raise TypeError("Controller is not in impedance mode.")
        self.controller.set_filter(filter_coeff)
    
    def set_impedance(self, impedance):
        """
        Set the stiffness (impedance) for the Cartesian impedance controller.
        """
        if not isinstance(self.controller, controllers.CartesianImpedance):
            raise TypeError("Controller is not in impedance mode.")
        self.controller.set_impedance(impedance)

    def set_impedance_nullspace_stiffness(self, nullspace_stiffness: float):
        """
        Set the nullspace stiffness for the Cartesian impedance controller.
        """
        if not isinstance(self.controller, controllers.CartesianImpedance):
            raise TypeError("Controller is not in impedance mode.")
        self.controller.set_nullspace_stiffness(nullspace_stiffness)
    
    def get_velocity_qd(self):
        """
        Get the current joint velocities (qd) from the IntegratedVelocity controller.
        """
        if not isinstance(self.controller, controllers.IntegratedVelocity):
            raise TypeError("Controller is not in velocity mode.")
        return self.controller.get_qd()

    def set_velocity_control(self, velocity):
        """
        Set the joint velocity control for the IntegratedVelocity controller.
        """
        if not isinstance(self.controller, controllers.IntegratedVelocity):
            raise TypeError("Controller is not in velocity mode.")
        self.controller.set_control(velocity)

    def set_velocity_damping(self, damping):
        """
        Set the damping for the IntegratedVelocity controller.
        """
        if not isinstance(self.controller, controllers.IntegratedVelocity):
            raise TypeError("Controller is not in velocity mode.")
        self.controller.set_damping(damping)

    def set_velocity_stiffness(self, stiffness):
        """
        Set the stiffness for the IntegratedVelocity controller.
        """
        if not isinstance(self.controller, controllers.IntegratedVelocity):
            raise TypeError("Controller is not in velocity mode.")
        self.controller.set_stiffness(stiffness)

    def set_position_control(self, position, velocity):
        """
        Set the joint position control for the JointPosition controller.
        """
        if not isinstance(self.controller, controllers.JointPosition):
            raise TypeError("Controller is not in position mode.")
        self.controller.set_control(position, velocity)

    def set_position_damping(self, damping):
        """
        Set the damping for the JointPosition controller.
        """
        if not isinstance(self.controller, controllers.JointPosition):
            raise TypeError("Controller is not in position mode.")
        self.controller.set_damping(damping)

    def set_position_stiffness(self, stiffness):
        """
        Set the stiffness for the JointPosition controller.
        """
        if not isinstance(self.controller, controllers.JointPosition):
            raise TypeError("Controller is not in position mode.")
        self.controller.set_stiffness(stiffness)

    def set_position_filter(self, filter_coeff: float):
        """
        Set the filter coefficient for the JointPosition controller.
        """
        if not isinstance(self.controller, controllers.JointPosition):
            raise TypeError("Controller is not in position mode.")
        self.controller.set_filter(filter_coeff)
    

    def set_torque_control(self, torque):
        """
        Set the joint torque control for the AppliedTorque controller.
        """
        if not isinstance(self.controller, controllers.AppliedTorque):
            raise TypeError("Controller is not in torque mode.")
        self.controller.set_control(torque)

    def set_torque_damping(self, damping):
        """
        Set the damping for the AppliedTorque controller.
        """
        if not isinstance(self.controller, controllers.AppliedTorque):
            raise TypeError("Controller is not in torque mode.")
        self.controller.set_damping(damping)

    def set_torque_filter(self, filter_coeff: float):
        """
        Set the filter coefficient for the AppliedTorque controller.
        """
        if not isinstance(self.controller, controllers.AppliedTorque):
            raise TypeError("Controller is not in torque mode.")
        self.controller.set_filter(filter_coeff)

    
    def set_force_control(self, force):
        """
        Set the Cartesian force control for the AppliedForce controller.
        """
        if not isinstance(self.controller, controllers.AppliedForce):
            raise TypeError("Controller is not in force mode.")
        self.controller.set_control(force)

    def set_force_damping(self, damping):
        """
        Set the damping for the AppliedForce controller.
        """
        if not isinstance(self.controller, controllers.AppliedForce):
            raise TypeError("Controller is not in force mode.")
        self.controller.set_damping(damping)

    def set_force_filter(self, filter_coeff: float):
        """
        Set the filter coefficient for the AppliedForce controller.
        """
        if not isinstance(self.controller, controllers.AppliedForce):
            raise TypeError("Controller is not in force mode.")
        self.controller.set_filter(filter_coeff)

    def set_pid_force_control(self, force):
        """
        Set the Cartesian force control for the Force controller.
        """
        if not isinstance(self.controller, controllers.Force):
            raise TypeError("Controller is not in pid_force mode.")
        self.controller.set_control(force)

    def set_pid_force_filter(self, filter_coeff: float):
        """
        Set the filter coefficient for the Force controller.
        """
        if not isinstance(self.controller, controllers.Force):
            raise TypeError("Controller is not in pid_force mode.")
        self.controller.set_filter(filter_coeff)

    def set_pid_force_integral_gain(self, k_i: float):
        """
        Set the integral gain for the Force controller.
        """
        if not isinstance(self.controller, controllers.Force):
            raise TypeError("Controller is not in pid_force mode.")
        self.controller.set_integral_gain(k_i)

    def set_pid_force_proportional_gain(self, k_p: float):
        """
        Set the proportional gain for the Force controller.
        """
        if not isinstance(self.controller, controllers.Force):
            raise TypeError("Controller is not in pid_force mode.")
        self.controller.set_proportional_gain(k_p)

    def print_state(self):
        log.info(self.instance.get_state())
    
    def set_control_mode(self, mode:str):
        self.mode = mode

    def get_state(self):
        return self.instance.get_state()
    
    def move_to_start(self):
        start_time = time.time()
        self.instance.move_to_start()
        elapsed_time = time.time() - start_time
        log.debug(f"Time taken to move to start: {elapsed_time:.2f} seconds")

    def get_joint_target_from_pose(self, target):
        """Gets joint configuration via IK.
        """
        flange_target = target @ self.tcp_t_flange
        return self.ik(flange_target)
    def ik(self, pose):
        return self.kinematics.ik(pose)[:7]

    def fk(self, jp):
        x = np.concatenate([jp, np.array([0, 0])])
        return self.kinematics.fk(x)
    # def move_to_pose(self, target: se3.Transform) -> bool:
    #     """Moves to the target that specifies TCP pose in base frame.
    #     """
    #     return self.move_to_joint_target(
    #     self.get_joint_target_from_pose(target))
    def getRT(self, pose):
        rotation_matrix = pose[0:3, 0:3]

        # 2. Create a Rotation object from the rotation matrix
        rot = R.from_matrix(rotation_matrix)
        euler_angles_deg = rot.as_euler('zyx', degrees=False)
        for i in range(3):
            if euler_angles_deg[i] < -np.pi/2:
                euler_angles_deg[i]+=np.pi*2
        return euler_angles_deg, pose[0:3, 3]
    
    def move_to_pose(self, pose):
        start_time = time.time()
        # self.instance.move_to_pose(pose)
        self.move_to_joint_target(self.get_joint_target_from_pose(pose))
        elapsed_time = time.time() - start_time
        log.debug(f"Time taken to move to pose: {elapsed_time:.2f} seconds")

        # log.info(f"poseOrin :\n{pose} ")
        
        s = self.get_state()
        # log.info(f"[move_to_pose] O_T_EE - O_T_EE_d == {np.array(s.O_T_EE) - np.array(s.O_T_EE_d)}")
        # log.info(f"[move_to_pose] O_T_EE - O_T_EE_c == {np.array(s.O_T_EE) - np.array(s.O_T_EE_c)}")
        r1,t1 = self.getRT(pose)
        r2,t2 = self.getRT(np.array(s.O_T_EE).reshape(4,4).T)
        log.info(f"[move_to_pose] r1== {r1}")
        log.info(f"[move_to_pose] r2== {r2}")

        dt = t2-t1
        dr = r2-r1
        log.info(f"[move_to_pose] dx,dy,dz== {dt}")
        log.info(f"[move_to_pose] dR,dP,dY== {dr}")
        self.arr_t.append(dt)
        self.arr_r.append(dr)

        avg_dt = np.mean(self.arr_t, keepdims=True, axis=0)
        log.info(f"[move_to_pose] average dt ==  {avg_dt}")
        avg_dr = np.mean(self.arr_r, keepdims=True, axis=0)
        log.info(f"[move_to_pose] average dr ==  {avg_dr}")

        div1 = np.var(self.arr_t, ddof=1, keepdims=True, axis=0)
        log.info(f"[move_to_pose] div dt ==  {div1}")

        div2 = np.var(self.arr_r, ddof=1, keepdims=True, axis=0)

        log.info(f"[move_to_pose] div dr ==  {div2}")
        # assert(abs(dt[0])<1.3e-3)
        # assert(abs(dt[1])<1e-3)
        # assert(abs(dt[2])<2.5e-3)

    def move_to_joint_target(self, q):
        start_time = time.time()
        self.instance.move_to_joint_position(q)
        elapsed_time = time.time() - start_time
        log.debug(f"Time taken to move to joint position: {elapsed_time:.2f} seconds")
        s = self.get_state()
        log.info(f"[move_to_joint_target] q - q_d == {np.array(s.q) - np.array(q)}")
    def get_joint_position_start(self):
      return panda_py.constants.JOINT_POSITION_START

    def get_joint_limits_lower(self):
        return panda_py.constants.JOINT_LIMITS_LOWER
    
    def get_joint_limits_upper(self):
        return panda_py.constants.JOINT_LIMITS_UPPER
