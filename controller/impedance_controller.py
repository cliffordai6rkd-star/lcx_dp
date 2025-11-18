from __future__ import annotations

from controller.controller_base import ControllerBase
from motion.pin_model import RobotModel
import numpy as np
from hardware.base.utils import RobotJointState, compute_pose_diff, convert_homo_2_7D_pose, convert_7D_2_homo
import pinocchio as pin
from hardware.base.utils import matrix_sqrt, scipy_matrix_sqrt

def get_pose_error(pose1, pose2):
    """
        @ pose: np.ndarray (4x4)
        @ output: return pose1 - pose2
    """
    SE3_1 = pin.SE3(pose1[:3,:3], pose1[:3, 3])
    SE3_2 = pin.SE3(pose2[:3,:3], pose2[:3, 3])
    
    T_err = SE3_2.actInv(SE3_1)
    err = pin.log6(T_err).vector
    return err
    
class ImpedanceController(ControllerBase):
    def __init__(self, config, robot_model: RobotModel):
        super().__init__(config, robot_model)
        self._gravity_compensation = config.get("enable_gravity_compensation", True)
        self._stiffness = np.diag(config["stiffness"])
        self._damping = config.get("damping", None)
        self.arm_joint_idxes = config.get("arm_joint_idxes", None)
        self.saturation_values = config["saturation"]
        self.Kp_nullspace = np.array(config.get("kp_null", None)).astype(np.float64)
        self.q_des = config.get("q_des", None)
        self.velocity_limits = config.get("velocity_limits", [1.57, 1.57, 1.57, 1.57, 2.5, 2.5, 2.5])
    
    def compute_controller(self, target: list[dict[str, np.ndarray]], 
                           robot_state: RobotJointState | None = None):
        # print(f'target dict: {target}')
        target = target[0]
        frame_name, target = next(iter(target.items()))
        if len(target) != 7:
            raise ValueError("target is not 7D pose containing [x,y,z,qx,qy,qz,qw],"
                             "please check whether the ee_pose target is correctly set")
        
        self._robot_model.update_kinematics(robot_state._positions, robot_state._velocities, 
                                            robot_state._accelerations)
        ee_pose_homo = self._robot_model.get_frame_pose(frame_name)
        ee_pose = convert_homo_2_7D_pose(ee_pose_homo)
        pose_error = compute_pose_diff(target, ee_pose)
        # print(f'pose err: {pose_error}, norm: {np.linalg.norm(pose_error)}')
        cur_twist = self._robot_model.get_frame_twist(frame_name, 
                                                      reference_frame = pin.LOCAL_WORLD_ALIGNED)
        vel_error = np.zeros(6) - cur_twist
        # print(f'pose err: {np.linalg.norm(pose_error[3:])}, vel: {np.linalg.norm(vel_error)}')
        spatial_acc = self._robot_model.get_frame_acc(frame_name, 
                                                      reference_frame = pin.LOCAL_WORLD_ALIGNED)
        # compute kinemtics/dynamics parameters
        J = self._robot_model.get_jacobian(frame_name, robot_state._positions, 
                                           dim=self.arm_joint_idxes,
                                           reference_frame = pin.LOCAL_WORLD_ALIGNED)
        M = self._robot_model.get_inertial_matrix(robot_state._positions, 
                                                  self.arm_joint_idxes)
        M_inv = np.linalg.pinv(M)
        task_inertial_inv = J @ M_inv @ J.T  
        task_inertial = np.linalg.pinv(task_inertial_inv)

        # compute scale
        s = self.velocity_scale_global(robot_state._velocities)
        # print(f'vel scale: {s}')
        # s = 1
        
        # compute the target end-effector wrench
        if self._damping is None:
            kp_sqrt = scipy_matrix_sqrt(self._stiffness)
            # kp_sqrt = np.sqrt(self._stiffness)
            # @TODO: check why the close loop damped system is not stable
            # task_inertial_sqrt = matrix_sqrt(task_inertial)
            task_inertial_sqrt = scipy_matrix_sqrt(task_inertial)
            self._damping = task_inertial_sqrt @ kp_sqrt + kp_sqrt @ task_inertial_sqrt
            # self._damping = 2.0 * kp_sqrt
        des_local_wrench = self._stiffness @ pose_error + self._damping @ vel_error
        # @TODO: check acceleration not stable
        des_local_wrench = task_inertial @ spatial_acc - des_local_wrench
        
        C = self._robot_model.get_coriolis_matrix(robot_state._positions, robot_state._velocities,
                                                  dims=self.arm_joint_idxes)
        friction_terms = C @ robot_state._velocities
        tau = friction_terms
        if self._gravity_compensation:
            tau = self._robot_model.id(robot_state._positions, 
                            robot_state._velocities, robot_state._accelerations)
        # print(f'base tau: {tau}')
        tau_impedance = J.T @ des_local_wrench
        # print(f'impedance tau: {tau_impedance}')
        tau -= s * tau_impedance
                
        # nullspace 
        if self.Kp_nullspace is not None and self.q_des is not None:
            dq_damping = 2 * np.sqrt(self.Kp_nullspace)
            # - np.diag(self.dq_damping * np.sqrt(self.Kp_nullspace)) @ robot_state._velocities
            null_err = np.diag(self.Kp_nullspace) @ (self.q_des - robot_state._positions) \
                        - np.diag(dq_damping * np.ones((self._robot_model.nv, self._robot_model.nv))) @ robot_state._velocities
            J_bar = M_inv @ J.T @ task_inertial
            tau_nullsapce = (np.eye(self._robot_model.nv) - J.T @ J_bar.T) @ null_err
            # print(f'nullspace tau: {tau_nullsapce}')
            tau += s * tau_nullsapce

        # tau=np.zeros(7)
        # saturation
        if not self.saturation_values is None:
            tau = np.clip(tau, self.saturation_values["min"], self.saturation_values["max"])
        return True, tau, 'torque'
    
    def set_damping(self, damping):
        self._damping = damping
        
    def velocity_scale_global(self, dq, s_min=0.1):
        abs_dq = np.abs(dq)
        margin = 0.15 * np.array(self.velocity_limits)
        
        low  = np.maximum(0.0, np.array(self.velocity_limits) - margin) 
        high = np.array(self.velocity_limits)
        
        s_j = np.ones_like(dq)
        mask = abs_dq > low
        span = np.maximum(high - low, 1e-12)
        s_j[mask] = s_min + (1.0 - s_min) * np.clip((high[mask] - abs_dq[mask]) / span[mask], 0.0, 1.0)

        s = float(np.clip(s_j.min(), 0.0, 1.0))
        return s
    