from __future__ import annotations

from controller.controller_base import ControllerBase
from motion.pin_model import RobotModel
import numpy as np
from hardware.base.utils import RobotJointState, compute_pose_diff, convert_homo_2_7D_pose
import pinocchio as pin
    
class ImpedanceController(ControllerBase):
    def __init__(self, config, robot_model: RobotModel):
        super().__init__(config, robot_model)
        self._gravity_compensation = config.get("enable_gravity_compensation", True)
        self._stiffness = np.diag(np.asarray(config["stiffness"], dtype=np.float64))
        self._fixed_damping = self._parse_damping(config.get("damping", None))
        self._damping_mode = config.get("damping_mode", "critical_task_inertia")
        self._damping_ratio = float(config.get("damping_ratio", 1.0))
        self._task_inertia_reg = float(config.get("task_inertia_regularization", 1e-4))
        self._min_eig = float(config.get("min_eigenvalue", 1e-8))
        self._pose_error_clip = config.get("error_clip", None)
        self._max_position_error = float(config.get("max_position_error", 0.04))
        self._max_orientation_error = float(config.get("max_orientation_error", 0.35))
        self._max_task_wrench = np.asarray(
            config.get("max_task_wrench", [100.0, 100.0, 100.0, 20.0, 20.0, 20.0]),
            dtype=np.float64,
        )
        self._enable_acceleration_feedforward = bool(
            config.get("enable_acceleration_feedforward", False)
        )
        self._acceleration_feedforward_gain = float(
            config.get("acceleration_feedforward_gain", 1.0)
        )
        self._max_task_acceleration = self._parse_optional_vector(
            config.get("max_task_acceleration", None)
        )
        if self._max_task_acceleration is not None and self._max_task_acceleration.size != 6:
            raise ValueError(
                f"max_task_acceleration should be 6D, got {self._max_task_acceleration.size}"
            )
        # self._target_filter_alpha = float(config.get("target_filter_alpha", 0.2))
        # self._filtered_target: np.ndarray | None = None
        self.arm_joint_idxes = config.get("arm_joint_idxes", None)
        self.saturation_values = config.get("saturation", None)
        self.Kp_nullspace = self._parse_optional_vector(config.get("kp_null", None))
        self.q_des = self._parse_optional_vector(config.get("q_des", None))
        self._nullspace_kd = config.get("dq_damping", None)
        self.velocity_limits = np.asarray(
            config.get("velocity_limits", [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]),
            dtype=np.float64,
        )
        self._velocity_soft_margin = float(config.get("velocity_soft_margin_ratio", 0.2))
        self._velocity_soft_brake_gain = float(config.get("velocity_soft_brake_gain", 4.0))
        self._velocity_hard_brake_gain = float(config.get("velocity_hard_brake_gain", 40.0))
        self._joint_velocity_damping = config.get("joint_velocity_damping", 0.0)
        self._delta_tau_max = float(config.get("delta_tau_max", 1.0))
        self._last_tau: np.ndarray | None = None
        self._damping = self._fixed_damping if self._fixed_damping is not None else self._critical_diag_damping()

    def _parse_optional_vector(self, value):
        if value is None:
            return None
        return np.asarray(value, dtype=np.float64)

    def _parse_damping(self, damping):
        if damping is None:
            return None
        damping = np.asarray(damping, dtype=np.float64)
        if damping.ndim == 1:
            if damping.size != 6:
                raise ValueError(f"Expected 6 damping values, got {damping.size}")
            return np.diag(damping)
        if damping.shape != (6, 6):
            raise ValueError(f"Damping matrix should be 6x6, got {damping.shape}")
        return damping

    def _clip_vector_norm(self, vector, max_norm):
        norm = np.linalg.norm(vector)
        if norm <= max_norm or norm < 1e-12:
            return vector
        return vector * (max_norm / norm)

    def _safe_spd_sqrt(self, matrix):
        symmetric_matrix = 0.5 * (matrix + matrix.T)
        eig_vals, eig_vecs = np.linalg.eigh(symmetric_matrix)
        eig_vals = np.clip(eig_vals, self._min_eig, None)
        return eig_vecs @ np.diag(np.sqrt(eig_vals)) @ eig_vecs.T

    def _critical_diag_damping(self):
        stiffness_diag = np.clip(np.diag(self._stiffness), self._min_eig, None)
        return np.diag(2.0 * self._damping_ratio * np.sqrt(stiffness_diag))

    def _compute_damping_matrix(self, J, M_inv):
        if self._fixed_damping is not None:
            return self._fixed_damping, None

        if self._damping_mode != "critical_task_inertia":
            return self._critical_diag_damping(), None

        task_inertia = self._compute_task_inertia(J, M_inv)
        task_inertia_sqrt = self._safe_spd_sqrt(task_inertia)
        stiffness_sqrt = self._safe_spd_sqrt(self._stiffness)
        damping = self._damping_ratio * (
            task_inertia_sqrt @ stiffness_sqrt + stiffness_sqrt @ task_inertia_sqrt
        )
        # cal_damping = np.zeros_like(damping)
        # np.fill_diagonal(cal_damping, np.diag(damping))
        # print(f'cal_damping: {cal_damping}')
        return damping, task_inertia

    def _compute_task_inertia(self, J, M_inv):
        task_inertia_inv = J @ M_inv @ J.T
        reg = self._task_inertia_reg * np.eye(6, dtype=np.float64)
        return np.linalg.pinv(task_inertia_inv + reg)
    
    def _compute_task_bias_acceleration(self, frame_name, q, dq):
        zero_qdd = np.zeros_like(q)
        task_bias_acc = self._robot_model.get_frame_acc(
            frame_name,
            q,
            dq,
            zero_qdd,
            reference_frame=pin.LOCAL_WORLD_ALIGNED,
            need_update=True,
        )
        task_bias = np.asarray(task_bias_acc.vector, dtype=np.float64)
        return task_bias

    def _get_active_state(self, vector):
        if self.arm_joint_idxes is None:
            return vector
        return vector[self.arm_joint_idxes]

    def _compute_nullspace_torque(self, q_active, dq_active, J, M_inv, task_inertia):
        if self.Kp_nullspace is None or self.q_des is None:
            return np.zeros_like(dq_active)

        if self.Kp_nullspace.size != q_active.size or self.q_des.size != q_active.size:
            raise ValueError(
                f"Nullspace config dim mismatch, kp_null: {self.Kp_nullspace.size}, "
                f"q_des: {self.q_des.size}, active joints: {q_active.size}"
            )

        if self._nullspace_kd is None:
            kd_null = 2.0 * np.sqrt(np.clip(self.Kp_nullspace, self._min_eig, None))
        else:
            kd_null = np.asarray(self._nullspace_kd, dtype=np.float64)
            if kd_null.ndim == 0:
                kd_null = np.ones_like(q_active) * float(kd_null)
            if kd_null.size != q_active.size:
                raise ValueError(
                    f"dq_damping dim mismatch, expected {q_active.size}, got {kd_null.size}"
                )

        tau_null_raw = self.Kp_nullspace * (self.q_des - q_active) - kd_null * dq_active
        if task_inertia is None:
            j_pinv = np.linalg.pinv(J)
            null_projector = np.eye(q_active.size) - J.T @ j_pinv.T
        else:
            j_bar = M_inv @ J.T @ task_inertia
            null_projector = np.eye(q_active.size) - J.T @ j_bar.T
        return null_projector @ tau_null_raw

    def _get_joint_velocity_damping(self, dof):
        damping = np.asarray(self._joint_velocity_damping, dtype=np.float64)
        if damping.ndim == 0:
            return np.ones(dof, dtype=np.float64) * float(damping)
        if damping.size == dof:
            return damping
        raise ValueError(f"joint_velocity_damping dim mismatch: {damping.size} vs {dof}")

    def _saturate_torque_rate(self, tau):
        if self._last_tau is None:
            return tau
        delta = tau - self._last_tau
        delta = np.clip(delta, -self._delta_tau_max, self._delta_tau_max)
        return self._last_tau + delta
    
    def compute_controller(self, target: list[dict[str, np.ndarray]], 
                           robot_state: RobotJointState | None = None):
        if isinstance(target, (list, tuple)):
            if len(target) == 0:
                raise ValueError("target list is empty")
            target = target[0]
        if not isinstance(target, dict):
            raise ValueError("target should be a dict or a single-item list containing a dict")
        frame_name, target_pose = next(iter(target.items()))
        
        if robot_state is None:
            raise ValueError("robot_state should not be None")
        q = np.asarray(robot_state._positions, dtype=np.float64)
        dq = np.asarray(robot_state._velocities, dtype=np.float64)
        qdd = np.asarray(robot_state._accelerations, dtype=np.float64)

        self._robot_model.update_kinematics(q, dq, qdd)
        ee_pose_homo = self._robot_model.get_frame_pose(frame_name)
        ee_pose = convert_homo_2_7D_pose(ee_pose_homo)
        # filtered_target = self._update_filtered_target(target_pose)
        filtered_target = target_pose
        pose_error = compute_pose_diff(filtered_target, ee_pose)
        # print(f'pose err: {pose_error}')
        if self._pose_error_clip:
            pose_error = np.clip(pose_error, self._pose_error_clip["min"], self._pose_error_clip["max"])
        pose_error[:3] = self._clip_vector_norm(pose_error[:3], self._max_position_error)
        pose_error[3:] = self._clip_vector_norm(pose_error[3:], self._max_orientation_error)

        cur_twist = self._robot_model.get_frame_twist(frame_name, 
                            reference_frame = pin.LOCAL_WORLD_ALIGNED)
        vel_error = -np.asarray(cur_twist.vector, dtype=np.float64)

        J = self._robot_model.get_jacobian(frame_name, q, 
            dim=self.arm_joint_idxes, reference_frame = pin.LOCAL_WORLD_ALIGNED)
        M = self._robot_model.get_inertial_matrix(q, self.arm_joint_idxes)
        M_inv = np.linalg.pinv(M, rcond=1e-6)
        self._damping, task_inertia = self._compute_damping_matrix(J, M_inv)
        des_local_wrench = self._stiffness @ pose_error + self._damping @ vel_error

        if self._enable_acceleration_feedforward:
            if task_inertia is None: 
                task_inertia = self._compute_task_inertia(J, M_inv)
            desired_task_acc = np.zeros(6, dtype=np.float64)
            desired_task_acc -= self._compute_task_bias_acceleration(frame_name, q, dq)
            if self._max_task_acceleration is not None:
                desired_task_acc = np.clip(
                    desired_task_acc, -self._max_task_acceleration, self._max_task_acceleration
                )
            des_local_wrench += self._acceleration_feedforward_gain * (
                task_inertia @ desired_task_acc
            )
            # print(f'ff des local wrench: {des_local_wrench}')
        des_local_wrench = np.clip(des_local_wrench, -self._max_task_wrench, self._max_task_wrench)

        dq_active = self._get_active_state(dq)
        q_active = self._get_active_state(q)
        s = self.velocity_scale_global(dq_active, s_min=0.25)

        C = self._robot_model.get_coriolis_matrix(q, dq, dims=self.arm_joint_idxes)
        tau = C @ dq_active
        if self._gravity_compensation:
            tau += self._robot_model.get_gravity_vector(q, self.arm_joint_idxes)

        tau += s * (J.T @ des_local_wrench)
        joint_velocity_damping = self._get_joint_velocity_damping(dq_active.size) * dq_active
        # print(f'js damping: {joint_velocity_damping}')
        tau -= joint_velocity_damping
                
        tau_nullsapce = self._compute_nullspace_torque(q_active, dq_active, J, M_inv, task_inertia)
        tau += s * tau_nullsapce
        # @TODO: figure out velocity safety
        tau = self._apply_velocity_safety(tau, dq_active)

        tau = self._saturate_torque_rate(tau)
        if self.saturation_values is not None and not self._gravity_compensation:
            tau = np.clip(tau, self.saturation_values["min"], self.saturation_values["max"])

        self._last_tau = tau.copy()
        return True, tau, 'torque'
    
    def set_damping(self, damping):
        self._fixed_damping = self._parse_damping(damping)
        if self._fixed_damping is not None:
            self._damping = self._fixed_damping
    
    def _get_active_velocity_limits(self, dof):
        limits = self.velocity_limits
        if limits.size == dof:
            return limits
        if self.arm_joint_idxes is not None and limits.size == self._robot_model.nv:
            return limits[self.arm_joint_idxes]
        raise ValueError(f"velocity_limits dim mismatch: {limits.size} vs {dof}")

    def velocity_scale_global(self, dq, s_min=0.05):
        """
            @param: s_min: bigger -> small influence of velocity reduction, 
                smaller -> strong influence of velocity reduction and bad tracking
        """
        abs_dq = np.abs(dq)
        limits = self._get_active_velocity_limits(dq.size)
        margin = self._velocity_soft_margin * limits
        
        low  = np.maximum(0.0, limits - margin) 
        high = limits
        
        s_j = np.ones_like(dq)
        mask = abs_dq > low
        span = np.maximum(high - low, 1e-12)
        s_j[mask] = s_min + (1.0 - s_min) * np.clip((high[mask] - abs_dq[mask]) / span[mask], 0.0, 1.0)

        s = float(np.clip(s_j.min(), 0.0, 1.0))
        return s

    def _apply_velocity_safety(self, tau, dq_active):
        limits = self._get_active_velocity_limits(dq_active.size)
        abs_dq = np.abs(dq_active)
        low = np.maximum(0.0, limits * (1.0 - self._velocity_soft_margin))
        span = np.maximum(limits - low, 1e-8)
        soft_ratio = np.clip((abs_dq - low) / span, 0.0, 1.0)
        soft_tau_reduction = self._velocity_soft_brake_gain * soft_ratio * dq_active
        
        over_limit = np.maximum(abs_dq - limits, 0.0)
        hard_tau_reductoon = self._velocity_hard_brake_gain * np.sign(dq_active) * over_limit
        # print(f'soft_tau_reduction: {soft_tau_reduction} hard {hard_tau_reductoon}')
        tau -= (soft_tau_reduction + hard_tau_reductoon)
        return tau
    
    def reset(self, frame_name: str, robot_state: RobotJointState) -> None:
        self._filtered_target = None
        self._last_tau = None
    
    def _normalize_quaternion(self, quat):
        norm = np.linalg.norm(quat)
        if norm < 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return quat / norm

    def _update_filtered_target(self, target):
        target = np.asarray(target, dtype=np.float64)
        if self._filtered_target is None:
            self._filtered_target = target.copy()
            return self._filtered_target

        alpha = np.clip(self._target_filter_alpha, 0.0, 1.0)
        self._filtered_target[:3] = (1.0 - alpha) * self._filtered_target[:3] + alpha * target[:3]

        prev_q = self._normalize_quaternion(self._filtered_target[3:])
        new_q = self._normalize_quaternion(target[3:])
        if np.dot(prev_q, new_q) < 0.0:
            new_q = -new_q
        blended_q = (1.0 - alpha) * prev_q + alpha * new_q
        self._filtered_target[3:] = self._normalize_quaternion(blended_q)
        return self._filtered_target
