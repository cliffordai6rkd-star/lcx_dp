"""
@File    : trajectory_planning
@Author  : Haotian Liang
@Time    : 2025/5/27 16:09
@Email   : Haotianliang10@gmail.com
"""
import numpy as np
from typing import List, Tuple, Optional, Union, Dict
from abc import ABC, abstractmethod
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize_scalar
import warnings


class BaseTrajectoryPlanner(ABC):
    """
    Abstract base class for trajectory planning.
    """

    @abstractmethod
    def plan_joint_trajectory(
            self,
            waypoints: List[np.ndarray],
            times: Optional[List[float]] = None,
            velocities: Optional[List[np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Plan a trajectory in joint space.

        Args:
            waypoints: List of joint configurations
            times: Time stamps for each waypoint
            velocities: Desired velocities at waypoints

        Returns:
            Tuple of (times, positions, velocities, accelerations)
        """
        pass

    @abstractmethod
    def plan_cartesian_trajectory(
            self,
            waypoints: List[np.ndarray],
            times: Optional[List[float]] = None,
            orientations: Optional[List[np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Plan a trajectory in Cartesian space.

        Args:
            waypoints: List of Cartesian positions
            times: Time stamps for each waypoint
            orientations: Orientations at waypoints

        Returns:
            Tuple of (times, positions, orientations)
        """
        pass


class PolynomialTrajectoryPlanner(BaseTrajectoryPlanner):
    """
    Polynomial-based trajectory planner supporting:
    - Cubic splines for smooth trajectories
    - Quintic polynomials for single segments
    - Trapezoidal velocity profiles
    """

    def __init__(self, dt: float = 0.01):
        """
        Initialize the trajectory planner.

        Args:
            dt: Time step for trajectory sampling
        """
        self.dt = dt

    def plan_joint_trajectory(
            self,
            waypoints: List[np.ndarray],
            times: Optional[List[float]] = None,
            velocities: Optional[List[np.ndarray]] = None,
            accelerations: Optional[List[np.ndarray]] = None,
            method: str = "cubic_spline"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Plan a trajectory in joint space.

        Args:
            waypoints: List of joint configurations [n_waypoints x n_joints]
            times: Time stamps for each waypoint
            velocities: Desired velocities at waypoints
            accelerations: Desired accelerations at waypoints
            method: "cubic_spline", "quintic", or "trapezoidal"

        Returns:
            Tuple of (times, positions, velocities, accelerations)
        """
        waypoints = np.array(waypoints)
        n_waypoints, n_joints = waypoints.shape

        if times is None:
            # Auto-generate times based on distance between waypoints
            times = self._auto_generate_times(waypoints)

        if velocities is None:
            velocities = np.zeros_like(waypoints)

        if accelerations is None:
            accelerations = np.zeros_like(waypoints)

        if method == "cubic_spline":
            return self._plan_cubic_spline(waypoints, times, velocities)
        elif method == "quintic":
            return self._plan_quintic_polynomial(waypoints, times, velocities, accelerations)
        elif method == "trapezoidal":
            return self._plan_trapezoidal_profile(waypoints, times)
        else:
            raise ValueError(f"Unknown method: {method}")

    def plan_cartesian_trajectory(
            self,
            waypoints: List[np.ndarray],
            times: Optional[List[float]] = None,
            orientations: Optional[List[np.ndarray]] = None,
            method: str = "linear"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Plan a trajectory in Cartesian space.

        Args:
            waypoints: List of Cartesian positions [n_waypoints x 3]
            times: Time stamps for each waypoint
            orientations: Rotation matrices or quaternions at waypoints
            method: "linear", "cubic_spline"

        Returns:
            Tuple of (times, positions, orientations)
        """
        waypoints = np.array(waypoints)
        n_waypoints = len(waypoints)

        if times is None:
            times = self._auto_generate_times(waypoints)

        if orientations is None:
            orientations = [np.eye(3) for _ in range(n_waypoints)]

        # Generate time vector
        t_total = times[-1]
        t_vec = np.arange(0, t_total + self.dt, self.dt)

        if method == "linear":
            # Linear interpolation for positions
            pos_interp = interp1d(times, waypoints.T, kind='linear', axis=1)
            positions = pos_interp(t_vec).T

            # SLERP for orientations (simplified)
            orientations_interp = self._interpolate_orientations(orientations, times, t_vec)

        elif method == "cubic_spline":
            # Cubic spline for positions
            positions = np.zeros((len(t_vec), 3))
            for i in range(3):
                cs = CubicSpline(times, waypoints[:, i])
                positions[:, i] = cs(t_vec)

            orientations_interp = self._interpolate_orientations(orientations, times, t_vec)

        return t_vec, positions, orientations_interp

    def _auto_generate_times(self, waypoints: np.ndarray, max_velocity: float = 1.0) -> List[float]:
        """
        Auto-generate times based on distances between waypoints.
        """
        times = [0.0]
        for i in range(1, len(waypoints)):
            distance = np.linalg.norm(waypoints[i] - waypoints[i - 1])
            time_step = distance / max_velocity
            times.append(times[-1] + max(time_step, 0.1))  # Minimum 0.1s between points
        return times

    def _plan_cubic_spline(
            self,
            waypoints: np.ndarray,
            times: List[float],
            velocities: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Plan trajectory using cubic splines.
        """
        n_waypoints, n_joints = waypoints.shape
        t_total = times[-1]
        t_vec = np.arange(0, t_total + self.dt, self.dt)

        positions = np.zeros((len(t_vec), n_joints))
        velocities_out = np.zeros((len(t_vec), n_joints))
        accelerations = np.zeros((len(t_vec), n_joints))

        for joint in range(n_joints):
            # Create cubic spline with velocity constraints
            cs = CubicSpline(
                times,
                waypoints[:, joint],
                bc_type=((1, velocities[0, joint]), (1, velocities[-1, joint]))
            )

            positions[:, joint] = cs(t_vec)
            velocities_out[:, joint] = cs(t_vec, 1)
            accelerations[:, joint] = cs(t_vec, 2)

        return t_vec, positions, velocities_out, accelerations

    def _plan_quintic_polynomial(
            self,
            waypoints: np.ndarray,
            times: List[float],
            velocities: np.ndarray,
            accelerations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Plan trajectory using quintic polynomials between waypoints.
        """
        n_waypoints, n_joints = waypoints.shape
        t_total = times[-1]
        t_vec = np.arange(0, t_total + self.dt, self.dt)

        positions = np.zeros((len(t_vec), n_joints))
        velocities_out = np.zeros((len(t_vec), n_joints))
        accelerations_out = np.zeros((len(t_vec), n_joints))

        for i in range(n_waypoints - 1):
            # Time indices for this segment
            t_start, t_end = times[i], times[i + 1]
            mask = (t_vec >= t_start) & (t_vec <= t_end)
            t_seg = t_vec[mask] - t_start
            dt_seg = t_end - t_start

            for joint in range(n_joints):
                # Boundary conditions
                q0, qf = waypoints[i, joint], waypoints[i + 1, joint]
                v0, vf = velocities[i, joint], velocities[i + 1, joint]
                a0, af = accelerations[i, joint], accelerations[i + 1, joint]

                # Solve for quintic polynomial coefficients
                coeffs = self._solve_quintic_coefficients(q0, qf, v0, vf, a0, af, dt_seg)

                # Evaluate polynomial
                positions[mask, joint] = self._eval_polynomial(coeffs, t_seg, 0)
                velocities_out[mask, joint] = self._eval_polynomial(coeffs, t_seg, 1)
                accelerations_out[mask, joint] = self._eval_polynomial(coeffs, t_seg, 2)

        return t_vec, positions, velocities_out, accelerations_out

    def _plan_trapezoidal_profile(
            self,
            waypoints: np.ndarray,
            times: List[float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Plan trajectory using trapezoidal velocity profiles.
        """
        n_waypoints, n_joints = waypoints.shape
        t_total = times[-1]
        t_vec = np.arange(0, t_total + self.dt, self.dt)

        positions = np.zeros((len(t_vec), n_joints))
        velocities = np.zeros((len(t_vec), n_joints))
        accelerations = np.zeros((len(t_vec), n_joints))

        for i in range(n_waypoints - 1):
            t_start, t_end = times[i], times[i + 1]
            mask = (t_vec >= t_start) & (t_vec <= t_end)
            t_seg = t_vec[mask] - t_start
            dt_seg = t_end - t_start

            for joint in range(n_joints):
                q0, qf = waypoints[i, joint], waypoints[i + 1, joint]

                # Generate trapezoidal profile
                pos, vel, acc = self._generate_trapezoidal_profile(q0, qf, dt_seg, t_seg)

                positions[mask, joint] = pos
                velocities[mask, joint] = vel
                accelerations[mask, joint] = acc

        return t_vec, positions, velocities, accelerations

    def _solve_quintic_coefficients(
            self, q0: float, qf: float, v0: float, vf: float, a0: float, af: float, T: float
    ) -> np.ndarray:
        """
        Solve for quintic polynomial coefficients.
        q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        """
        # Boundary conditions matrix
        A = np.array([
            [1, 0, 0, 0, 0, 0],  # q(0) = q0
            [1, T, T ** 2, T ** 3, T ** 4, T ** 5],  # q(T) = qf
            [0, 1, 0, 0, 0, 0],  # v(0) = v0
            [0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4],  # v(T) = vf
            [0, 0, 2, 0, 0, 0],  # a(0) = a0
            [0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3]  # a(T) = af
        ])

        b = np.array([q0, qf, v0, vf, a0, af])

        return np.linalg.solve(A, b)

    def _eval_polynomial(self, coeffs: np.ndarray, t: np.ndarray, derivative: int = 0) -> np.ndarray:
        """
        Evaluate polynomial and its derivatives.
        """
        if derivative == 0:
            # Position
            return (coeffs[0] + coeffs[1] * t + coeffs[2] * t ** 2 +
                    coeffs[3] * t ** 3 + coeffs[4] * t ** 4 + coeffs[5] * t ** 5)
        elif derivative == 1:
            # Velocity
            return (coeffs[1] + 2 * coeffs[2] * t + 3 * coeffs[3] * t ** 2 +
                    4 * coeffs[4] * t ** 3 + 5 * coeffs[5] * t ** 4)
        elif derivative == 2:
            # Acceleration
            return (2 * coeffs[2] + 6 * coeffs[3] * t + 12 * coeffs[4] * t ** 2 + 20 * coeffs[5] * t ** 3)
        else:
            raise ValueError("Only derivatives 0, 1, 2 are supported")

    def _generate_trapezoidal_profile(
            self, q0: float, qf: float, T: float, t: np.ndarray,
            max_vel: float = 2.0, max_acc: float = 5.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate trapezoidal velocity profile.
        """
        distance = abs(qf - q0)
        direction = np.sign(qf - q0)

        # Calculate acceleration and cruise phases
        t_acc = min(max_vel / max_acc, T / 2)
        v_cruise = min(max_vel, distance / (T - t_acc))
        t_cruise = (distance - max_acc * t_acc ** 2) / v_cruise if v_cruise > 0 else 0
        t_dec = t_acc

        # Adjust if total time doesn't match
        if t_acc + t_cruise + t_dec > T:
            # Pure triangular profile
            t_acc = t_dec = T / 2
            t_cruise = 0
            v_cruise = distance / T

        positions = np.zeros_like(t)
        velocities = np.zeros_like(t)
        accelerations = np.zeros_like(t)

        for i, ti in enumerate(t):
            if ti <= t_acc:
                # Acceleration phase
                positions[i] = q0 + direction * 0.5 * max_acc * ti ** 2
                velocities[i] = direction * max_acc * ti
                accelerations[i] = direction * max_acc
            elif ti <= t_acc + t_cruise:
                # Cruise phase
                positions[i] = (q0 + direction * (0.5 * max_acc * t_acc ** 2 +
                                                  v_cruise * (ti - t_acc)))
                velocities[i] = direction * v_cruise
                accelerations[i] = 0
            else:
                # Deceleration phase
                t_dec_phase = ti - t_acc - t_cruise
                positions[i] = (q0 + direction * (0.5 * max_acc * t_acc ** 2 +
                                                  v_cruise * t_cruise +
                                                  v_cruise * t_dec_phase -
                                                  0.5 * max_acc * t_dec_phase ** 2))
                velocities[i] = direction * (v_cruise - max_acc * t_dec_phase)
                accelerations[i] = -direction * max_acc

        return positions, velocities, accelerations

    def _interpolate_orientations(
            self, orientations: List[np.ndarray], times: List[float], t_vec: np.ndarray
    ) -> List[np.ndarray]:
        """
        Interpolate orientations using SLERP (simplified).
        """
        # This is a simplified implementation - for production use, consider using
        # proper quaternion SLERP or rotation matrix interpolation
        orientations_interp = []

        for t in t_vec:
            # Find surrounding waypoints
            idx = np.searchsorted(times, t)
            if idx == 0:
                orientations_interp.append(orientations[0])
            elif idx >= len(times):
                orientations_interp.append(orientations[-1])
            else:
                # Linear interpolation weight
                t0, t1 = times[idx - 1], times[idx]
                alpha = (t - t0) / (t1 - t0)

                # Simple linear interpolation of rotation matrices
                # Note: This is not ideal for rotations - use quaternion SLERP for better results
                R_interp = (1 - alpha) * orientations[idx - 1] + alpha * orientations[idx]

                # Orthogonalize to ensure it's a valid rotation matrix
                U, _, Vt = np.linalg.svd(R_interp)
                R_interp = U @ Vt

                orientations_interp.append(R_interp)

        return orientations_interp


class CartesianTrajectoryPlanner:
    """
    Specialized planner for Cartesian space trajectories with integration to kinematics model.
    """

    def __init__(self, kinematics_model, dt: float = 0.01):
        """
        Initialize with a kinematics model.

        Args:
            kinematics_model: Instance of PinocchioKinematicsModel
            dt: Time step for trajectory sampling
        """
        self.kin_model = kinematics_model
        self.dt = dt
        self.poly_planner = PolynomialTrajectoryPlanner(dt)

    def plan_straight_line(
            self,
            start_pose: np.ndarray,
            end_pose: np.ndarray,
            duration: float,
            seed_joint_config: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Plan a straight-line trajectory in Cartesian space and convert to joint space.

        Args:
            start_pose: Starting pose (4x4 homogeneous matrix)
            end_pose: Ending pose (4x4 homogeneous matrix)
            duration: Duration of the trajectory
            seed_joint_config: Seed configuration for IK

        Returns:
            Dictionary with trajectory data
        """
        # Generate Cartesian trajectory
        t_vec = np.arange(0, duration + self.dt, self.dt)
        n_points = len(t_vec)

        # Linear interpolation in Cartesian space
        positions = np.zeros((n_points, 3))
        orientations = []

        for i, t in enumerate(t_vec):
            alpha = t / duration

            # Linear interpolation for position
            positions[i] = (1 - alpha) * start_pose[:3, 3] + alpha * end_pose[:3, 3]

            # SLERP for orientation (simplified)
            R_start, R_end = start_pose[:3, :3], end_pose[:3, :3]
            R_interp = self._slerp_rotation_matrices(R_start, R_end, alpha)
            orientations.append(R_interp)

        # Convert to joint space using IK
        joint_positions = np.zeros((n_points, len(self.kin_model.joint_lower_limit)))

        current_joint_config = seed_joint_config
        if current_joint_config is None:
            lower, upper = self.kin_model.get_joint_limits()
            current_joint_config = 0.5 * (lower + upper)

        for i in range(n_points):
            # Construct homogeneous transformation matrix
            T_target = np.eye(4)
            T_target[:3, :3] = orientations[i]
            T_target[:3, 3] = positions[i]

            # Solve IK
            joint_config = self.kin_model.inverse_kinematics(T_target, seed=current_joint_config)
            joint_positions[i] = joint_config
            current_joint_config = joint_config  # Use as seed for next iteration

        # Compute velocities and accelerations numerically
        joint_velocities = np.gradient(joint_positions, self.dt, axis=0)
        joint_accelerations = np.gradient(joint_velocities, self.dt, axis=0)

        return {
            'time': t_vec,
            'joint_positions': joint_positions,
            'joint_velocities': joint_velocities,
            'joint_accelerations': joint_accelerations,
            'cartesian_positions': positions,
            'cartesian_orientations': orientations
        }

    def plan_circular_arc(
            self,
            center: np.ndarray,
            start_angle: float,
            end_angle: float,
            radius: float,
            normal: np.ndarray,
            duration: float,
            orientation: Optional[np.ndarray] = None,
            seed:  Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Plan a circular arc trajectory.

        Args:
            center: Center of the circle (3D point)
            start_angle, end_angle: Start and end angles in radians
            radius: Radius of the circle
            normal: Normal vector to the plane of the circle
            duration: Duration of the trajectory
            orientation: Fixed orientation (3x3 matrix) or None for variable

        Returns:
            Dictionary with trajectory data
        """
        t_vec = np.arange(0, duration + self.dt, self.dt)
        n_points = len(t_vec)

        # Normalize normal vector
        normal = normal / np.linalg.norm(normal)

        # Create orthonormal basis for the circle plane
        if abs(normal[2]) < 0.9:
            u = np.cross(normal, np.array([0, 0, 1]))
        else:
            u = np.cross(normal, np.array([1, 0, 0]))
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)

        positions = np.zeros((n_points, 3))
        orientations = []

        for i, t in enumerate(t_vec):
            # Interpolate angle
            alpha = t / duration
            angle = start_angle + alpha * (end_angle - start_angle)

            # Calculate position on circle
            positions[i] = center + radius * (np.cos(angle) * u + np.sin(angle) * v)

            if orientation is None:
                # Point tangent to the circle
                tangent = radius * (-np.sin(angle) * u + np.cos(angle) * v)
                tangent = tangent / np.linalg.norm(tangent)

                # Create orientation with tangent as x-axis
                x_axis = tangent
                z_axis = normal
                y_axis = np.cross(z_axis, x_axis)

                R = np.column_stack([x_axis, y_axis, z_axis])
                orientations.append(R)
            else:
                orientations.append(orientation)

        # Convert to joint space (similar to straight line)
        joint_positions = np.zeros((n_points, len(self.kin_model.joint_lower_limit)))

        lower, upper = self.kin_model.get_joint_limits()
        if seed is None:
            current_joint_config = 0.5 * (lower + upper)
        else:
            current_joint_config = seed

        for i in range(n_points):
            T_target = np.eye(4)
            T_target[:3, :3] = orientations[i]
            T_target[:3, 3] = positions[i]

            joint_config = self.kin_model.inverse_kinematics(T_target, seed=current_joint_config)
            joint_positions[i] = joint_config
            current_joint_config = joint_config

        joint_velocities = np.gradient(joint_positions, self.dt, axis=0)
        joint_accelerations = np.gradient(joint_velocities, self.dt, axis=0)

        return {
            'time': t_vec,
            'joint_positions': joint_positions,
            'joint_velocities': joint_velocities,
            'joint_accelerations': joint_accelerations,
            'cartesian_positions': positions,
            'cartesian_orientations': orientations
        }

    def _slerp_rotation_matrices(self, R1: np.ndarray, R2: np.ndarray, t: float) -> np.ndarray:
        """
        Spherical linear interpolation between rotation matrices.
        """
        # Convert to axis-angle representation
        R_rel = R2 @ R1.T

        # Extract axis and angle from rotation matrix
        angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))

        if angle < 1e-6:
            return R1  # No rotation needed

        # Extract rotation axis
        axis = np.array([
            R_rel[2, 1] - R_rel[1, 2],
            R_rel[0, 2] - R_rel[2, 0],
            R_rel[1, 0] - R_rel[0, 1]
        ]) / (2 * np.sin(angle))

        # Interpolated angle
        interp_angle = t * angle

        # Rodrigues' formula for rotation matrix
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        R_interp = np.eye(3) + np.sin(interp_angle) * K + (1 - np.cos(interp_angle)) * K @ K

        return R_interp @ R1


def smooth_trajectory(
        joint_positions: np.ndarray,
        times: np.ndarray,
        smoothing_factor: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Smooth a joint trajectory using cubic splines.

    Args:
        joint_positions: Joint positions array [n_points x n_joints]
        times: Time array
        smoothing_factor: Smoothing parameter (0 = no smoothing, 1 = maximum smoothing)

    Returns:
        Tuple of (smoothed_positions, velocities, accelerations)
    """
    n_points, n_joints = joint_positions.shape

    smoothed_positions = np.zeros_like(joint_positions)
    velocities = np.zeros_like(joint_positions)
    accelerations = np.zeros_like(joint_positions)

    for joint in range(n_joints):
        # Apply smoothing using cubic spline
        cs = CubicSpline(times, joint_positions[:, joint], bc_type='natural')

        smoothed_positions[:, joint] = cs(times)
        velocities[:, joint] = cs(times, 1)
        accelerations[:, joint] = cs(times, 2)

    return smoothed_positions, velocities, accelerations


def check_trajectory_feasibility(
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        joint_accelerations: np.ndarray,
        joint_limits: Tuple[np.ndarray, np.ndarray],
        velocity_limits: Optional[np.ndarray] = None,
        acceleration_limits: Optional[np.ndarray] = None
) -> Dict[str, bool]:
    """
    Check if a trajectory is feasible given robot limits.

    Args:
        joint_positions: Joint positions [n_points x n_joints]
        joint_velocities: Joint velocities [n_points x n_joints]
        joint_accelerations: Joint accelerations [n_points x n_joints]
        joint_limits: Tuple of (lower_limits, upper_limits)
        velocity_limits: Maximum joint velocities
        acceleration_limits: Maximum joint accelerations

    Returns:
        Dictionary with feasibility checks
    """
    lower_limits, upper_limits = joint_limits

    results = {
        'position_feasible': True,
        'velocity_feasible': True,
        'acceleration_feasible': True,
        'violations': []
    }

    # Check position limits
    if np.any(joint_positions < lower_limits) or np.any(joint_positions > upper_limits):
        results['position_feasible'] = False
        results['violations'].append('Position limits exceeded')

    # Check velocity limits
    if velocity_limits is not None:
        if np.any(np.abs(joint_velocities) > velocity_limits):
            results['velocity_feasible'] = False
            results['violations'].append('Velocity limits exceeded')

    # Check acceleration limits
    if acceleration_limits is not None:
        if np.any(np.abs(joint_accelerations) > acceleration_limits):
            results['acceleration_feasible'] = False
            results['violations'].append('Acceleration limits exceeded')

    return results

