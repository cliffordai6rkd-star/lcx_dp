"""
@File    : kinematics_add_ik_my_implement
@Author  : Haotian Liang
@Time    : 2025/4/27 11:31
@Email   :Haotianliang10@gmail.com
"""
"""
@File    : kinematics
@Author  : Haotian Liang
@Time    : 2025/4/25 14:33
@Email   :Haotianliang10@gmail.com
"""

import numpy as np
import pinocchio as pin
from typing import Optional, Tuple, Dict
from abc import ABC, abstractmethod
import numpy.linalg as LA



# Define the abstract base class
class BaseKinematicsModel(ABC):
    """
    Abstract interface for robot kinematics modeling.
    Provides methods for forward kinematics, inverse kinematics,
    Jacobian calculation, and inverse velocity kinematics.
    """

    @abstractmethod
    def forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Calculate the end-effector pose from given joint positions.

        Args:
            joint_positions: Joint positions array of shape (n,) where n is the number of joints

        Returns:
            pose: End-effector pose as a homogeneous transformation matrix of shape (4, 4)
                 or as a pose vector (position and orientation)
        """
        pass

    @abstractmethod
    def inverse_kinematics(
            self,
            target_pose: np.ndarray,
            seed: Optional[np.ndarray] = None,
            joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Calculate joint positions to achieve a target end-effector pose.

        Args:
            target_pose: Target end-effector pose as a homogeneous transformation matrix
                        of shape (4, 4) or as a pose vector
            seed: Initial guess for joint positions, shape (n,)
            joint_limits: Tuple of (lower_limits, upper_limits) arrays of shape (n,)

        Returns:
            joint_positions: Calculated joint positions of shape (n,)
        """
        pass

    @abstractmethod
    def jacobian(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Calculate the Jacobian matrix at the given joint positions.

        Args:
            joint_positions: Joint positions array of shape (n,)

        Returns:
            jacobian_matrix: Jacobian matrix of shape (6, n) mapping joint velocities
                           to end-effector twist
        """
        pass

    @abstractmethod
    def inverse_velocity_kinematics(
            self,
            twist: np.ndarray,
            joint_positions: np.ndarray,
            joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Calculate joint velocities to achieve a desired end-effector twist.

        Args:
            twist: Desired end-effector twist vector of shape (6,)
                  (linear and angular velocities)
            joint_positions: Current joint positions of shape (n,)
            joint_limits: Tuple of (lower_limits, upper_limits) arrays of shape (n,)

        Returns:
            joint_velocities: Calculated joint velocities of shape (n,)
        """
        pass

    @abstractmethod
    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the joint limits of the robot.

        Returns:
            joint_limits: Tuple of (lower_limits, upper_limits) arrays of shape (n,)
        """
        pass


# Implement the Pinocchio-based model
class PinocchioKinematicsModel(BaseKinematicsModel):
    def __init__(self, urdf_path: str, base_link: str = None, end_effector_link: str = None):
        """
        Initialize the Pinocchio kinematics model from a URDF file.

        Args:
            urdf_path: Path to the URDF file
            base_link: Name of the base link (if None, uses the root link)
            end_effector_link: Name of the end-effector link

        """
        # Call the parent class initializer
        super().__init__()

        self.urdf_path = urdf_path
        # Build the Pinocchio model from URDF
        # 使用 pinocchio.buildModelFromUrdf() 加载 URDF 文件
        self.model = pin.buildModelFromUrdf(urdf_path)
        # 创建对应的 Pinocchio 数据对象
        self.data = pin.Data(self.model)

        # Store model information
        self.n_joints = self.model.nq  # Number of joints in configuration space

        # Get joint limits directly from the model's position limits
        # In Pinocchio, these are stored in the model, not in individual joints
        self.joint_lower_limit = self.model.lowerPositionLimit.copy()
        self.joint_upper_limit = self.model.upperPositionLimit.copy()

        # 处理基座链接（base link）和末端执行器链接（end-effector link）的指定
        # Handle the base link
        if base_link is None:
            self.base_id = 0  # Use the root frame as base
        else:
            if not self.model.existFrame(base_link):
                raise ValueError(f"Base link '{base_link}' not found in the model")
            self.base_id = self.model.getFrameId(base_link)

        # Handle the end-effector link
        if end_effector_link is None:
            # If not specified, use the last operational frame
            frames = [f for f in self.model.frames if f.type == pin.FrameType.OPERATIONAL]
            if not frames:
                raise ValueError("No operational frames found in the model")
            self.ee_frame_id = frames[-1].id
            self.ee_frame_name = frames[-1].name
        else:
            if not self.model.existFrame(end_effector_link):
                raise ValueError(f"End-effector link '{end_effector_link}' not found in the model")
            self.ee_frame_id = self.model.getFrameId(end_effector_link)
            self.ee_frame_name = end_effector_link

        # Handle the case where the model might include a floating base
        # In Pinocchio, the first 7 indices might be for the floating base
        # We need to extract only the actual robot joints' limits
        if self.model.nv != self.model.nq:  # This can indicate a quaternion representation
            # Find the number of actual robot joints (excluding floating base)
            # nv 通常会多出 6 个 DOF（SE(3)）
            active_joints_dof = self.model.nv - 6  # Subtract 6 DOF for the floating base

            # Get the position limits for actual joints only
            self.joint_lower_limit = self.joint_lower_limit[7:7 + active_joints_dof]
            self.joint_upper_limit = self.joint_upper_limit[7:7 + active_joints_dof]

        # Set infinite limits to a large finite value for numerical stability
        # Pinocchio 模型中可能会存在 ±inf 的 joint limit（未指定）
        inf_mask = np.isinf(self.joint_lower_limit)
        if np.any(inf_mask):
            self.joint_lower_limit[inf_mask] = -1e10

        inf_mask = np.isinf(self.joint_upper_limit)
        if np.any(inf_mask):
            self.joint_upper_limit[inf_mask] = 1e10

    def forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Calculate the forward kinematics to get the end-effector pose.

        Args:
            joint_positions: Joint angles array of shape (n,) where n is the number of joints

        Returns:
            pose: End-effector pose as a homogeneous transformation matrix of shape (4, 4)
        """

        # Validate input dimensions
        if joint_positions.shape[0] != len(self.joint_lower_limit):
            raise ValueError(f"Expected joint angles of dimension {len(self.joint_lower_limit)}, "
                             f"but got {joint_positions.shape[0]}")

        # Create a configuration vector (may need padding depending on Pinocchio model structure)
        q = np.zeros(self.model.nq)

        # If the model has a floating base (7 DoF for the first joint),
        # we need special handling, otherwise we can directly use joint_positions
        if self.model.nq > len(joint_positions):
            # The first 7 values typically represent the floating base (3 for position, 4 for quaternion)
            # Set the floating base to identity transformation
            q[0:7] = np.array([0, 0, 0, 0, 0, 0, 1])  # [x, y, z, qx, qy, qz, qw]

            # Fill in the actual joint angles
            q[7:7 + len(joint_positions)] = joint_positions
        else:
            # Direct assignment if dimensions match
            q[:] = joint_positions

        # Compute forward kinematics
        pin.forwardKinematics(self.model, self.data, q)

        # Update the placement of all frames
        pin.updateFramePlacements(self.model, self.data)

        # Get the transformation matrix for the end-effector frame
        T_ee = self.data.oMf[self.ee_frame_id].copy()
        # print('T_ee Type:',type(T_ee),'\n',T_ee)
        # print(T_ee.homogeneous)

        # Return the 4x4 homogeneous transformation matrix
        return np.array(T_ee.homogeneous)

    def inverse_kinematics(
            self,
            target_pose: np.ndarray,
            seed: Optional[np.ndarray] = None,
            joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            max_iter: int = 10000,
            tol: float = 1e-7,
            lambda_reg: float = 0.1
    ) -> np.ndarray:
        """
        Solve inverse kinematics to find joint angles that achieve the target pose.

        Args:
            target_pose: Target end-effector pose as a 4x4 homogeneous transformation matrix
            seed: Initial guess for joint angles, shape (n,)
            joint_limits: Tuple of (lower_limits, upper_limits) arrays. If None, use model defaults
            max_iter: Maximum number of iterations for the solver
            tol: Tolerance for convergence
            lambda_reg: Damping factor for the damped least squares method

        Returns:
            joint_angles: Solved joint angles of shape (n,)
        """

        # Get the number of joints
        n_joints = len(self.joint_lower_limit)

        # Use provided joint limits or default to model limits
        if joint_limits is None:
            lower_limits = self.joint_lower_limit
            upper_limits = self.joint_upper_limit
        else:
            lower_limits, upper_limits = joint_limits

        # Initialize joint angles with seed or middle of joint range if not provided
        if seed is None:
            q = 0.5 * (lower_limits + upper_limits)
        else:
            q = seed.copy()

        # Ensure q is within joint limits
        q = np.clip(q, lower_limits, upper_limits)

        # Convert target pose to SE3 placement
        target_placement = pin.SE3(target_pose[:3, :3], target_pose[:3, 3])

        # Initialize variables for the iterative solver
        converged = False

        for i in range(max_iter):
            # Compute current forward kinematics
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            # Get current end-effector placement
            current_placement = self.data.oMf[self.ee_frame_id]

            # Compute the error in SE3 (log maps the difference to a spatial velocity)
            err_se3 = pin.log(current_placement.inverse() * target_placement).vector

            # Check for convergence
            if np.linalg.norm(err_se3) < tol:
                converged = True
                break

            # Compute the Jacobian at the current configuration
            pin.computeJointJacobians(self.model, self.data, q)
            J = pin.getFrameJacobian(self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.LOCAL)

            # Damped least squares method
            JJT = J @ J.T + lambda_reg * np.eye(6)
            delta_q = J.T @ np.linalg.solve(JJT, err_se3)

            # Update joint angles
            q = q + delta_q

            # Project back to joint limits
            q = np.clip(q, lower_limits, upper_limits)

        if not converged:
            # If the solver did not converge, return the best solution found
            print(f"Warning: IK did not converge after {max_iter} iterations. Best error: {np.linalg.norm(err_se3)}")

        return q

    def inverse_kinematics_SDLS(
            self,
            target_pose: np.ndarray,
            seed: Optional[np.ndarray] = None,
            joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            max_iter: int = 10000,
            tol: float = 1e-7,
            lambda_min: float = 0.1,
            lambda_max: float = 0.5,
            wn: float = 0.01
    ) -> np.ndarray:
        """
        Solve inverse kinematics using Selective Damped Least Squares (SDLS) method.

        Args:
            target_pose: Target end-effector pose as a 4x4 homogeneous transformation matrix
            seed: Initial guess for joint angles, shape (n,)
            joint_limits: Tuple of (lower_limits, upper_limits) arrays. If None, use model defaults
            max_iter: Maximum number of iterations for the solver
            tol: Tolerance for convergence
            lambda_min: Minimum damping factor
            lambda_max: Maximum damping factor
            wn: Weight for null-space optimization to stay close to seed position

        Returns:
            joint_angles: Solved joint angles of shape (n,)
        """



        # Get the number of joints
        n_joints = len(self.joint_lower_limit)

        # Use provided joint limits or default to model limits
        if joint_limits is None:
            lower_limits = self.joint_lower_limit
            upper_limits = self.joint_upper_limit
        else:
            lower_limits, upper_limits = joint_limits

        # Initialize joint angles with seed or middle of joint range if not provided
        if seed is None:
            q = 0.5 * (lower_limits + upper_limits)
            q_ref = q.copy()  # Reference configuration for null-space optimization
        else:
            q = seed.copy()
            q_ref = seed.copy()

        # Ensure q is within joint limits
        q = np.clip(q, lower_limits, upper_limits)

        # Convert target pose to SE3 placement
        target_placement = pin.SE3(target_pose[:3, :3], target_pose[:3, 3])

        # Initialize variables for the iterative solver
        converged = False

        for i in range(max_iter):
            # Compute current forward kinematics
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            # Get current end-effector placement
            current_placement = self.data.oMf[self.ee_frame_id]

            # Compute the error in SE3 (log maps the difference to a spatial velocity)
            err_se3 = pin.log(current_placement.inverse() * target_placement).vector

            # Check for convergence
            if np.linalg.norm(err_se3) < tol:
                converged = True
                break

            # Compute the Jacobian at the current configuration
            pin.computeJointJacobians(self.model, self.data, q)
            J = pin.getFrameJacobian(self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.LOCAL)

            # Compute SVD decomposition of the Jacobian
            U, s, Vh = LA.svd(J, full_matrices=False)

            # Selective Damping - calculate damping factors for each singular value
            lambda_values = np.zeros_like(s)
            for j in range(len(s)):
                if s[j] < lambda_min:
                    lambda_values[j] = lambda_max
                else:
                    # Scale damping inversely with the singular value
                    lambda_values[j] = lambda_max * (1.0 - (s[j] - lambda_min) / (1.0 - lambda_min))
                    lambda_values[j] = max(lambda_values[j], lambda_min)

            # Compute the damped pseudoinverse using SVD components
            s_inv = np.array([(s[j] / (s[j] ** 2 + lambda_values[j] ** 2)) for j in range(len(s))])
            J_pinv = Vh.T @ np.diag(s_inv) @ U.T

            # Primary task: move end-effector to target
            delta_q_primary = J_pinv @ err_se3

            # Null space optimization: try to stay close to reference configuration
            if wn > 0:
                # Projection operator into the null space of J
                N = np.eye(n_joints) - J_pinv @ J
                # Secondary task: minimize distance to reference configuration
                delta_q_secondary = wn * N @ (q_ref - q)
                # Combine primary and secondary tasks
                delta_q = delta_q_primary + delta_q_secondary
            else:
                delta_q = delta_q_primary

            # Update joint angles
            q = q + delta_q

            # Project back to joint limits
            q = np.clip(q, lower_limits, upper_limits)

        if not converged:
            # If the solver did not converge, return the best solution found
            print(f"Warning: IK did not converge after {max_iter} iterations. Best error: {np.linalg.norm(err_se3)}")

        return q
    # todo
    def inverse_kinematics_dls_min_motion(
            self,
            target_pose: np.ndarray,
            seed: Optional[np.ndarray] = None,
            joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            max_iter: int = 100,
            tol: float = 1e-6,
            lambda_reg: float = 0.1,
            alpha: float = 0.1  # 控制“最小变化”强度
    ) -> np.ndarray:

        n = len(self.joint_lower_limit)
        lower_limits = self.joint_lower_limit if joint_limits is None else joint_limits[0]
        upper_limits = self.joint_upper_limit if joint_limits is None else joint_limits[1]

        q_ref = 0.5 * (lower_limits + upper_limits) if seed is None else seed.copy()
        q = np.clip(q_ref.copy(), lower_limits, upper_limits)

        target_placement = pin.SE3(target_pose[:3, :3], target_pose[:3, 3])
        I = np.eye(n)

        for _ in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            err_se3 = pin.log(self.data.oMf[self.ee_frame_id].inverse() * target_placement).vector
            if np.linalg.norm(err_se3) < tol:
                break

            pin.computeJointJacobians(self.model, self.data, q)
            J = pin.getFrameJacobian(self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.LOCAL)

            # DLS 伪逆
            JJT = J @ J.T + lambda_reg * np.eye(6)
            J_pinv = J.T @ np.linalg.solve(JJT, np.eye(6))  # shape: (n, 6)

            delta_q_main = J_pinv @ err_se3

            # null-space projection
            q_diff = q_ref - q
            delta_q_null = alpha * (I - J_pinv @ J) @ q_diff

            # combine update
            delta_q = delta_q_main + delta_q_null
            q = np.clip(q + delta_q, lower_limits, upper_limits)

        return q

    def jacobian(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Calculate the Jacobian matrix at the given joint positions.

        Args:
            joint_positions: Joint positions array of shape (n,)

        Returns:
            jacobian_matrix: Jacobian matrix of shape (6, n) mapping joint velocities
                           to end-effector twist
        """

        # Validate input dimensions
        if joint_positions.shape[0] != len(self.joint_lower_limit):
            raise ValueError(f"Expected joint positions of dimension {len(self.joint_lower_limit)}, "
                             f"but got {joint_positions.shape[0]}")

        # Create a configuration vector for Pinocchio
        q = np.zeros(self.model.nq)

        # Handle model with floating base if necessary
        if self.model.nq > len(joint_positions):
            # Set the floating base to identity transformation
            q[0:7] = np.array([0, 0, 0, 0, 0, 0, 1])  # [x, y, z, qx, qy, qz, qw]
            # Fill in the actual joint positions
            q[7:7 + len(joint_positions)] = joint_positions
        else:
            # Direct assignment if dimensions match
            q[:] = joint_positions

        # Compute the Jacobian at the current configuration
        pin.computeJointJacobians(self.model, self.data, q)

        # Update frame placements to ensure the end-effector frame is updated
        pin.updateFramePlacements(self.model, self.data)

        # Get the Jacobian matrix for the end-effector frame (in the base frame)
        full_jacobian = pin.getFrameJacobian(self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.WORLD)

        # Extract the relevant part of the Jacobian if we have a floating base
        if self.model.nq > len(joint_positions):
            # Extract only the columns corresponding to the actual joints (skip floating base)
            jacobian_matrix = full_jacobian[:, 7:7 + len(joint_positions)]
        else:
            jacobian_matrix = full_jacobian

        # Return the Jacobian matrix with shape (6, dof)
        return jacobian_matrix

    def inverse_velocity_kinematics(
            self,
            twist: np.ndarray,
            joint_positions: np.ndarray,
            joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            damping: float = 0.1
    ) -> np.ndarray:
        """
        Calculate joint velocities to achieve a desired end-effector twist.

        Args:
            twist: Desired end-effector twist vector of shape (6,)
                  (first 3 elements: linear velocity, last 3 elements: angular velocity)
            joint_positions: Current joint positions of shape (n,)
            joint_limits: Tuple of (lower_limits, upper_limits) arrays of shape (n,)
            damping: Damping factor for the pseudoinverse

        Returns:
            joint_velocities: Calculated joint velocities of shape (n,)
        """

        # Validate input dimensions
        if joint_positions.shape[0] != len(self.joint_lower_limit):
            raise ValueError(f"Expected joint positions of dimension {len(self.joint_lower_limit)}, "
                             f"but got {joint_positions.shape[0]}")

        if twist.shape[0] != 6:
            raise ValueError(f"Expected twist of dimension 6, but got {twist.shape[0]}")

        # Create a configuration vector for Pinocchio
        q = np.zeros(self.model.nq)

        # Handle model with floating base if necessary
        if self.model.nq > len(joint_positions):
            # Set the floating base to identity transformation
            q[0:7] = np.array([0, 0, 0, 0, 0, 0, 1])  # [x, y, z, qx, qy, qz, qw]
            # Fill in the actual joint positions
            q[7:7 + len(joint_positions)] = joint_positions
        else:
            # Direct assignment if dimensions match
            q[:] = joint_positions

        # Compute the Jacobian at the current configuration
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        # Get the Jacobian matrix for the end-effector frame (in world frame)
        J = pin.getFrameJacobian(self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.WORLD)

        # Extract the relevant part of the Jacobian if we have a floating base
        if self.model.nq > len(joint_positions):
            J = J[:, 7:7 + len(joint_positions)]

        # Compute the damped pseudoinverse of the Jacobian
        # Use SVD for numerical stability
        U, s, Vh = np.linalg.svd(J, full_matrices=False)

        # Apply damping to singular values
        s_damped = s / (s ** 2 + damping ** 2)

        # Compute the damped pseudoinverse
        J_pinv = Vh.T @ np.diag(s_damped) @ U.T

        # Compute joint velocities
        joint_velocities = J_pinv @ twist

        # Apply velocity limits if joint limits are provided
        if joint_limits is not None:
            lower_limits, upper_limits = joint_limits

            # Simple velocity scaling based on joint limits to avoid exceeding limits
            # This is a basic approach - more sophisticated methods could be implemented
            time_to_limits = np.ones_like(joint_positions)

            for i in range(len(joint_positions)):
                if joint_velocities[i] > 0:
                    time_to_limits[i] = (upper_limits[i] - joint_positions[i]) / (joint_velocities[i] + 1e-10)
                elif joint_velocities[i] < 0:
                    time_to_limits[i] = (lower_limits[i] - joint_positions[i]) / (joint_velocities[i] - 1e-10)

            # Find the minimum time to hit a limit
            min_time = np.min(time_to_limits)

            # Scale velocities if necessary to avoid exceeding limits
            if min_time < 1.0 and min_time > 0:
                joint_velocities *= min_time

        return joint_velocities

    def inverse_velocity_kinematics(
            self,
            twist: np.ndarray,
            joint_positions: np.ndarray,
            joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            damping: float = 0.1
    ) -> np.ndarray:
        """
        Calculate joint velocities to achieve a desired end-effector twist.

        Args:
            twist: Desired end-effector twist vector of shape (6,)
                  (first 3 elements: linear velocity, last 3 elements: angular velocity)
            joint_positions: Current joint positions of shape (n,)
            joint_limits: Tuple of (lower_limits, upper_limits) arrays of shape (n,)
            damping: Damping factor for the pseudoinverse

        Returns:
            joint_velocities: Calculated joint velocities of shape (n,)
        """

        # Validate input dimensions
        if joint_positions.shape[0] != len(self.joint_lower_limit):
            raise ValueError(f"Expected joint positions of dimension {len(self.joint_lower_limit)}, "
                             f"but got {joint_positions.shape[0]}")

        if twist.shape[0] != 6:
            raise ValueError(f"Expected twist of dimension 6, but got {twist.shape[0]}")

        # Create a configuration vector for Pinocchio
        q = np.zeros(self.model.nq)

        # Handle model with floating base if necessary
        if self.model.nq > len(joint_positions):
            # Set the floating base to identity transformation
            q[0:7] = np.array([0, 0, 0, 0, 0, 0, 1])  # [x, y, z, qx, qy, qz, qw]
            # Fill in the actual joint positions
            q[7:7 + len(joint_positions)] = joint_positions
        else:
            # Direct assignment if dimensions match
            q[:] = joint_positions

        # Compute the Jacobian at the current configuration
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        # Get the Jacobian matrix for the end-effector frame (in world frame)
        J = pin.getFrameJacobian(self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.WORLD)

        # Extract the relevant part of the Jacobian if we have a floating base
        if self.model.nq > len(joint_positions):
            J = J[:, 7:7 + len(joint_positions)]

        # Compute the damped pseudoinverse of the Jacobian
        # Use SVD for numerical stability
        U, s, Vh = np.linalg.svd(J, full_matrices=False)

        # Apply damping to singular values
        s_damped = s / (s ** 2 + damping ** 2)

        # Compute the damped pseudoinverse
        J_pinv = Vh.T @ np.diag(s_damped) @ U.T

        # Compute joint velocities
        joint_velocities = J_pinv @ twist

        # Apply velocity limits if joint limits are provided
        if joint_limits is not None:
            lower_limits, upper_limits = joint_limits

            # Simple velocity scaling based on joint limits to avoid exceeding limits
            # This is a basic approach - more sophisticated methods could be implemented
            time_to_limits = np.ones_like(joint_positions)

            for i in range(len(joint_positions)):
                if joint_velocities[i] > 0:
                    time_to_limits[i] = (upper_limits[i] - joint_positions[i]) / (joint_velocities[i] + 1e-10)
                elif joint_velocities[i] < 0:
                    time_to_limits[i] = (lower_limits[i] - joint_positions[i]) / (joint_velocities[i] - 1e-10)

            # Find the minimum time to hit a limit
            min_time = np.min(time_to_limits)

            # Scale velocities if necessary to avoid exceeding limits
            if min_time < 1.0 and min_time > 0:
                joint_velocities *= min_time

        return joint_velocities

    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the joint limits of the robot.

        Returns:
            joint_limits: Tuple of (lower_limits, upper_limits) arrays of shape (n,)
        """
        return self.joint_lower_limit, self.joint_upper_limit

    def fk_vel(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """
        Calculate the end-effector velocity (twist) given joint positions and velocities.

        Args:
            q: Joint positions array of shape (n,)
            dq: Joint velocities array of shape (n,)

        Returns:
            twist: End-effector twist vector of shape (6,) representing [vx, vy, vz, wx, wy, wz]
                  in the base (world) frame
        """

        # Validate input dimensions
        if q.shape[0] != len(self.joint_lower_limit):
            raise ValueError(f"Expected joint positions of dimension {len(self.joint_lower_limit)}, "
                             f"but got {q.shape[0]}")

        if dq.shape[0] != len(self.joint_lower_limit):
            raise ValueError(f"Expected joint velocities of dimension {len(self.joint_lower_limit)}, "
                             f"but got {dq.shape[0]}")

        # Create configuration vectors for Pinocchio
        q_pin = np.zeros(self.model.nq)
        dq_pin = np.zeros(self.model.nv)

        # Handle model with floating base if necessary
        if self.model.nq > len(q):
            # Set the floating base to identity transformation
            q_pin[0:7] = np.array([0, 0, 0, 0, 0, 0, 1])  # [x, y, z, qx, qy, qz, qw]
            # Fill in the actual joint positions
            q_pin[7:7 + len(q)] = q

            # For velocity, if the model has a floating base with 6 DoF velocity
            if self.model.nv >= 6:
                # Set the floating base velocities to zero
                dq_pin[0:6] = np.zeros(6)
                # Fill in the actual joint velocities
                dq_pin[6:6 + len(dq)] = dq
        else:
            # Direct assignment if dimensions match
            q_pin[:] = q
            dq_pin[:] = dq

        # Compute forward kinematics with velocity
        pin.forwardKinematics(self.model, self.data, q_pin, dq_pin)

        # Update frame placements and velocities
        pin.updateFramePlacements(self.model, self.data)

        # Get the end-effector velocity in the world frame
        # WORLD frame is recommended for most applications
        ee_velocity = pin.getFrameVelocity(self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.WORLD)

        # Convert to a 6D numpy array [vx, vy, vz, wx, wy, wz]
        # Note: Pinocchio's spatial velocity is [wx, wy, wz, vx, vy, vz], so we need to reorder
        twist = np.array([ee_velocity.linear[0], ee_velocity.linear[1], ee_velocity.linear[2],
                          ee_velocity.angular[0], ee_velocity.angular[1], ee_velocity.angular[2]])

        return twist

    def fk_all(self, q: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate the poses of all links in the robot given joint positions.

        Args:
            q: Joint positions array of shape (n,)

        Returns:
            poses: Dictionary mapping link names to 4x4 homogeneous transformation matrices
                   representing poses in the base frame
        """

        # Validate input dimensions
        if q.shape[0] != len(self.joint_lower_limit):
            raise ValueError(f"Expected joint positions of dimension {len(self.joint_lower_limit)}, "
                             f"but got {q.shape[0]}")

        # Create a configuration vector for Pinocchio
        q_pin = np.zeros(self.model.nq)

        # Handle model with floating base if necessary
        if self.model.nq > len(q):
            # Set the floating base to identity transformation
            q_pin[0:7] = np.array([0, 0, 0, 0, 0, 0, 1])  # [x, y, z, qx, qy, qz, qw]
            # Fill in the actual joint positions
            q_pin[7:7 + len(q)] = q
        else:
            # Direct assignment if dimensions match
            q_pin[:] = q

        # Compute forward kinematics
        pin.forwardKinematics(self.model, self.data, q_pin)

        # Update all frame placements
        pin.updateFramePlacements(self.model, self.data)

        # Dictionary to store the poses
        poses = {}

        # Iterate through all frames
        for frame_id, frame in enumerate(self.model.frames):
            # Get the frame type - we'll include all frames that aren't universe or joint frames
            # This approach is more robust to different versions of Pinocchio
            frame_type = frame.type

            # Check if this is a body/link frame (anything that isn't a joint or universe frame)
            # In newer Pinocchio versions, we'd use pin.FrameType.BODY or pin.FrameType.OPERATIONAL
            # But we'll use a more general approach for compatibility
            if frame_type != 0:  # Skip universe frames (typically type 0)
                # Get the frame's global placement
                T = self.data.oMf[frame_id]

                # Convert to 4x4 homogeneous matrix and store in the dictionary
                poses[frame.name] = np.array(T.homogeneous)

        return poses

    # 1. Levenberg-Marquardt Method (LM)
    def inverse_kinematics_LM(
            self,
            target_pose: np.ndarray,
            seed: Optional[np.ndarray] = None,
            joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            max_iter: int = 5000,
            tol: float = 1e-6,
            lambda_init: float = 0.1,
            lambda_min: float = 1e-6,
            lambda_max: float = 1e3,
            lambda_factor: float = 10.0
    ) -> np.ndarray:
        """
        Solve inverse kinematics using the Levenberg-Marquardt method.

        The LM algorithm dynamically adjusts the damping parameter based on the progress:
        - Increases damping when the error increases (more like gradient descent)
        - Decreases damping when the error decreases (more like Gauss-Newton)

        Args:
            target_pose: Target end-effector pose as a 4x4 homogeneous transformation matrix
            seed: Initial guess for joint angles
            joint_limits: Tuple of (lower_limits, upper_limits) arrays
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            lambda_init: Initial damping parameter
            lambda_min: Minimum damping parameter
            lambda_max: Maximum damping parameter
            lambda_factor: Factor to multiply/divide lambda by

        Returns:
            joint_angles: Solved joint angles
        """
        # Get the number of joints
        n_joints = len(self.joint_lower_limit)

        # Use provided joint limits or default to model limits
        if joint_limits is None:
            lower_limits = self.joint_lower_limit
            upper_limits = self.joint_upper_limit
        else:
            lower_limits, upper_limits = joint_limits

        # Initialize joint angles with seed or middle of joint range
        if seed is None:
            q = 0.5 * (lower_limits + upper_limits)
        else:
            q = seed.copy()

        # Ensure q is within joint limits
        q = np.clip(q, lower_limits, upper_limits)

        # Convert target pose to SE3 placement
        target_placement = pin.SE3(target_pose[:3, :3], target_pose[:3, 3])

        # Initialize damping parameter
        lambda_k = lambda_init

        # Initialize variables for the iterative solver
        converged = False
        last_error_norm = float('inf')

        for i in range(max_iter):
            # Compute current forward kinematics
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            # Get current end-effector placement
            current_placement = self.data.oMf[self.ee_frame_id]

            # Compute the error in SE3
            err_se3 = pin.log(current_placement.inverse() * target_placement).vector
            error_norm = np.linalg.norm(err_se3)

            # Check for convergence
            if error_norm < tol:
                converged = True
                break

            # Compute the Jacobian at the current configuration
            pin.computeJointJacobians(self.model, self.data, q)
            J = pin.getFrameJacobian(self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.LOCAL)

            # Compute J^T * J and J^T * err
            JtJ = J @ J.T
            Jt_err = J.T @ err_se3

            # Add damping to the diagonal of J^T * J
            # (increase numerical stability and control step size)
            JtJ_damped = JtJ + lambda_k * np.eye(JtJ.shape[0])

            # Solve the normal equations
            try:
                v = np.linalg.solve(JtJ_damped, err_se3)
                delta_q = J.T @ v

                # Try the update
                q_new = np.clip(q + delta_q, lower_limits, upper_limits)

                # Check if the new position reduces the error
                pin.forwardKinematics(self.model, self.data, q_new)
                pin.updateFramePlacements(self.model, self.data)
                new_err_se3 = pin.log(self.data.oMf[self.ee_frame_id].inverse() * target_placement).vector
                new_error_norm = np.linalg.norm(new_err_se3)

                if new_error_norm < error_norm:
                    # Accept the update and decrease lambda (more like Gauss-Newton)
                    q = q_new
                    lambda_k = max(lambda_min, lambda_k / lambda_factor)
                    last_error_norm = new_error_norm
                else:
                    # Reject the update and increase lambda (more like gradient descent)
                    lambda_k = min(lambda_max, lambda_k * lambda_factor)
            except np.linalg.LinAlgError:
                # If matrix is singular, increase lambda and continue
                lambda_k = min(lambda_max, lambda_k * lambda_factor)
                continue

        if not converged:
            print(f"Warning: LM IK did not converge after {max_iter} iterations. Best error: {error_norm}")

        return q

    # 2. Gauss-Newton Method
    def inverse_kinematics_GaussNewton(
            self,
            target_pose: np.ndarray,
            seed: Optional[np.ndarray] = None,
            joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            max_iter: int = 5000,
            tol: float = 1e-6,
            step_size: float = 1.0
    ) -> np.ndarray:
        """
        Solve inverse kinematics using the Gauss-Newton method.

        The Gauss-Newton algorithm is similar to Newton's method but it approximates
        the Hessian as J^T * J, ignoring second-order terms.

        Args:
            target_pose: Target end-effector pose as a 4x4 homogeneous transformation matrix
            seed: Initial guess for joint angles
            joint_limits: Tuple of (lower_limits, upper_limits) arrays
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            step_size: Step size factor (1.0 = full Gauss-Newton step)

        Returns:
            joint_angles: Solved joint angles
        """
        # Get the number of joints
        n_joints = len(self.joint_lower_limit)

        # Use provided joint limits or default to model limits
        if joint_limits is None:
            lower_limits = self.joint_lower_limit
            upper_limits = self.joint_upper_limit
        else:
            lower_limits, upper_limits = joint_limits

        # Initialize joint angles with seed or middle of joint range
        if seed is None:
            q = 0.5 * (lower_limits + upper_limits)
        else:
            q = seed.copy()

        # Ensure q is within joint limits
        q = np.clip(q, lower_limits, upper_limits)

        # Convert target pose to SE3 placement
        target_placement = pin.SE3(target_pose[:3, :3], target_pose[:3, 3])

        # Initialize variables for the iterative solver
        converged = False

        for i in range(max_iter):
            # Compute current forward kinematics
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            # Get current end-effector placement
            current_placement = self.data.oMf[self.ee_frame_id]

            # Compute the error in SE3 (log maps the difference to a spatial velocity)
            err_se3 = pin.log(current_placement.inverse() * target_placement).vector

            # Check for convergence
            if np.linalg.norm(err_se3) < tol:
                converged = True
                break

            # Compute the Jacobian at the current configuration
            pin.computeJointJacobians(self.model, self.data, q)
            J = pin.getFrameJacobian(self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.LOCAL)

            # Pseudo-inverse of the Jacobian using the normal equations approach
            # For Gauss-Newton: delta_q = (J^T * J)^(-1) * J^T * err
            JtJ = J @ J.T

            try:
                # Solve the normal equations
                v = np.linalg.solve(JtJ, err_se3)
                delta_q = J.T @ v

                # Apply step size and update joint angles
                q = q + step_size * delta_q

                # Project back to joint limits
                q = np.clip(q, lower_limits, upper_limits)
            except np.linalg.LinAlgError:
                # If the matrix is singular, use a damped approach
                reg = 1e-3 * np.eye(JtJ.shape[0])
                v = np.linalg.solve(JtJ + reg, err_se3)
                delta_q = J.T @ v

                # Apply step size and update joint angles
                q = q + step_size * delta_q

                # Project back to joint limits
                q = np.clip(q, lower_limits, upper_limits)

        if not converged:
            print(
                f"Warning: Gauss-Newton IK did not converge after {max_iter} iterations. Best error: {np.linalg.norm(err_se3)}")

        return q

    # 3. L-BFGS Method
    def inverse_kinematics_LBFGS(
            self,
            target_pose: np.ndarray,
            seed: Optional[np.ndarray] = None,
            joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            max_iter: int = 5000,
            tol: float = 1e-6,
            m: int = 5  # Number of corrections to approximate the inverse Hessian
    ) -> np.ndarray:
        """
        Solve inverse kinematics using the L-BFGS (Limited-memory BFGS) method.

        L-BFGS is a quasi-Newton method that approximates the inverse Hessian
        using a limited memory of previous iterations.

        Args:
            target_pose: Target end-effector pose as a 4x4 homogeneous transformation matrix
            seed: Initial guess for joint angles
            joint_limits: Tuple of (lower_limits, upper_limits) arrays
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            m: Number of corrections to approximate the inverse Hessian

        Returns:
            joint_angles: Solved joint angles
        """
        import scipy.optimize as optimize

        # Use provided joint limits or default to model limits
        if joint_limits is None:
            lower_limits = self.joint_lower_limit
            upper_limits = self.joint_upper_limit
        else:
            lower_limits, upper_limits = joint_limits

        # Initialize joint angles with seed or middle of joint range
        if seed is None:
            q_init = 0.5 * (lower_limits + upper_limits)
        else:
            q_init = seed.copy()

        # Ensure q_init is within joint limits
        q_init = np.clip(q_init, lower_limits, upper_limits)

        # Convert target pose to SE3 placement
        target_placement = pin.SE3(target_pose[:3, :3], target_pose[:3, 3])

        # Define the objective function to minimize (squared error)
        def objective(q):
            # Ensure joint limits
            q_bounded = np.clip(q, lower_limits, upper_limits)

            # Calculate forward kinematics
            pin.forwardKinematics(self.model, self.data, q_bounded)
            pin.updateFramePlacements(self.model, self.data)

            # Calculate pose error
            current_placement = self.data.oMf[self.ee_frame_id]
            err_se3 = pin.log(current_placement.inverse() * target_placement).vector

            # Return squared error (scalar objective)
            return 0.5 * np.sum(err_se3 ** 2)

        # Define the gradient of the objective function
        def gradient(q):
            # Ensure joint limits
            q_bounded = np.clip(q, lower_limits, upper_limits)

            # Calculate forward kinematics
            pin.forwardKinematics(self.model, self.data, q_bounded)
            pin.updateFramePlacements(self.model, self.data)

            # Calculate pose error
            current_placement = self.data.oMf[self.ee_frame_id]
            err_se3 = pin.log(current_placement.inverse() * target_placement).vector

            # Calculate Jacobian
            pin.computeJointJacobians(self.model, self.data, q_bounded)
            J = pin.getFrameJacobian(self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.LOCAL)

            # Gradient is J^T * err
            return J.T @ err_se3

        # Set up the bounds for L-BFGS-B
        bounds = list(zip(lower_limits, upper_limits))

        # Run L-BFGS-B optimization
        result = optimize.minimize(
            objective,
            q_init,
            method='L-BFGS-B',
            jac=gradient,
            bounds=bounds,
            options={
                'maxiter': max_iter,
                'ftol': tol,
                'gtol': 1e-5,
                'maxcor': m  # Number of corrections used in the L-BFGS update
            }
        )

        # Get the solution
        q_solution = result.x

        # Ensure the solution is within joint limits (should already be, but just to be safe)
        q_solution = np.clip(q_solution, lower_limits, upper_limits)

        # Check convergence
        if not result.success:
            print(f"Warning: L-BFGS IK optimization did not converge. Status: {result.message}")

        return q_solution

    # 4. Newton-Raphson Method with Line Search
    def inverse_kinematics_NewtonRaphson(
            self,
            target_pose: np.ndarray,
            seed: Optional[np.ndarray] = None,
            joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            max_iter: int = 5000,
            tol: float = 1e-6,
            damping: float = 0.1,
            line_search: bool = True,
            max_line_search_iter: int = 10
    ) -> np.ndarray:
        """
        Solve inverse kinematics using the Newton-Raphson method with optional line search.

        This method uses the Jacobian pseudo-inverse with optional line search to ensure
        decrease in the error at each iteration.

        Args:
            target_pose: Target end-effector pose as a 4x4 homogeneous transformation matrix
            seed: Initial guess for joint angles
            joint_limits: Tuple of (lower_limits, upper_limits) arrays
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            damping: Damping factor for the pseudo-inverse
            line_search: Whether to use line search
            max_line_search_iter: Maximum iterations for line search

        Returns:
            joint_angles: Solved joint angles
        """
        # Get the number of joints
        n_joints = len(self.joint_lower_limit)

        # Use provided joint limits or default to model limits
        if joint_limits is None:
            lower_limits = self.joint_lower_limit
            upper_limits = self.joint_upper_limit
        else:
            lower_limits, upper_limits = joint_limits

        # Initialize joint angles with seed or middle of joint range
        if seed is None:
            q = 0.5 * (lower_limits + upper_limits)
        else:
            q = seed.copy()

        # Ensure q is within joint limits
        q = np.clip(q, lower_limits, upper_limits)

        # Convert target pose to SE3 placement
        target_placement = pin.SE3(target_pose[:3, :3], target_pose[:3, 3])

        # Calculate the initial error
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        # Get current end-effector placement
        current_placement = self.data.oMf[self.ee_frame_id]
        err_se3 = pin.log(current_placement.inverse() * target_placement).vector
        initial_error = np.linalg.norm(err_se3)

        # Function to compute the error norm at a given configuration
        def compute_error(joint_config):
            # Compute FK at the given configuration
            pin.forwardKinematics(self.model, self.data, joint_config)
            pin.updateFramePlacements(self.model, self.data)

            # Compute the error vector
            current_pose = self.data.oMf[self.ee_frame_id]
            error = pin.log(current_pose.inverse() * target_placement).vector

            # Return the error norm
            return np.linalg.norm(error)

        # Iterate to solve the IK
        for i in range(max_iter):
            # Compute current forward kinematics
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            # Get current end-effector placement
            current_placement = self.data.oMf[self.ee_frame_id]

            # Compute the error in SE3
            err_se3 = pin.log(current_placement.inverse() * target_placement).vector
            error_norm = np.linalg.norm(err_se3)

            # Check for convergence
            if error_norm < tol:
                converged = True
                break

            # Compute the Jacobian at the current configuration
            pin.computeJointJacobians(self.model, self.data, q)
            J = pin.getFrameJacobian(self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.LOCAL)

            # Compute the damped pseudo-inverse
            J_pinv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(6))

            # Compute Newton step
            delta_q = J_pinv @ err_se3

            # Apply line search if enabled
            if line_search and error_norm > tol:
                alpha = 1.0  # Initial step size
                beta = 0.5  # Reduction factor
                current_error = error_norm

                # Line search iterations
                for j in range(max_line_search_iter):
                    # Try the new configuration
                    q_new = np.clip(q + alpha * delta_q, lower_limits, upper_limits)
                    new_error = compute_error(q_new)

                    # If error decreased, accept the step
                    if new_error < current_error:
                        q = q_new
                        break

                    # Otherwise, reduce the step size
                    alpha *= beta

                    # If step size becomes too small, break
                    if alpha < 1e-5:
                        break

                # If line search failed to improve (all steps rejected), take a small step
                if j == max_line_search_iter - 1 or alpha < 1e-5:
                    q = np.clip(q + 1e-3 * delta_q / np.linalg.norm(delta_q), lower_limits, upper_limits)
            else:
                # No line search, just update with full step
                q = np.clip(q + delta_q, lower_limits, upper_limits)

        if i == max_iter - 1:
            print(f"Warning: Newton-Raphson IK did not converge after {max_iter} iterations. Best error: {error_norm}")

        return q

    # def inverse_kinematics_pino(
    #         self,
    #         target_pose: np.ndarray,
    #         seed: Optional[np.ndarray] = None,
    #         joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    #         max_iter: int = 1000,
    #         eps: float = 1e-4,
    #         dt: float = 1e-1,
    #         damp: float = 1e-12
    # ) -> np.ndarray:
    #     """
    #     Solve inverse kinematics using an iterative damped least squares approach
    #     with Jlog6 correction (based on SE(3) pose error).
    #
    #     Args:
    #         target_pose: Target end-effector pose as a 4x4 homogeneous transformation matrix
    #         seed: Optional initial joint configuration (shape: (n,))
    #         joint_limits: Optional joint limits (lower_limits, upper_limits)
    #         max_iter: Maximum number of iterations
    #         eps: Convergence threshold
    #         dt: Step size for integration
    #         damp: Damping factor for stability
    #
    #     Returns:
    #         joint_angles: Solved joint configuration (shape: (n,))
    #     """
    #     n_joints = len(self.joint_lower_limit)
    #
    #     # Use provided or default joint limits
    #     lower_limits, upper_limits = (self.joint_lower_limit, self.joint_upper_limit) if joint_limits is None else joint_limits
    #
    #     # Initialize q
    #     if seed is None:
    #         q = 0.5 * (lower_limits + upper_limits)
    #     else:
    #         q = np.clip(seed.copy(), lower_limits, upper_limits)
    #
    #     # Target pose
    #     target_placement = pin.SE3(target_pose[:3, :3], target_pose[:3, 3])
    #
    #     success = False
    #
    #     for i in range(max_iter):
    #         # Forward kinematics
    #         pin.forwardKinematics(self.model, self.data, q)
    #         pin.updateFramePlacements(self.model, self.data)
    #
    #         # Compute error
    #         current_placement = self.data.oMf[self.ee_frame_id]
    #         iMd = current_placement.inverse() * target_placement
    #         err = pin.log(iMd).vector
    #
    #         if np.linalg.norm(err) < eps:
    #             success = True
    #             break
    #
    #         # Jacobian
    #         pin.computeJointJacobians(self.model, self.data, q)
    #         J = pin.getFrameJacobian(self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.LOCAL)
    #
    #         # Correct Jacobian with Jlog6
    #         J = -np.dot(pin.Jlog6(iMd.inverse()), J)
    #
    #         # Solve for velocity
    #         JJT = J @ J.T + damp * np.eye(6)
    #         v = -J.T @ np.linalg.solve(JJT, err)
    #
    #         # Update q
    #         q = pin.integrate(self.model, q, v * dt)
    #         q = np.clip(q, lower_limits, upper_limits)
    #
    #         if i % 100 == 0:
    #             print(f"Iteration {i}: error norm = {np.linalg.norm(err):.6f}")
    #
    #     if not success:
    #         print(f"Warning: Pino IK did not converge after {max_iter} iterations. Final error: {np.linalg.norm(err):.6f}")
    #
    #     return q
    def inverse_kinematics_pino(self, target_pose, seed=None, joint_limits=None, max_iter=1000, eps=1e-4, dt=0.1,
                                     damp=1e-12):
        model = self.model  # 局部变量，避免self.model每次访问
        data = self.data

        lower_limits, upper_limits = (
        self.joint_lower_limit, self.joint_upper_limit) if joint_limits is None else joint_limits
        q = 0.5 * (lower_limits + upper_limits) if seed is None else np.clip(seed.copy(), lower_limits, upper_limits)

        target_placement = pin.SE3(target_pose[:3, :3], target_pose[:3, 3])
        success = False

        for i in range(max_iter):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)

            current_placement = data.oMf[self.ee_frame_id]
            iMd = current_placement.inverse() * target_placement
            err = pin.log(iMd).vector

            if np.linalg.norm(err) < eps:
                success = True
                break

            pin.computeJointJacobians(model, data, q)
            J = pin.getFrameJacobian(model, data, self.ee_frame_id, pin.ReferenceFrame.LOCAL)
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)

            JJT = J @ J.T + damp * np.eye(6)
            v = -J.T @ np.linalg.solve(JJT, err)

            q += v * dt
            # Only clip if necessary
            if np.any(q < lower_limits) or np.any(q > upper_limits):
                q = np.clip(q, lower_limits, upper_limits)

            # NO PRINT, save time
        if not success:
            print(
                f"Warning: Pino IK did not converge after {max_iter} iterations. Final error: {np.linalg.norm(err):.6f}")

        return q


class rtbKinematicsModel(BaseKinematicsModel):
    None


class kdlKinematicsModel(BaseKinematicsModel):
    None







