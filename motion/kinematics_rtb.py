"""
@File    : kinematics_rtb
@Author  : Haotian Liang
@Time    : 2025/4/28 12:36
@Email   :Haotianliang10@gmail.com
"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import Optional, Tuple
import roboticstoolbox as rtb
from spatialmath import SE3
from kinematics import BaseKinematicsModel


class RTBKinematicsModel(BaseKinematicsModel):
    """
    Implementation of robot kinematics modeling using Robotics Toolbox for Python.
    Provides methods for forward kinematics, inverse kinematics,
    Jacobian calculation, and inverse velocity kinematics.
    """

    def __init__(self, urdf_path: str, gripper: Optional[str] = None):
        """
        Initialize the RTB kinematics model from a URDF file.

        Args:
            urdf_path: Path to the URDF file describing the robot
            gripper: Optional gripper name or configuration
        """
        # Load the robot model from URDF
        self.robot = rtb.models.URDF.from_file(urdf_path)

        # Store the gripper information if provided
        self.gripper = gripper

        # Cache robot's degrees of freedom (number of joints)
        self.dof = self.robot.n

        # Print robot information for debugging
        print(f"Loaded robot model with {self.dof} DOF")

    def forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Calculate the end-effector pose from given joint positions.

        Args:
            joint_positions: Joint positions array of shape (n,) where n is the number of joints

        Returns:
            pose: End-effector pose as a homogeneous transformation matrix of shape (4, 4)
        """
        # Validate input dimensions
        if len(joint_positions) != self.dof:
            raise ValueError(f"Expected {self.dof} joint positions, got {len(joint_positions)}")

        # Calculate forward kinematics
        T = self.robot.fkine(joint_positions)

        # Return the homogeneous transformation matrix
        return T.A  # Convert from SE3 to numpy array (4x4)

    def inverse_kinematics(
            self,
            target_pose: np.ndarray,
            seed: Optional[np.ndarray] = None,
            joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Calculate joint positions to achieve a target end-effector pose using Levenberg-Marquardt.

        Args:
            target_pose: Target end-effector pose as a homogeneous transformation matrix
                        of shape (4, 4)
            seed: Initial guess for joint positions, shape (n,)
            joint_limits: Tuple of (lower_limits, upper_limits) arrays of shape (n,)

        Returns:
            joint_positions: Calculated joint positions of shape (n,)

        Raises:
            RuntimeError: If inverse kinematics solution cannot be found
        """
        # Use robot's joint limits if not provided
        if joint_limits is None:
            lower_limits, upper_limits = self.get_joint_limits()
        else:
            lower_limits, upper_limits = joint_limits

        # Use random valid configuration as seed if not provided
        if seed is None:
            # Generate random values between 0 and 1, then scale to joint limits
            random_values = np.random.random(self.dof)
            seed = lower_limits + random_values * (upper_limits - lower_limits)

        # Convert target pose to SE3 if it's a numpy array
        if isinstance(target_pose, np.ndarray):
            target_pose = SE3(target_pose)

        # Solve inverse kinematics using Levenberg-Marquardt method
        solution = self.robot.ikine_LM(
            target_pose,
            q0=seed,
            ilimit=500,  # Iteration limit
            rlimit=100,  # Lambda limit
            tol=1e-8,  # Tolerance
            mask=[1, 1, 1, 1, 1, 1]  # Consider all 6 DOF in task space
        )

        # Check if a solution was found
        if not solution.success:
            raise RuntimeError(
                f"Inverse kinematics failed to converge: {solution.reason}. "
                f"Final error: {solution.residual}"
            )

        # Clip solution to joint limits if provided
        joint_positions = np.clip(solution.q, lower_limits, upper_limits)

        return joint_positions

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
        if len(joint_positions) != self.dof:
            raise ValueError(f"Expected {self.dof} joint positions, got {len(joint_positions)}")

        # Calculate the Jacobian using RTB
        J = self.robot.jacobe(joint_positions)

        # RTB returns the Jacobian matrix in the end-effector frame
        # Return as numpy array with shape (6, n)
        return J

    def inverse_velocity_kinematics(
            self,
            twist: np.ndarray,
            joint_positions: np.ndarray,
            joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Calculate joint velocities to achieve a desired end-effector twist
        using the pseudoinverse of the Jacobian.

        Args:
            twist: Desired end-effector twist vector of shape (6,)
                  (linear and angular velocities)
            joint_positions: Current joint positions of shape (n,)
            joint_limits: Tuple of (lower_limits, upper_limits) arrays of shape (n,)

        Returns:
            joint_velocities: Calculated joint velocities of shape (n,)
        """
        # Validate input dimensions
        if len(joint_positions) != self.dof:
            raise ValueError(f"Expected {self.dof} joint positions, got {len(joint_positions)}")
        if len(twist) != 6:
            raise ValueError(f"Expected twist vector of size 6, got {len(twist)}")

        # Calculate the Jacobian
        J = self.jacobian(joint_positions)

        # Calculate the pseudoinverse using SVD for better numerical stability
        J_pinv = np.linalg.pinv(J)

        # Calculate joint velocities using pseudoinverse
        joint_velocities = J_pinv @ twist

        # If joint limits are provided, zero out velocities that would exceed limits
        if joint_limits is not None:
            lower_limits, upper_limits = joint_limits

            # Check joints at their limits and trying to move beyond
            at_lower = np.logical_and(joint_positions <= lower_limits, joint_velocities < 0)
            at_upper = np.logical_and(joint_positions >= upper_limits, joint_velocities > 0)

            # Zero out those velocities
            joint_velocities[at_lower] = 0.0
            joint_velocities[at_upper] = 0.0

        return joint_velocities

    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the joint limits of the robot.

        Returns:
            joint_limits: Tuple of (lower_limits, upper_limits) arrays of shape (n,)
        """
        # Extract joint limits from the robot model
        lower_limits = np.array([joint.qlim[0] for joint in self.robot.joints
                                 if joint.jtype != "fixed"])
        upper_limits = np.array([joint.qlim[1] for joint in self.robot.joints
                                 if joint.jtype != "fixed"])

        # Handle case where limits might be None or NaN
        for i in range(self.dof):
            if lower_limits[i] is None or np.isnan(lower_limits[i]):
                lower_limits[i] = -np.pi
            if upper_limits[i] is None or np.isnan(upper_limits[i]):
                upper_limits[i] = np.pi

        return lower_limits, upper_limits
