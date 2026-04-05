#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
import tempfile
import pytest

# Import our custom kinematics implementation
from kinematics_rtb import RTBKinematicsModel

def calculate_pose_error(T1: np.ndarray, T2: np.ndarray) -> tuple:
    """
    Calculate position and orientation error between two homogeneous transformation matrices.

    Args:
        T1: First homogeneous transformation matrix (4x4)
        T2: Second homogeneous transformation matrix (4x4)

    Returns:
        tuple: (position_error, orientation_error)
            - position_error: Euclidean distance between positions
            - orientation_error: Angle of the equivalent axis-angle representation
    """
    # Calculate position error (Euclidean distance)
    pos_error = np.linalg.norm(T1[:3, 3] - T2[:3, 3])

    # Calculate orientation error using the rotation matrix
    # Convert to SE3 for easier computation
    se3_T1 = SE3(T1)
    se3_T2 = SE3(T2)

    # Get relative rotation
    rel_rot = se3_T1.inv() * se3_T2

    # Extract angle from axis-angle representation
    _, angle = rel_rot.angvec()

    return pos_error, angle


@pytest.fixture
def robot_model():
    # Create a Panda robot model
    # robot = rtb.models.Panda()


    # Save the robot model to a temporary URDF file
    temp_dir = tempfile.mkdtemp()
    urdf_file = os.path.join(temp_dir, "panda.urdf")
    robot = rtb.ERobot.URDF(urdf_file)
    # print(f"Saved robot URDF to {urdf_file}")

    # Create our kinematics model
    kin_model = RTBKinematicsModel(urdf_file)

    return kin_model, robot


def test_joint_limits(robot_model):
    """Test getting joint limits from the model."""
    kin_model, robot = robot_model

    # Get joint limits
    lower_limits, upper_limits = kin_model.get_joint_limits()

    # Assert that we have the correct number of limits
    assert len(lower_limits) == robot.n
    assert len(upper_limits) == robot.n

    # Assert that lower limits are less than upper limits
    assert np.all(lower_limits < upper_limits)


def test_forward_kinematics(robot_model):
    """Test the forward kinematics implementation."""
    kin_model, robot = robot_model

    # Get joint limits
    lower_limits, upper_limits = kin_model.get_joint_limits()

    # Generate random joint position within limits
    random_values = np.random.random(robot.n)
    q_rand = lower_limits + random_values * (upper_limits - lower_limits)

    # Test forward kinematics
    fk_pose = kin_model.forward_kinematics(q_rand)

    # Assert that the result is a 4x4 homogeneous transformation matrix
    assert fk_pose.shape == (4, 4)

    # Compare with the robot's own FK to make sure they're close
    robot_fk = robot.fkine(q_rand).A
    pos_error, orient_error = calculate_pose_error(fk_pose, robot_fk)

    # Assert that the errors are small
    assert pos_error < 1e-6
    assert orient_error < 1e-6


def test_inverse_kinematics(robot_model):
    """Test the inverse kinematics implementation with FK-IK cycle."""
    kin_model, robot = robot_model

    # Get joint limits
    lower_limits, upper_limits = kin_model.get_joint_limits()

    # Generate random joint position within limits
    random_values = np.random.random(robot.n)
    q_rand = lower_limits + random_values * (upper_limits - lower_limits)

    # Get the forward kinematics pose
    fk_pose = kin_model.forward_kinematics(q_rand)

    # Test inverse kinematics
    ik_joints = kin_model.inverse_kinematics(fk_pose, seed=q_rand)

    # Assert that the result has the right shape
    assert len(ik_joints) == robot.n

    # Check if IK solution gives the same pose
    fk_pose_from_ik = kin_model.forward_kinematics(ik_joints)

    # Calculate pose error
    pos_error, orient_error = calculate_pose_error(fk_pose, fk_pose_from_ik)

    # Assert that the errors are small
    assert pos_error < 1e-3  # Position error less than 1mm
    assert orient_error < 1e-2  # Orientation error less than ~0.5 degrees


def test_jacobian(robot_model):
    """Test the Jacobian calculation."""
    kin_model, robot = robot_model

    # Get joint limits
    lower_limits, upper_limits = kin_model.get_joint_limits()

    # Generate random joint position within limits
    random_values = np.random.random(robot.n)
    q_rand = lower_limits + random_values * (upper_limits - lower_limits)

    # Test Jacobian
    J = kin_model.jacobian(q_rand)

    # Assert that the Jacobian has the right shape
    assert J.shape == (6, robot.n)

    # Compare with the robot's own Jacobian to make sure they're close
    robot_J = robot.jacobe(q_rand)
    J_error = np.linalg.norm(J - robot_J)

    # Assert that the error is small
    assert J_error < 1e-6


def test_inverse_velocity_kinematics(robot_model):
    """Test the inverse velocity kinematics implementation."""
    kin_model, robot = robot_model

    # Get joint limits
    lower_limits, upper_limits = kin_model.get_joint_limits()

    # Generate random joint position within limits
    random_values = np.random.random(robot.n)
    q_rand = lower_limits + random_values * (upper_limits - lower_limits)

    # Create a random twist (small to avoid joint limit issues)
    twist = np.random.uniform(-0.1, 0.1, 6)

    # Test inverse velocity kinematics
    joint_velocities = kin_model.inverse_velocity_kinematics(twist, q_rand)

    # Assert that the result has the right shape
    assert len(joint_velocities) == robot.n

    # Get the Jacobian
    J = kin_model.jacobian(q_rand)

    # Verify that J * qd ≈ twist
    reconstructed_twist = J @ joint_velocities
    twist_error = np.linalg.norm(twist - reconstructed_twist)

    # Assert that the error is small
    assert twist_error < 1e-6


def test_joint_limit_handling(robot_model):
    """Test that joint limits are respected in IK and velocity IK."""
    kin_model, robot = robot_model

    # Get joint limits
    orig_lower_limits, orig_upper_limits = kin_model.get_joint_limits()

    # Create tighter joint limits for testing
    lower_limits = orig_lower_limits + 0.1 * (orig_upper_limits - orig_lower_limits)
    upper_limits = orig_upper_limits - 0.1 * (orig_upper_limits - orig_lower_limits)
    joint_limits = (lower_limits, upper_limits)

    # Generate random joint position at the limits
    q_at_limit = upper_limits.copy()  # At upper limit

    # Get FK pose
    fk_pose = kin_model.forward_kinematics(q_at_limit)

    # Try IK with limits
    ik_joints = kin_model.inverse_kinematics(fk_pose, seed=q_at_limit, joint_limits=joint_limits)

    # Assert joints are within limits
    assert np.all(ik_joints >= lower_limits)
    assert np.all(ik_joints <= upper_limits)

    # Test velocity IK with limits
    # Create a twist that would cause movement beyond limits
    twist = np.ones(6) * 0.1

    # Get velocities with limit handling
    joint_velocities = kin_model.inverse_velocity_kinematics(
        twist, q_at_limit, joint_limits=joint_limits
    )

    # Verify that velocities that would exceed limits are zero
    expected_zeros = np.logical_or(
        np.isclose(q_at_limit, upper_limits) & (joint_velocities > 0),
        np.isclose(q_at_limit, lower_limits) & (joint_velocities < 0)
    )

    # Check that velocities that would exceed limits are close to zero
    for i in range(len(joint_velocities)):
        if expected_zeros[i]:
            assert abs(joint_velocities[i]) < 1e-10


# Run the tests if the script is executed directly
if __name__ == "__main__":
    pytest.main(["-v", __file__])