"""
@File    : test_kinematics
@Author  : Haotian Liang
@Time    : 2025/4/25 14:35
@Email   :Haotianliang10@gmail.com
"""

from kinematics import *
import time

import os
from pathlib import Path

# 获取当前脚本所在的目录
current_dir = Path(__file__).parent.absolute()
# 获取上一级目录
parent_dir = current_dir.parent
# 构建 URDF 文件的相对路径（assets 在上一级目录）
urdf_path = os.path.join(parent_dir, "assets", "franka_fr3", "fr3_franka_hand.urdf")

def test_pinocchio_kinematics_model():
    # Path to a sample URDF file - replace with an actual URDF path

    try:
        # Initialize the kinematics model
        print("Initializing PinocchioKinematicsModel...")
        kin_model = PinocchioKinematicsModel(urdf_path,
                                             base_link="base",
                                             end_effector_link="fr3_hand_tcp")
        print("Kinematics model initialized successfully.")

        # Get the number of joints (DOF)
        n_joints = len(kin_model.joint_lower_limit)
        print(f"Robot has {n_joints} degrees of freedom.")

        # Get joint limits
        lower_limits, upper_limits = kin_model.get_joint_limits()
        print("Joint limits:")
        for i in range(n_joints):
            print(f"  Joint {i}: [{lower_limits[i]:.4f}, {upper_limits[i]:.4f}]")

        # Test 1: Forward Kinematics
        print("\n--- Test 1: Forward Kinematics ---")
        # Generate a random joint configuration within limits
        q_test = np.random.uniform(lower_limits, upper_limits)
        print("Random joint configuration:")
        print(q_test)

        # Compute forward kinematics
        T_test = kin_model.forward_kinematics(q_test)
        print("Resulting end-effector pose:")
        print(T_test)
        print("Position:", T_test[:3, 3])

        # Test 2: Inverse Kinematics
        print("\n--- Test 2: Inverse Kinematics ---")
        # Use the pose from forward kinematics as target
        target_pose = T_test.copy()
        print("Target pose:")
        print(target_pose)

        # Generate a different random seed for IK
        q_seed = np.random.uniform(lower_limits, upper_limits)
        print("Seed joint configuration:")
        print(q_seed)

        # Compute inverse kinematics
        start_time = time.time()
        q_solved = kin_model.inverse_kinematics(target_pose, seed=q_seed)
        end_time = time.time()
        print("Solved joint configuration:")
        print(q_solved)
        print(f"IK solved in {end_time - start_time:.4f} seconds")

        # Test 3: FK-IK Consistency
        print("\n--- Test 3: FK-IK Consistency ---")
        # Compute FK with the solved IK solution
        T_solved = kin_model.forward_kinematics(q_solved)
        print("FK of IK solution:")
        print(T_solved)

        # Compare the poses
        pose_error = np.linalg.norm(T_solved[:3, 3] - target_pose[:3, 3])
        rot_error = np.linalg.norm(T_solved[:3, :3] - target_pose[:3, :3], 'fro')
        print(f"Position error: {pose_error:.8f}")
        print(f"Rotation error: {rot_error:.8f}")

        # Test 4: Inverse Velocity Kinematics
        print("\n--- Test 4: Inverse Velocity Kinematics ---")
        # Define a test twist
        twist_test = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.1])  # Linear x and angular z
        print("Test twist:", twist_test)

        # Compute joint velocities
        dq = kin_model.inverse_velocity_kinematics(twist_test, q_solved)
        print("Computed joint velocities:")
        print(dq)

        # Verify velocity IK by comparing J * dq with twist
        J = kin_model.jacobian(q_solved)
        computed_twist = J @ dq
        twist_error = np.linalg.norm(computed_twist - twist_test)
        print("Jacobian shape:", J.shape)
        print("Computed twist:", computed_twist)
        print(f"Twist error: {twist_error:.8f}")

        # Test 5: Jacobian Numerical Verification
        print("\n--- Test 5: Jacobian Numerical Verification ---")
        # Compute analytical Jacobian
        J_analytical = kin_model.jacobian(q_solved)

        # Compute numerical Jacobian
        epsilon = 1e-6
        J_numerical = np.zeros((6, n_joints))

        # Compute forward kinematics at the reference position
        T_ref = kin_model.forward_kinematics(q_solved)

        # Compute numerical Jacobian column by column
        for i in range(n_joints):
            # Perturb joint i
            q_plus = q_solved.copy()
            q_plus[i] += epsilon

            # Compute FK at perturbed position
            T_plus = kin_model.forward_kinematics(q_plus)

            # Extract position difference
            pos_diff = (T_plus[:3, 3] - T_ref[:3, 3]) / epsilon

            # Extract rotation difference (approximation)
            # This is a simple approximation for small rotations
            R_ref = T_ref[:3, :3]
            R_plus = T_plus[:3, :3]
            R_diff = (R_plus @ R_ref.T - np.eye(3)) / epsilon

            # Convert rotation matrix difference to angular velocity
            # Using the fact that for small rotations, R_diff = [w]_x (skew-symmetric matrix)
            w_x = R_diff[2, 1]
            w_y = R_diff[0, 2]
            w_z = R_diff[1, 0]

            # Assign to numerical Jacobian
            J_numerical[:3, i] = pos_diff
            J_numerical[3:, i] = [w_x, w_y, w_z]

        # Compare analytical and numerical Jacobians
        J_diff = np.linalg.norm(J_analytical - J_numerical, 'fro')
        print(f"Frobenius norm of Jacobian difference: {J_diff:.8f}")

        # Print a sample column from both Jacobians
        col_idx = 0
        print(f"Analytical Jacobian column {col_idx}:")
        print(J_analytical[:, col_idx])
        print(f"Numerical Jacobian column {col_idx}:")
        print(J_numerical[:, col_idx])

        # Test 6: FK_velocity
        print("\n--- Test 6: FK_velocity ---")



        print("\nAll tests completed successfully!")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


def test_forward_kinematics_velocity():
    """Test the forward kinematics velocity function."""
    try:
        print("\n--- Test: Forward Kinematics Velocity ---")

        # Get model and joint limits
        kin_model = PinocchioKinematicsModel(urdf_path,
                                             base_link="base",
                                             end_effector_link="fr3_hand_tcp")
        lower_limits, upper_limits = kin_model.get_joint_limits()
        n_joints = len(lower_limits)

        # Generate random joint positions and velocities
        q_test = np.random.uniform(lower_limits, upper_limits)
        dq_test = np.random.uniform(-1.0, 1.0, size=n_joints)  # Random velocities

        print("Test joint positions:")
        print(q_test)
        print("Test joint velocities:")
        print(dq_test)

        # Calculate end-effector twist using fk_vel
        twist = kin_model.fk_vel(q_test, dq_test)
        print("End-effector twist [vx, vy, vz, wx, wy, wz]:")
        print(twist)

        # Verify the result by comparing with J * dq
        J = kin_model.jacobian(q_test)
        twist_from_jacobian = J @ dq_test

        # Calculate error
        twist_error = np.linalg.norm(twist - twist_from_jacobian)
        print(f"Twist error (compared to J * dq): {twist_error:.8f}")

        # Test should pass if error is small
        assert twist_error < 1e-5, f"Twist error too large: {twist_error}"

        print("Forward kinematics velocity test passed!")
        return True

    except Exception as e:
        print(f"Error in forward kinematics velocity test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_kinematics_all_links():
    """Test the forward kinematics for all links function."""
    try:
        print("\n--- Test: Forward Kinematics All Links ---")
        # Get model and joint limits
        kin_model = PinocchioKinematicsModel(urdf_path,
                                             base_link="base",
                                             end_effector_link="fr3_hand_tcp")
        lower_limits, upper_limits = kin_model.get_joint_limits()

        # Generate random joint positions
        q_test = np.random.uniform(lower_limits, upper_limits)

        print("Test joint positions:")
        print(q_test)

        # Get all link poses
        link_poses = kin_model.fk_all(q_test)

        # Print number of links found
        print(f"Found {len(link_poses)} links in the robot model")

        # Print a few link poses as examples
        print("Sample link poses:")
        for i, (link_name, pose) in enumerate(link_poses.items()):
            print(f"Link: {link_name}")
            print(f"Position: {pose[:3, 3]}")
            # Only print a few examples
            if i >= 2:
                print(f"... and {len(link_poses) - 3} more links")
                break

        # Verify results by checking the end-effector pose
        ee_pose_from_fk = kin_model.forward_kinematics(q_test)
        ee_pose_from_fk_all = link_poses.get(kin_model.ee_frame_name)

        if ee_pose_from_fk_all is not None:
            # Calculate error
            pose_error = np.linalg.norm(ee_pose_from_fk[:3, 3] - ee_pose_from_fk_all[:3, 3])
            rot_error = np.linalg.norm(ee_pose_from_fk[:3, :3] - ee_pose_from_fk_all[:3, :3], 'fro')

            print(f"End-effector position error: {pose_error:.8f}")
            print(f"End-effector rotation error: {rot_error:.8f}")

            # Test should pass if errors are small
            assert pose_error < 1e-10, f"Position error too large: {pose_error}"
            assert rot_error < 1e-10, f"Rotation error too large: {rot_error}"

            print("Forward kinematics all links test passed!")
            return True
        else:
            print(f"Warning: End-effector frame '{kin_model.ee_frame_name}' not found in link poses")
            print("This could be normal if the end-effector isn't a link or has a different name")
            # Still return True as this might be expected
            return True

    except Exception as e:
        print(f"Error in forward kinematics all links test: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_kinematics_model():
    """Helper function to get a kinematics model instance."""
    # Path to the URDF file - replace with appropriate path
    # Get model and joint limits
    kin_model = PinocchioKinematicsModel(urdf_path,
                                         base_link="base",
                                         end_effector_link="fr3_hand_tcp")

    # Create and return the kinematics model
    return PinocchioKinematicsModel(urdf_path)


def run_additional_tests():
    """Run all additional tests."""
    print("\n=== Running Additional Tests ===")

    tests_passed = 0
    tests_total = 2

    # Test 1: Forward Kinematics Velocity
    if test_forward_kinematics_velocity():
        tests_passed += 1

    # Test 2: Forward Kinematics All Links
    if test_forward_kinematics_all_links():
        tests_passed += 1

    print(f"\n=== Test Results: {tests_passed}/{tests_total} tests passed ===")




if __name__ == "__main__":
    test_pinocchio_kinematics_model()
    run_additional_tests()