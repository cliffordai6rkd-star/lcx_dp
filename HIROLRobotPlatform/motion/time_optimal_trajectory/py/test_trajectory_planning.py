"""
@File    : test_trajectory_planning
@Author  : Haotian Liang
@Time    : 2025/5/27 16:14
@Email   :Haotianliang10@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

# Add the parent directory to the path to import kinematics module
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from motion.kinematics_ import PinocchioKinematicsModel
from trajectory_planning import (
    PolynomialTrajectoryPlanner,
    CartesianTrajectoryPlanner,
    smooth_trajectory,
    check_trajectory_feasibility
)

# URDF path setup
urdf_path = os.path.join(parent_dir, "assets", "franka_fr3", "fr3_franka_hand.urdf")


def test_joint_space_trajectory():
    """Test joint space trajectory planning."""
    print("\n=== Testing Joint Space Trajectory Planning ===")

    try:
        # Initialize kinematics model
        kin_model = PinocchioKinematicsModel(
            urdf_path,
            base_link="base",
            end_effector_link="fr3_hand_tcp"
        )

        # Get joint limits
        lower_limits, upper_limits = kin_model.get_joint_limits()
        n_joints = len(lower_limits)

        print(f"Robot has {n_joints} joints")

        # Create trajectory planner
        planner = PolynomialTrajectoryPlanner(dt=0.01)

        # Define waypoints
        waypoints = []
        waypoints.append(np.zeros(n_joints))  # Home position
        waypoints.append(0.3 * (lower_limits + upper_limits))  # Middle position
        waypoints.append(0.7 * (lower_limits + upper_limits))  # Another position
        waypoints.append(np.zeros(n_joints))  # Back to home

        print(f"Planning trajectory through {len(waypoints)} waypoints")

        # Test different planning methods
        methods = ["cubic_spline", "quintic", "trapezoidal"]

        for method in methods:
            print(f"\n--- Testing {method} method ---")

            # Plan trajectory
            times, positions, velocities, accelerations = planner.plan_joint_trajectory(
                waypoints, method=method
            )

            print(f"Trajectory duration: {times[-1]:.2f} seconds")
            print(f"Number of trajectory points: {len(times)}")

            # Check feasibility
            feasibility = check_trajectory_feasibility(
                positions, velocities, accelerations, (lower_limits, upper_limits)
            )

            print(f"Position feasible: {feasibility['position_feasible']}")
            print(f"Velocity feasible: {feasibility['velocity_feasible']}")
            print(f"Acceleration feasible: {feasibility['acceleration_feasible']}")

            if feasibility['violations']:
                print("Violations:", feasibility['violations'])

            # Plot results for the first joint
            plt.figure(figsize=(12, 8))

            plt.subplot(3, 1, 1)
            plt.plot(times, positions[:, 0], 'b-', linewidth=2)
            plt.ylabel('Position (rad)')
            plt.title(f'Joint 1 Trajectory - {method.title()}')
            plt.grid(True)

            plt.subplot(3, 1, 2)
            plt.plot(times, velocities[:, 0], 'g-', linewidth=2)
            plt.ylabel('Velocity (rad/s)')
            plt.grid(True)

            plt.subplot(3, 1, 3)
            plt.plot(times, accelerations[:, 0], 'r-', linewidth=2)
            plt.ylabel('Acceleration (rad/s²)')
            plt.xlabel('Time (s)')
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(f'joint_trajectory_{method}.png', dpi=150, bbox_inches='tight')
            plt.show()

        print("Joint space trajectory planning test completed successfully!")
        return True

    except Exception as e:
        print(f"Error in joint space trajectory test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cartesian_trajectory():
    """Test Cartesian space trajectory planning."""
    print("\n=== Testing Cartesian Space Trajectory Planning ===")

    try:
        # Initialize kinematics model
        kin_model = PinocchioKinematicsModel(
            urdf_path,
            base_link="base",
            end_effector_link="fr3_hand_tcp"
        )

        # Create Cartesian trajectory planner
        cart_planner = CartesianTrajectoryPlanner(kin_model, dt=0.01)

        # Get a reference pose by forward kinematics
        lower_limits, upper_limits = kin_model.get_joint_limits()
        q_home = np.array([1.93522609e-03, -7.83874706e-01, 3.44093733e-04,
               -2.35621553e+00, -4.98195832e-03, 1.57430503e+00, 7.78670807e-01,0,0])
        T_home = kin_model.forward_kinematics(q_home)

        print("Home pose:")
        print(f"Position: {T_home[:3, 3]}")
        print(f"Orientation:\n{T_home[:3, :3]}")

        # Test straight line trajectory
        print("\n--- Testing straight line trajectory ---")

        # Define start and end poses
        start_pose = T_home.copy()
        end_pose = T_home.copy()
        end_pose[:3, 3] += np.array([0.1, 0.1, 0.1])  # Move 10cm in each direction

        # Plan straight line trajectory
        duration = 5.0
        trajectory = cart_planner.plan_straight_line(
            start_pose, end_pose, duration, seed_joint_config=q_home
        )

        times = trajectory['time']
        joint_positions = trajectory['joint_positions']
        cartesian_positions = trajectory['cartesian_positions']

        print(f"Straight line trajectory planned with {len(times)} points")
        print(f"Start position: {cartesian_positions[0]}")
        print(f"End position: {cartesian_positions[-1]}")

        # Verify end-effector positions using forward kinematics
        T_start_verify = kin_model.forward_kinematics(joint_positions[0])
        T_end_verify = kin_model.forward_kinematics(joint_positions[-1])

        start_error = np.linalg.norm(T_start_verify[:3, 3] - cartesian_positions[0])
        end_error = np.linalg.norm(T_end_verify[:3, 3] - cartesian_positions[-1])

        print(f"Start position error: {start_error:.6f}")
        print(f"End position error: {end_error:.6f}")

        # Plot Cartesian trajectory
        plt.figure(figsize=(15, 10))

        # 3D trajectory plot
        ax1 = plt.subplot(2, 2, 1, projection='3d')
        ax1.plot(cartesian_positions[:, 0],
                 cartesian_positions[:, 1],
                 cartesian_positions[:, 2], 'b-', linewidth=2)
        ax1.scatter(cartesian_positions[0, 0], cartesian_positions[0, 1], cartesian_positions[0, 2],
                    color='green', s=100, label='Start')
        ax1.scatter(cartesian_positions[-1, 0], cartesian_positions[-1, 1], cartesian_positions[-1, 2],
                    color='red', s=100, label='End')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory')
        ax1.legend()

        # Individual coordinate plots
        for i, coord in enumerate(['X', 'Y', 'Z']):
            plt.subplot(2, 2, i + 2)
            plt.plot(times, cartesian_positions[:, i], 'b-', linewidth=2)
            plt.ylabel(f'{coord} Position (m)')
            plt.xlabel('Time (s)')
            plt.title(f'{coord} Coordinate vs Time')
            plt.grid(True)

        plt.tight_layout()
        plt.savefig('cartesian_straight_line.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Test circular arc trajectory
        print("\n--- Testing circular arc trajectory ---")

        center = T_home[:3, 3] + np.array([0.3, 0.2, -0.1])  # 10cm above home
        radius = 0.2  # 20cm radius
        normal = np.array([0, 0, 1])  # Horizontal circle

        arc_trajectory = cart_planner.plan_circular_arc(
            center=center,
            start_angle=0,
            end_angle=np.pi,  # Half circle
            radius=radius,
            normal=normal,
            duration=3.0,
            seed = q_home
        )

        arc_times = arc_trajectory['time']
        arc_cart_pos = arc_trajectory['cartesian_positions']
        arc_joints_pos = arc_trajectory['joint_positions']

        print(f"Circular arc trajectory planned with {len(arc_times)} points")
        print('number of Circular arc trajectory',len(arc_joints_pos))

        # Plot circular trajectory
        plt.figure(figsize=(12, 8))

        # 3D plot
        ax = plt.subplot(1, 2, 1, projection='3d')
        ax.plot(arc_cart_pos[:, 0], arc_cart_pos[:, 1], arc_cart_pos[:, 2], 'r-', linewidth=2)
        ax.scatter(arc_cart_pos[0, 0], arc_cart_pos[0, 1], arc_cart_pos[0, 2],
                   color='green', s=100, label='Start')
        ax.scatter(arc_cart_pos[-1, 0], arc_cart_pos[-1, 1], arc_cart_pos[-1, 2],
                   color='red', s=100, label='End')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Circular Arc Trajectory')
        ax.legend()

        # 2D projection
        plt.subplot(1, 2, 2)
        plt.plot(arc_cart_pos[:, 0], arc_cart_pos[:, 1], 'r-', linewidth=2)
        plt.scatter(arc_cart_pos[0, 0], arc_cart_pos[0, 1], color='green', s=100, label='Start')
        plt.scatter(arc_cart_pos[-1, 0], arc_cart_pos[-1, 1], color='red', s=100, label='End')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('Arc Trajectory (XY Projection)')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig('cartesian_circular_arc.png', dpi=150, bbox_inches='tight')
        plt.show()

        print("Cartesian trajectory planning test completed successfully!")
        return True

    except Exception as e:
        print(f"Error in Cartesian trajectory test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trajectory_smoothing():
    """Test trajectory smoothing functionality."""
    print("\n=== Testing Trajectory Smoothing ===")

    try:
        # Initialize kinematics model
        kin_model = PinocchioKinematicsModel(
            urdf_path,
            base_link="base",
            end_effector_link="fr3_hand_tcp"
        )

        lower_limits, upper_limits = kin_model.get_joint_limits()
        n_joints = len(lower_limits)

        # Create a noisy trajectory
        planner = PolynomialTrajectoryPlanner(dt=0.01)
        waypoints = [
            np.zeros(n_joints),
            0.3 * (lower_limits + upper_limits),
            0.7 * (lower_limits + upper_limits),
            np.zeros(n_joints)
        ]

        times, positions_orig, velocities_orig, accelerations_orig = planner.plan_joint_trajectory(
            waypoints, method="cubic_spline"
        )

        # Add noise to simulate measurement errors
        noise_level = 0.01
        positions_noisy = positions_orig + np.random.normal(0, noise_level, positions_orig.shape)

        # Apply smoothing
        positions_smooth, velocities_smooth, accelerations_smooth = smooth_trajectory(
            positions_noisy, times, smoothing_factor=0.1
        )

        print(f"Original trajectory has {len(times)} points")
        print(f"Applied smoothing with factor 0.1")

        # Compare original, noisy, and smoothed trajectories
        plt.figure(figsize=(15, 10))

        joint_idx = 0  # Plot first joint

        plt.subplot(3, 1, 1)
        plt.plot(times, positions_orig[:, joint_idx], 'b-', label='Original', linewidth=2)
        plt.plot(times, positions_noisy[:, joint_idx], 'r--', label='Noisy', alpha=0.7)
        plt.plot(times, positions_smooth[:, joint_idx], 'g-', label='Smoothed', linewidth=2)
        plt.ylabel('Position (rad)')
        plt.title(f'Joint {joint_idx + 1} Trajectory Smoothing')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(times, velocities_orig[:, joint_idx], 'b-', label='Original', linewidth=2)
        plt.plot(times, velocities_smooth[:, joint_idx], 'g-', label='Smoothed', linewidth=2)
        plt.ylabel('Velocity (rad/s)')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(times, accelerations_orig[:, joint_idx], 'b-', label='Original', linewidth=2)
        plt.plot(times, accelerations_smooth[:, joint_idx], 'g-', label='Smoothed', linewidth=2)
        plt.ylabel('Acceleration (rad/s²)')
        plt.xlabel('Time (s)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('trajectory_smoothing.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Calculate smoothing metrics
        position_error = np.mean(np.abs(positions_smooth - positions_orig))
        print(f"Mean position error after smoothing: {position_error:.6f} rad")

        print("Trajectory smoothing test completed successfully!")
        return True

    except Exception as e:
        print(f"Error in trajectory smoothing test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trajectory_feasibility():
    """Test trajectory feasibility checking."""
    print("\n=== Testing Trajectory Feasibility Checking ===")

    try:
        # Initialize kinematics model
        kin_model = PinocchioKinematicsModel(
            urdf_path,
            base_link="base",
            end_effector_link="fr3_hand_tcp"
        )

        lower_limits, upper_limits = kin_model.get_joint_limits()
        n_joints = len(lower_limits)

        # Create trajectory planner
        planner = PolynomialTrajectoryPlanner(dt=0.01)

        # Test 1: Feasible trajectory
        print("\n--- Test 1: Feasible trajectory ---")
        waypoints_feasible = [
            0.2 * (lower_limits + upper_limits),
            0.4 * (lower_limits + upper_limits),
            0.6 * (lower_limits + upper_limits),
            0.8 * (lower_limits + upper_limits)
        ]

        times, positions, velocities, accelerations = planner.plan_joint_trajectory(
            waypoints_feasible, method="cubic_spline"
        )

        # Define realistic limits
        velocity_limits = np.full(n_joints, 2.0)  # 2 rad/s max
        acceleration_limits = np.full(n_joints, 5.0)  # 5 rad/s² max

        feasibility = check_trajectory_feasibility(
            positions, velocities, accelerations,
            (lower_limits, upper_limits),
            velocity_limits, acceleration_limits
        )

        print("Feasibility results:")
        print(f"  Position feasible: {feasibility['position_feasible']}")
        print(f"  Velocity feasible: {feasibility['velocity_feasible']}")
        print(f"  Acceleration feasible: {feasibility['acceleration_feasible']}")
        if feasibility['violations']:
            print(f"  Violations: {feasibility['violations']}")

        # Test 2: Infeasible trajectory (positions outside limits)
        print("\n--- Test 2: Infeasible trajectory (position limits) ---")
        waypoints_infeasible = [
            lower_limits - 0.1,  # Below lower limit
            upper_limits + 0.1,  # Above upper limit
        ]

        times, positions, velocities, accelerations = planner.plan_joint_trajectory(
            waypoints_infeasible, method="cubic_spline"
        )

        feasibility = check_trajectory_feasibility(
            positions, velocities, accelerations,
            (lower_limits, upper_limits),
            velocity_limits, acceleration_limits
        )

        print("Feasibility results:")
        print(f"  Position feasible: {feasibility['position_feasible']}")
        print(f"  Velocity feasible: {feasibility['velocity_feasible']}")
        print(f"  Acceleration feasible: {feasibility['acceleration_feasible']}")
        if feasibility['violations']:
            print(f"  Violations: {feasibility['violations']}")

        # Test 3: High-speed trajectory (velocity limits exceeded)
        print("\n--- Test 3: High-speed trajectory (velocity limits) ---")
        waypoints_fast = [
            np.zeros(n_joints),
            0.9 * (lower_limits + upper_limits)
        ]

        # Very short time to force high velocities
        times_fast = [0.0, 0.1]

        times, positions, velocities, accelerations = planner.plan_joint_trajectory(
            waypoints_fast, times=times_fast, method="cubic_spline"
        )

        feasibility = check_trajectory_feasibility(
            positions, velocities, accelerations,
            (lower_limits, upper_limits),
            velocity_limits, acceleration_limits
        )

        print("Feasibility results:")
        print(f"  Position feasible: {feasibility['position_feasible']}")
        print(f"  Velocity feasible: {feasibility['velocity_feasible']}")
        print(f"  Acceleration feasible: {feasibility['acceleration_feasible']}")
        if feasibility['violations']:
            print(f"  Violations: {feasibility['violations']}")

        print(f"Max velocity in trajectory: {np.max(np.abs(velocities)):.2f} rad/s")
        print(f"Velocity limit: {velocity_limits[0]} rad/s")

        print("Trajectory feasibility checking test completed successfully!")
        return True

    except Exception as e:
        print(f"Error in trajectory feasibility test: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_complete_workflow():
    """Demonstrate a complete trajectory planning workflow."""
    print("\n=== Complete Trajectory Planning Workflow Demo ===")

    try:
        # Initialize kinematics model
        kin_model = PinocchioKinematicsModel(
            urdf_path,
            base_link="base",
            end_effector_link="fr3_hand_tcp"
        )

        # Get joint limits
        lower_limits, upper_limits = kin_model.get_joint_limits()
        n_joints = len(lower_limits)

        print(f"Robot initialized with {n_joints} joints")

        # Step 1: Define task - pick and place operation
        print("\n--- Step 1: Define Pick and Place Task ---")

        # Home position
        q_home = 0.5 * (lower_limits + upper_limits)
        T_home = kin_model.forward_kinematics(q_home)

        # Pick position (offset from home)
        T_pick = T_home.copy()
        T_pick[:3, 3] += np.array([0.1, 0.1, -0.05])

        # Place position
        T_place = T_home.copy()
        T_place[:3, 3] += np.array([-0.1, 0.1, 0.05])

        print(f"Home position: {T_home[:3, 3]}")
        print(f"Pick position: {T_pick[:3, 3]}")
        print(f"Place position: {T_place[:3, 3]}")

        # Step 2: Plan Cartesian trajectories
        print("\n--- Step 2: Plan Cartesian Trajectories ---")

        cart_planner = CartesianTrajectoryPlanner(kin_model, dt=0.01)

        # Home to pick
        traj1 = cart_planner.plan_straight_line(T_home, T_pick, duration=2.0, seed_joint_config=q_home)

        # Pick to place (with intermediate waypoint)
        T_intermediate = T_pick.copy()
        T_intermediate[:3, 3] += np.array([0, 0, 0.1])  # Lift up

        traj2 = cart_planner.plan_straight_line(T_pick, T_intermediate, duration=1.0)
        traj3 = cart_planner.plan_straight_line(T_intermediate, T_place, duration=2.0)

        # Place back to home
        traj4 = cart_planner.plan_straight_line(T_place, T_home, duration=2.0)

        print("Planned 4 trajectory segments")

        # Step 3: Combine trajectories
        print("\n--- Step 3: Combine Trajectories ---")

        # Combine all trajectory segments
        all_times = []
        all_joint_pos = []
        all_cart_pos = []

        current_time = 0

        for i, traj in enumerate([traj1, traj2, traj3, traj4]):
            times = traj['time'] + current_time
            all_times.extend(times)
            all_joint_pos.extend(traj['joint_positions'])
            all_cart_pos.extend(traj['cartesian_positions'])
            current_time = times[-1]
            print(f"  Segment {i + 1}: {len(times)} points, duration {times[-1] - times[0]:.2f}s")

        all_times = np.array(all_times)
        all_joint_pos = np.array(all_joint_pos)
        all_cart_pos = np.array(all_cart_pos)

        print(f"Combined trajectory: {len(all_times)} points, total duration {all_times[-1]:.2f}s")

        # Step 4: Check feasibility
        print("\n--- Step 4: Check Feasibility ---")

        # Compute velocities and accelerations
        dt = all_times[1] - all_times[0]
        joint_velocities = np.gradient(all_joint_pos, dt, axis=0)
        joint_accelerations = np.gradient(joint_velocities, dt, axis=0)

        # Check feasibility
        velocity_limits = np.full(n_joints, 2.0)
        acceleration_limits = np.full(n_joints, 5.0)

        feasibility = check_trajectory_feasibility(
            all_joint_pos, joint_velocities, joint_accelerations,
            (lower_limits, upper_limits),
            velocity_limits, acceleration_limits
        )

        print("Overall feasibility:")
        print(f"  Position feasible: {feasibility['position_feasible']}")
        print(f"  Velocity feasible: {feasibility['velocity_feasible']}")
        print(f"  Acceleration feasible: {feasibility['acceleration_feasible']}")

        if feasibility['violations']:
            print(f"  Violations: {feasibility['violations']}")

        # Step 5: Apply smoothing if needed
        if not all([feasibility['position_feasible'], feasibility['velocity_feasible'],
                    feasibility['acceleration_feasible']]):
            print("\n--- Step 5: Apply Smoothing ---")

            smoothed_pos, smoothed_vel, smoothed_acc = smooth_trajectory(
                all_joint_pos, all_times, smoothing_factor=0.2
            )

            # Recheck feasibility
            feasibility_smooth = check_trajectory_feasibility(
                smoothed_pos, smoothed_vel, smoothed_acc,
                (lower_limits, upper_limits),
                velocity_limits, acceleration_limits
            )

            print("Feasibility after smoothing:")
            print(f"  Position feasible: {feasibility_smooth['position_feasible']}")
            print(f"  Velocity feasible: {feasibility_smooth['velocity_feasible']}")
            print(f"  Acceleration feasible: {feasibility_smooth['acceleration_feasible']}")

            all_joint_pos = smoothed_pos
            joint_velocities = smoothed_vel
            joint_accelerations = smoothed_acc

        # Step 6: Visualize results
        print("\n--- Step 6: Visualize Results ---")

        plt.figure(figsize=(18, 12))

        # 3D Cartesian trajectory
        ax1 = plt.subplot(2, 3, 1, projection='3d')
        ax1.plot(all_cart_pos[:, 0], all_cart_pos[:, 1], all_cart_pos[:, 2], 'b-', linewidth=2)

        # Mark key points
        ax1.scatter(*T_home[:3, 3], color='green', s=100, label='Home')
        ax1.scatter(*T_pick[:3, 3], color='red', s=100, label='Pick')
        ax1.scatter(*T_place[:3, 3], color='orange', s=100, label='Place')

        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory')
        ax1.legend()

        # Joint positions
        plt.subplot(2, 3, 2)
        for i in range(min(3, n_joints)):  # Plot first 3 joints
            plt.plot(all_times, all_joint_pos[:, i], label=f'Joint {i + 1}')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (rad)')
        plt.title('Joint Positions')
        plt.legend()
        plt.grid(True)

        # Joint velocities
        plt.subplot(2, 3, 3)
        for i in range(min(3, n_joints)):
            plt.plot(all_times, joint_velocities[:, i], label=f'Joint {i + 1}')
        plt.axhline(y=velocity_limits[0], color='r', linestyle='--', label='Limit')
        plt.axhline(y=-velocity_limits[0], color='r', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (rad/s)')
        plt.title('Joint Velocities')
        plt.legend()
        plt.grid(True)

        # Joint accelerations
        plt.subplot(2, 3, 4)
        for i in range(min(3, n_joints)):
            plt.plot(all_times, joint_accelerations[:, i], label=f'Joint {i + 1}')
        plt.axhline(y=acceleration_limits[0], color='r', linestyle='--', label='Limit')
        plt.axhline(y=-acceleration_limits[0], color='r', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (rad/s²)')
        plt.title('Joint Accelerations')
        plt.legend()
        plt.grid(True)

        # Cartesian velocity
        cart_velocities = np.gradient(all_cart_pos, dt, axis=0)
        cart_speeds = np.linalg.norm(cart_velocities, axis=1)

        plt.subplot(2, 3, 5)
        plt.plot(all_times, cart_speeds, 'b-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (m/s)')
        plt.title('End-Effector Speed')
        plt.grid(True)

        # XY trajectory projection
        plt.subplot(2, 3, 6)
        plt.plot(all_cart_pos[:, 0], all_cart_pos[:, 1], 'b-', linewidth=2)
        plt.scatter(T_home[0, 3], T_home[1, 3], color='green', s=100, label='Home')
        plt.scatter(T_pick[0, 3], T_pick[1, 3], color='red', s=100, label='Pick')
        plt.scatter(T_place[0, 3], T_place[1, 3], color='orange', s=100, label='Place')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('XY Trajectory Projection')
        plt.axis('equal')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('complete_workflow.png', dpi=150, bbox_inches='tight')
        plt.show()

        print("\nComplete trajectory planning workflow demo finished successfully!")

        # Summary
        print(f"\n=== Summary ===")
        print(f"Total trajectory duration: {all_times[-1]:.2f} seconds")
        print(f"Number of trajectory points: {len(all_times)}")
        print(f"Maximum joint velocity: {np.max(np.abs(joint_velocities)):.2f} rad/s")
        print(f"Maximum joint acceleration: {np.max(np.abs(joint_accelerations)):.2f} rad/s²")
        print(f"Maximum end-effector speed: {np.max(cart_speeds):.3f} m/s")

        return True

    except Exception as e:
        print(f"Error in complete workflow demo: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all trajectory planning tests."""
    print("=== Running All Trajectory Planning Tests ===")

    tests = [
        # ("Joint Space Trajectory", test_joint_space_trajectory),
        ("Cartesian Trajectory", test_cartesian_trajectory),
        # ("Trajectory Smoothing", test_trajectory_smoothing),
        # ("Trajectory Feasibility", test_trajectory_feasibility),
        # ("Complete Workflow Demo", demo_complete_workflow)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"Running: {test_name}")
        print(f"{'=' * 60}")

        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))

    # Print summary
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{test_name:.<50} {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed successfully!")
    else:
        print(f"⚠️  {total - passed} test(s) failed")

    return passed == total


if __name__ == "__main__":
    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        print("Matplotlib available - plots will be generated")
    except ImportError:
        print("Warning: Matplotlib not available - no plots will be generated")


        # Mock matplotlib for testing without plots
        class MockPlt:
            def figure(self, *args, **kwargs): pass

            def subplot(self, *args, **kwargs): return self

            def plot(self, *args, **kwargs): pass

            def scatter(self, *args, **kwargs): pass

            def xlabel(self, *args, **kwargs): pass

            def ylabel(self, *args, **kwargs): pass

            def zlabel(self, *args, **kwargs): pass

            def title(self, *args, **kwargs): pass

            def legend(self, *args, **kwargs): pass

            def grid(self, *args, **kwargs): pass

            def tight_layout(self, *args, **kwargs): pass

            def savefig(self, *args, **kwargs): pass

            def show(self, *args, **kwargs): pass

            def axhline(self, *args, **kwargs): pass

            def axis(self, *args, **kwargs): pass

            def set_xlabel(self, *args, **kwargs): pass

            def set_ylabel(self, *args, **kwargs): pass

            def set_zlabel(self, *args, **kwargs): pass

            def set_title(self, *args, **kwargs): pass


        plt = MockPlt()

    # Run all tests
    success = run_all_tests()

    if success:
        print("\n🚀 Trajectory planning module is ready to use!")
        print("\nExample usage:")
        print("""
# Basic usage example:
from kinematics import PinocchioKinematicsModel
from trajectory_planning import PolynomialTrajectoryPlanner, CartesianTrajectoryPlanner

# Initialize kinematics model
kin_model = PinocchioKinematicsModel(urdf_path, base_link="base", end_effector_link="fr3_hand_tcp")

# Plan joint space trajectory
planner = PolynomialTrajectoryPlanner(dt=0.01)
waypoints = [q1, q2, q3]  # Joint configurations
times, positions, velocities, accelerations = planner.plan_joint_trajectory(waypoints)

# Plan Cartesian trajectory
cart_planner = CartesianTrajectoryPlanner(kin_model)
trajectory = cart_planner.plan_straight_line(start_pose, end_pose, duration=5.0)
        """)
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")