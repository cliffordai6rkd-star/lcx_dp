"""
@File    : test_kinematics_add_ik_my_imple
@Author  : Haotian Liang
@Time    : 2025/4/27 11:34
@Email   :Haotianliang10@gmail.com
"""
"""
@File    : test_ik_methods
@Author  : Haotian Liang
@Time    : 2025/4/27
@Email   : Haotianliang10@gmail.com
"""

import os
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Callable
from kinematics_add_ik_my_implement import PinocchioKinematicsModel

# Get the path to the URDF file
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent
urdf_path = os.path.join(parent_dir, "assets", "franka_fr3", "fr3_franka_hand.urdf")


def benchmark_ik_methods(
        ik_methods: Dict[str, Callable],
        num_trials: int = 10,
        random_seed: int = 42
) -> Dict[str, Dict[str, List[float]]]:
    """
    Benchmark different IK methods and compare their performance.

    Args:
        ik_methods: Dict mapping method names to IK solver functions
        num_trials: Number of random poses to test
        random_seed: Seed for random number generation

    Returns:
        Dictionary with benchmark results
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Create the kinematics model
    kin_model = PinocchioKinematicsModel(urdf_path, base_link="base", end_effector_link="fr3_hand_tcp")

    # Get joint limits
    lower_limits, upper_limits = kin_model.get_joint_limits()
    n_joints = len(lower_limits)

    # Initialize results dictionary
    results = {
        method_name: {
            "success_rate": [],
            "convergence_time": [],
            "position_error": [],
            "rotation_error": [],
            "iterations": []
        }
        for method_name in ik_methods.keys()
    }

    # Run the benchmark for each trial
    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}")

        # Generate a random joint configuration within limits
        q_random = np.random.uniform(lower_limits, upper_limits)

        # Compute the forward kinematics to get a target pose
        target_pose = kin_model.forward_kinematics(q_random)

        # Generate a different random seed for IK
        q_seed = np.random.uniform(lower_limits, upper_limits)

        # Test each IK method
        for method_name, ik_solver in ik_methods.items():
            print(f"Testing {method_name}...")

            # Measure time
            start_time = time.time()

            # Run the IK solver with a maximum of 1000 iterations
            try:
                q_solved = ik_solver(target_pose, seed=q_seed, max_iter=5000)
                success = True
            except Exception as e:
                print(f"Error with {method_name}: {e}")
                success = False

            end_time = time.time()
            convergence_time = end_time - start_time

            # Skip further processing if the method failed
            if not success:
                results[method_name]["success_rate"].append(0)
                results[method_name]["convergence_time"].append(convergence_time)
                results[method_name]["position_error"].append(float('inf'))
                results[method_name]["rotation_error"].append(float('inf'))
                results[method_name]["iterations"].append(0)
                continue

            # Compute the forward kinematics of the solution
            solved_pose = kin_model.forward_kinematics(q_solved)

            # Compute position and rotation errors
            position_error = np.linalg.norm(solved_pose[:3, 3] - target_pose[:3, 3])
            rotation_error = np.linalg.norm(solved_pose[:3, :3] - target_pose[:3, :3], 'fro')

            # Consider IK successful if position error is small
            ik_success = position_error < 1e-3

            # Store results
            results[method_name]["success_rate"].append(1 if ik_success else 0)
            results[method_name]["convergence_time"].append(convergence_time)
            results[method_name]["position_error"].append(position_error)
            results[method_name]["rotation_error"].append(rotation_error)
            results[method_name]["iterations"].append(0)  # Not available for all methods

            print(f"  Success: {ik_success}")
            print(f"  Time: {convergence_time:.4f} seconds")
            print(f"  Position error: {position_error:.8f}")
            print(f"  Rotation error: {rotation_error:.8f}")

    return results


def plot_benchmark_results(results: Dict[str, Dict[str, List[float]]]):
    """
    Plot the benchmark results.

    Args:
        results: Dictionary with benchmark results
    """
    method_names = list(results.keys())
    num_methods = len(method_names)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Calculate statistics
    stats = {}
    for method_name, method_results in results.items():
        stats[method_name] = {
            "success_rate": np.mean(method_results["success_rate"]) * 100,
            "avg_time": np.mean(method_results["convergence_time"]),
            "avg_pos_error": np.mean(method_results["position_error"]),
            "avg_rot_error": np.mean(method_results["rotation_error"])
        }

    # Plot success rate
    success_rates = [stats[method]["success_rate"] for method in method_names]
    axes[0].bar(method_names, success_rates)
    axes[0].set_title("Success Rate (%)")
    axes[0].set_ylim(0, 105)
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # Plot convergence time
    avg_times = [stats[method]["avg_time"] for method in method_names]
    axes[1].bar(method_names, avg_times)
    axes[1].set_title("Average Convergence Time (s)")
    axes[1].grid(True, linestyle='--', alpha=0.7)

    # Plot position error
    avg_pos_errors = [stats[method]["avg_pos_error"] for method in method_names]
    axes[2].bar(method_names, avg_pos_errors)
    axes[2].set_title("Average Position Error")
    axes[2].set_yscale('log')
    axes[2].grid(True, linestyle='--', alpha=0.7)

    # Plot rotation error
    avg_rot_errors = [stats[method]["avg_rot_error"] for method in method_names]
    axes[3].bar(method_names, avg_rot_errors)
    axes[3].set_title("Average Rotation Error")
    axes[3].set_yscale('log')
    axes[3].grid(True, linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig("ik_benchmark_results.png")
    plt.show()


def test_single_pose():
    """
    Test all IK methods on a single pose and compare the results.
    """
    # Create the kinematics model
    kin_model = PinocchioKinematicsModel(urdf_path, base_link="base", end_effector_link="fr3_hand_tcp")

    # Get joint limits
    lower_limits, upper_limits = kin_model.get_joint_limits()
    n_joints = len(lower_limits)

    print(f"Robot has {n_joints} degrees of freedom.")

    # Generate a random joint configuration within limits
    np.random.seed(30)  # For reproducibility
    q_random = np.random.uniform(lower_limits, upper_limits)

    # Compute the forward kinematics to get a target pose
    target_pose = kin_model.forward_kinematics(q_random)
    print("Target end-effector pose:")
    print(target_pose)

    # Generate a different random seed for IK
    q_seed = np.random.uniform(lower_limits, upper_limits)

    # Test each IK method
    ik_methods = {
        "DLS": kin_model.inverse_kinematics,
        "LM": kin_model.inverse_kinematics_LM,
        "Gauss-Newton": kin_model.inverse_kinematics_GaussNewton,
        "L-BFGS": kin_model.inverse_kinematics_LBFGS,
        "Newton-Raphson": kin_model.inverse_kinematics_NewtonRaphson
    }


    print("\nTesting all IK methods:")

    # Table header
    header = f"{'Method':<15} | {'Time (s)':<10} | {'Position Error':<15} | {'Rotation Error':<15} | {'Joint Distance':<15}"
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)

    for method_name, ik_solver in ik_methods.items():
        # Measure time
        start_time = time.time()

        try:
            # Solve IK
            q_solved = ik_solver(target_pose, seed=q_seed)

            # Compute FK of solution
            solved_pose = kin_model.forward_kinematics(q_solved)

            # Compute errors
            position_error = np.linalg.norm(solved_pose[:3, 3] - target_pose[:3, 3])
            rotation_error = np.linalg.norm(solved_pose[:3, :3] - target_pose[:3, :3], 'fro')

            # Compute joint distance from seed
            joint_distance = np.linalg.norm(q_solved - q_seed)

            # Record time
            elapsed_time = time.time() - start_time

            # Print results
            print(
                f"{method_name:<15} | {elapsed_time:<10.6f} | {position_error:<15.8f} | {rotation_error:<15.8f} | {joint_distance:<15.8f}")

        except Exception as e:
            print(f"{method_name:<15} | Failed: {str(e)}")

    print(separator)


if __name__ == "__main__":
    # Test a single pose with all methods
    print("=== Testing Single Pose ===")
    test_single_pose()

    # Uncomment to run the benchmark (takes longer)
    # print("\n=== Running Benchmark ===")
    # ik_methods = {
    #     "DLS": kin_model.inverse_kinematics,
    #     "LM": kin_model.inverse_kinematics_LM,
    #     "Gauss-Newton": kin_model.inverse_kinematics_GaussNewton,
    #     "L-BFGS": kin_model.inverse_kinematics_LBFGS,
    #     "Newton-Raphson": kin_model.inverse_kinematics_NewtonRaphson
    # }
    # results = benchmark_ik_methods(ik_methods, num_trials=5)
    # plot_benchmark_results(results)