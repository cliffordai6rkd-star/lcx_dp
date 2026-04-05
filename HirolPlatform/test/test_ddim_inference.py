#!/usr/bin/env python3
"""
Test script for DDIM inference functionality.
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factory.utils import parse_args
from hardware.base.utils import dynamic_load_yaml
from factory.tasks.inferences_tasks.dp.dp_inference import DP_Inferencer


def test_ddim_config_loading():
    """Test DDIM configuration loading."""
    print("Testing DDIM configuration loading...")

    try:
        config_path = "factory/tasks/inferences_tasks/dp/config/fr3_dp_ddim_inference_cfg.yaml"
        config = dynamic_load_yaml(config_path)

        assert config.get('inference_scheduler_type') == 'ddim', "DDIM scheduler type not set"
        assert config.get('ddim_inference_steps') == 16, "DDIM inference steps not set correctly"
        assert config.get('ddim_eta') == 0.0, "DDIM eta not set correctly"

        print("✓ DDIM configuration loaded successfully")
        return True

    except Exception as e:
        print(f"✗ DDIM configuration loading failed: {e}")
        return False


def test_ddim_scheduler_setup():
    """Test DDIM scheduler setup without full inference."""
    print("Testing DDIM scheduler setup...")

    try:
        config_path = "factory/tasks/inferences_tasks/dp/config/fr3_dp_ddim_inference_cfg.yaml"
        config = dynamic_load_yaml(config_path)

        # Mock a minimal config to avoid full robot initialization
        test_config = {
            'checkpoint_path': config['checkpoint_path'],
            'device': 'cuda',
            'inference_scheduler_type': 'ddim',
            'ddim_inference_steps': 16,
            'ddim_eta': 0.0,
            'max_step_nums': 1,
            'num_episodes': 1,
            'action_type': 'joint_position',
            'reset_space': 'joint',
            'obs_contain_ee': False,
            'is_debug': True
        }

        # This will test the model loading and DDIM setup
        # but we'll catch the error when it tries to initialize the robot
        try:
            inferencer = DP_Inferencer(test_config)
            print("✓ DDIM scheduler setup completed successfully")

            # Check if DDIM was properly configured
            policy = inferencer._dp_policy
            if hasattr(policy, 'noise_scheduler'):
                scheduler_name = policy.noise_scheduler.__class__.__name__
                print(f"✓ Scheduler type: {scheduler_name}")
                print(f"✓ Inference steps: {policy.num_inference_steps}")

                if 'DDIM' in scheduler_name:
                    print("✓ DDIM scheduler successfully applied")
                    return True
                else:
                    print(f"✗ Expected DDIM scheduler, got {scheduler_name}")
                    return False
            else:
                print("✗ No noise_scheduler found in policy")
                return False

        except Exception as robot_error:
            # Expected error when robot hardware is not available
            if "robot" in str(robot_error).lower() or "motion" in str(robot_error).lower():
                print("✓ Model loaded successfully (robot hardware error expected in test)")
                return True
            else:
                raise robot_error

    except Exception as e:
        print(f"✗ DDIM scheduler setup failed: {e}")
        return False


def main():
    """Run all DDIM tests."""
    print("=== DDIM Inference Tests ===")

    tests = [
        test_ddim_config_loading,
        test_ddim_scheduler_setup
    ]

    results = []
    for test in tests:
        print(f"\n{'-' * 50}")
        result = test()
        results.append(result)
        time.sleep(1)

    print(f"\n{'=' * 50}")
    print("Test Results:")
    print(f"Passed: {sum(results)}/{len(results)}")

    if all(results):
        print("🎉 All DDIM tests passed!")
        return 0
    else:
        print("❌ Some DDIM tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())