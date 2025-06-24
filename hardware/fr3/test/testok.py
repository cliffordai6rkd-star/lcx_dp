import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import glog as log
log.setLevel("INFO")

from hardware.fr3.agent import Agent
import numpy as np
from tools import file_utils
from franka_bindings import (
    JointPositions,
)

cur_path = os.path.dirname(os.path.abspath(__file__))
robot_config_file = os.path.join(
cur_path, '../config/agent.yaml')
config = file_utils.read_config(robot_config_file)

print(config)

r = Agent(config)

a = r.get_arm()

try:
    
    # First move the robot to a suitable joint configuration
    print("WARNING: This example will move the robot!")
    print("Please make sure to have the user stop button at hand!")
    input("Press Enter to continue...")
    
    # Start joint position control with external control loop
    # active_control = robot.start_joint_position_control(ControllerMode.JointImpedance)
    
    initial_position = [0.0] * 7
    time_elapsed = 0.0
    motion_finished = False
    
    # External control loop
    while not motion_finished:
        # Read robot state and duration
        # duration = a.get_duration()
        # robot_state = a.get_state()

        robot_state, duration = a.read()

        # Update time
        time_elapsed += duration
        
        # On first iteration, capture initial position
        if time_elapsed <= duration:
            initial_position = robot_state.q_d if hasattr(robot_state, 'q_d') else robot_state.q
        
        # Calculate delta angle using the same formula as in C++ example
        delta_angle = np.pi / 8.0 * (1 - np.cos(np.pi / 2.5 * time_elapsed))
        
        # Update joint positions
        new_positions = [
            initial_position[0],
            initial_position[1],
            initial_position[2],
            initial_position[3] + delta_angle,
            initial_position[4] + delta_angle,
            initial_position[5],
            initial_position[6] + delta_angle
        ]
        
        # Set joint positions
        joint_positions = JointPositions(new_positions)
        
        # Set motion_finished flag to True on the last update
        if time_elapsed >= 5.0:
            joint_positions.motion_finished = True
            motion_finished = True
            print("Finished motion, shutting down example")
        
        # Send command to robot
        # a.controller.writeOnce(joint_positions)
        a.set_joint_positions(new_positions)
    
except Exception as e:
    print(f"Error occurred: {e}")
finally:
    a.stop()

