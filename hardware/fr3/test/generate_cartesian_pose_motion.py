
from franka_bindings import (
    Robot, 
    ControllerMode, 
    JointPositions,
)

def set_default_behavior(robot):
    robot.set_collision_behavior(
        [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0], [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0], [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        [20.0, 20.0, 20.0, 20.0, 20.0, 20.0], [20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0], [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    )
    robot.set_joint_impedance([3000, 3000, 3000, 2500, 2500, 2000, 2000])
    robot.set_cartesian_impedance([3000, 3000, 3000, 300, 300, 300])

def main():
    # Connect to robot
    robot = Robot("192.168.1.101")
    set_default_behavior(robot=robot)
    

