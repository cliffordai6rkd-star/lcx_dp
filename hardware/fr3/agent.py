import numpy as np
from typing import Text, Mapping, Any, Callable, Sequence, Union

from data_types import robot_data
from data_types import se3

from hardware.fr3.arm import Arm
from hardware.fr3.gripper import Gripper
from hardware.base.robot import Robot
from panda_py import libfranka, Desk
import glog as log

class Agent(Robot):
    def __init__(self,  config: Mapping[Text, Any], urdf: Text, 
                 ip:str='192.168.1.101',
                 username = 'franka', 
                 password = 'franka123'):
        desk = Desk(ip, username, password)

        if desk.is_locked():
            desk.unlock()
        if not desk.has_control():
            desk.take_control()
            desk.activate_fci()
        
        self._arm = Arm(ip)
        self._gripper = Gripper(ip)

    def print_state(self):
        log.info(f"Has realtime kernel: {libfranka.has_realtime_kernel()}")
        self._arm.get_state()
        self._gripper.get_state()
        # log.info(f"==={self._arm.get_spatial_mass_matrix()}")

    def move_to_start(self):
        self._arm.move_to_start()

    def get_pose(self):
        return self._arm.get_ee_pose()
    
    def move_to_joint_position(self, q):
        self._arm.move_to_joint_position(q)

    def grasp(self):
        self._gripper.grasp(width=0, speed=0.2, force=10, epsilon_inner=0.04, epsilon_outer=0.04)
    
