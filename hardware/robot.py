import sys
sys.path.append('.')

import logging, abc
import numpy as np
from typing import Text, Mapping, Any, Callable, Sequence, Union
import time

from data_types import robot_data
from data_types import se3

from hardware.arm import Arm
from hardware.gripper import Gripper
# from hardware import camera as camera_lib
# from hardware import ft_sensor as ft_sensor_lib

class Robot(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    def print_state(self):
        pass

    def get_latency(self) -> float:
        """
        Returns the latency of the robot's communication system in seconds.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_transmission_rate(self) -> float:
        """
        Returns the transmission rate of the robot's communication system in Mbps.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_frame_loss_rate(self) -> float:
        """
        Returns the frame loss rate of the robot's communication system as a percentage.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_repeatability(self) -> float:
        """
        Returns the repeatability of the robot in millimeters.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_tracking_accuracy(self) -> float:
        """
        Returns the tracking accuracy of the robot in millimeters.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_hand_eye_calibration_accuracy(self) -> float:
        """
        Returns the hand-eye calibration accuracy of the robot in millimeters.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

class FR3(Robot):
    def __init__(self, config: Mapping[Text, Any], urdf: Text, arm: Arm, gripper: Gripper):
        self._arm = arm
        self._gripper = gripper
        self._state = robot_data.RobotState()
    def print_state(self):
        print('empty ...')
        pass


class Monte03(Robot):
    def __init__(self, config: Mapping[Text, Any], urdf: Text, 
                 arm_l: Arm, gripper_l: Gripper,
                 arm_r: Arm, gripper_r: Gripper):
        pass

class UnitreeG1(Robot):
    def __init__(self):
        pass

class AgibotG1(Robot):
    def __init__(self):
        pass