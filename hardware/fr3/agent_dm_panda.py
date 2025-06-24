from typing import Text, Mapping, Any

from hardware.fr3.arm_dm_panda import Arm
# from hardware.fr3.gripper_panda_py import Gripper
from hardware.base.robot import Robot
import glog as log
import dm_env
import numpy as np
from dm_env import specs
from absl import logging

# dm_robotics imports
from dm_robotics.panda import arm_constants, environment, run_loop, utils
from dm_robotics.panda import parameters as params

class Agent(Robot):
    def __init__(self,  config: Mapping[Text, Any],
        env: environment.PandaEnvironment,
        ):
        self._arm = Arm(config=config['arm'], env=env)

    def print_state(self):
        self._arm.print_state()

    def get_arm(self):
        return self._arm