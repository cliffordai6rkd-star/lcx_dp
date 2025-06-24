import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import glog as log
log.setLevel("INFO")

from hardware.fr3.agent import Agent
import numpy as np
from tools import file_utils
from franka_bindings import (
    Robot, 
    ControllerMode, 
    JointPositions,
)
import time

cur_path = os.path.dirname(os.path.abspath(__file__))
robot_config_file = os.path.join(
cur_path, '../config/agent.yaml')
config = file_utils.read_config(robot_config_file)

print(config)

r = Agent(config)

a = r.get_arm()

a.move_to_start()
