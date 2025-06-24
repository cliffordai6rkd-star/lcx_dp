import sys, os, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))

sys.path.append("../../dependencies/libfranka-python/franka_bindings")
from franka_bindings import JointPositions

import glog as log
log.setLevel("INFO")

from hardware.fr3.agent import Agent
import numpy as np
from tools import file_utils

cur_path = os.path.dirname(os.path.abspath(__file__))
robot_config_file = os.path.join(
cur_path, './config/agent.yaml')
config = file_utils.read_config(robot_config_file)

print(config)

r = Agent(config)

a = r.get_arm()

a.move_to_start()

a.stop()
print("Robot stopped.")