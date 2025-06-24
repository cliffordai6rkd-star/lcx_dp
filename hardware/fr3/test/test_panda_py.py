import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import glog as log
log.setLevel("INFO")

from hardware.fr3.agent_panda_py import Agent
from panda_py import ik
import numpy as np
import panda_py
from panda_py import constants
from tools import file_utils

cur_path = os.path.dirname(os.path.abspath(__file__))
robot_config_file = os.path.join(
cur_path, '../config/agent.yaml')
config = file_utils.read_config(robot_config_file)

print(config)

r = Agent(config)

r.print_state()

a = r.get_arm()
g = r.get_gripper()
# g.open()

g.move(0.02, 0.2)

xx = np.array([
        0.0,
        -np.pi / 4,
        0.0,
        -3 * np.pi / 4,
        0.0,
        np.pi / 2 + np.pi / 8,
        np.pi / 4,
    ])
count = 10
while count > 0:
    count -= 1
    a.move_to_start()
    
    a.move_to_joint_target(xx)  # this can work

    pose = r.get_pose()
    pose[2,3] -= .1
    q = ik(pose)
    a.move_to_joint_target(q)

    log.info(f"controller time: {r.get_controller_time()}")
    pose = r.get_pose()
    pose[1,3] -= .1
    a.move_to_pose(pose)
    log.info(f"controller time: {r.get_controller_time()}")


    T_0 = a.fk(constants.JOINT_POSITION_START)
    T_0[1, 3] = 0.25
    T_1 = T_0.copy()
    T_1[1, 3] = -0.25

    a.move_to_pose(T_0)

    a.move_to_joint_target(xx)  # this can work

    a.move_to_start()

