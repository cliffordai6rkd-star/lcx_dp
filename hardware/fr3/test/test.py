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
from data_types import se3

cur_path = os.path.dirname(os.path.abspath(__file__))
robot_config_file = os.path.join(
cur_path, '../config/agent.yaml')
config = file_utils.read_config(robot_config_file)

print(config)

r = Agent(config)

a = r.get_arm()

a.move_to_start()
# while True:
#     # with a.lock:
#     #     state, duration = a.read()
#     time.sleep(1)

# xx = np.array([
#     0.0,
#     -np.pi / 4,
#     0.0,
#     -3 * np.pi / 4,
#     0.0,
#     np.pi / 2 + np.pi / 8,
#     np.pi / 4,
# ])
# a.move_to_joint_target(xx)  # this can work


# g = a.get_gripper()
# # g.homing()

# # g.grasp(width=0.02, speed=0.04, force=10, epsilon_inner=0.04, epsilon_outer=0.04)

# # g.move(width=0.06, speed=0.2)
# time.sleep(0.1)

# a.start_realtime_control()
# rt_control = a.get_realtime_control()
# state = rt_control.get_current_state()
# print(f"Joint positions: {state.q}")
# #TODO: 将sim验证过的代码放到这里
# a.move_to_start()

pose = a.get_tcp_pose()

pose[2,3] += 0.14

# pose[1,3] += 0.24

# q = a.ik(pose)
# log.info(f"ik: {q}")

a.move_to_pose(se3.Transform(matrix=pose))


pose[2,3] -= 0.14
pose[2,3] -= 0.14

a.move_to_pose(se3.Transform(matrix=pose))
# a.move_to_joint_target(q)

a.move_to_start()

a.stop()

