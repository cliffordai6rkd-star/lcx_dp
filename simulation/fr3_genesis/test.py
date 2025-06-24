import sys, os, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))

sys.path.append("../../dependencies/libfranka-python/franka_bindings")
from franka_bindings import JointPositions

import glog as log
log.setLevel("INFO")

from hardware.fr3.agent import Agent
import numpy as np
from tools import file_utils

from data_types import se3
import time


cur_path = os.path.dirname(os.path.abspath(__file__))
robot_config_file = os.path.join(
cur_path, './config/agent.yaml')
config = file_utils.read_config(robot_config_file)

print(config)

r = Agent(config)

a = r.get_arm()

g = a.get_gripper()

# g.homing()

# g.grasp(width=0.02, speed=0.2, force=10, epsilon_inner=0.04, epsilon_outer=0.04)

# g.move(width=0.06, speed=0.2)

a.print_state()

# a.move_to_start()
# time.sleep(20)

pose = a.get_tcp_pose()

log.info(f"pose: \n{pose}")
log.info(f"IK: \n   {a.get_joint_positions()}\n== {a.ik(pose)}")

pose[2,3] -= 0.14
# # log.info(f"pose: \n{pose}")

q = a.ik(pose)
log.info(f"ik: {q}")

# # # Extend q by two values: 0.02, 0.02
# # xq = np.append(a.ik(pose), [0.00, 0.00])
# # log.info(f"fk: {a.fk(xq)}")

desired_position = q.copy()
# a.set_joint_positions(desired_position)

# time.sleep(10)
# a.print_state()


initial_position = a.get_joint_positions()

amplitude = np.pi / 8.0
frequency = 0.2  # Hz
run_time = 5.0  # seconds
elapsed_time = 0.0

while elapsed_time < run_time:

    try:
        duration = a.get_duration()
    except Exception as e:
        print(f"Error reading state: {e}")
        break

    elapsed_time += duration
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    
    jp = (initial_position + (desired_position - initial_position) * elapsed_time / run_time)
    # log.info(f"{initial_position + (desired_position - initial_position) * elapsed_time / run_time}")
    try:
        # if run_time - elapsed_time < 0.01: 
        #     jp.motion_finished = True
        a.set_joint_positions(jp)
    except Exception as e:
        print(f"Error writing joint positions: {e}")
        continue

    

# a.print_state()

# joint_positions = JointPositions(desired_position)





# joint_positions.motion_finished = False
# try:
#     a.set_joint_positions(joint_positions)
# except Exception as e:
#     print(f"Error writing joint positions: {e}")



a.stop()
print("Robot stopped.")

g.stop()
log.info("Gripper stop done")

# r.move_to_start()

# pose = r.get_pose()
# pose[2,3] -= .1
# q = ik(pose)
# r.move_to_joint_position(q)

# log.info(f"controller time: {r.get_controller_time()}")
# pose = r.get_pose()
# pose[1,3] -= .1
# r.move_to_pose(pose)
# log.info(f"controller time: {r.get_controller_time()}")


# T_0 = panda_py.fk(constants.JOINT_POSITION_START)
# T_0[1, 3] = 0.25
# T_1 = T_0.copy()
# T_1[1, 3] = -0.25

# r.move_to_pose(T_0)

# r.move_to_start()

