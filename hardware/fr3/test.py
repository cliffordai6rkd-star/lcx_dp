import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import glog as log
log.setLevel("INFO")

from hardware.fr3.agent import Agent
from panda_py import ik
config = None
urdf = None
r = Agent(config, urdf)

r.print_state()
r.grasp()

r.move_to_start()
pose = r.get_pose()
pose[2,3] -= .1
q = ik(pose)
r.move_to_joint_position(q)