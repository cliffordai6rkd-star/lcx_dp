from __future__ import annotations

from motion.pin_model import RobotModel
from controller.controller_base import ControllerBase
from hardware.base.utils import RobotJointState, convert_7D_2_homo, convert_quat_to_rot_matrix, convert_homo_2_7D_pose
from controller.g1_controller_ref import G1_29_ArmIK
import glog as log
import numpy as np
import time
import os

class G1Controller(ControllerBase):
    def __init__(self, config, robot_model: RobotModel):
        super().__init__(config, robot_model)
        self._urdf = config["urdf"]
        self._use_tau = config.get("use_tau", False)
        cur_path = os.path.dirname(os.path.abspath(__file__))
        self._urdf = os.path.join(cur_path, '..', self._urdf)
        self._ik = G1_29_ArmIK(self._urdf, self._use_tau)

    def compute_controller(self, target, robot_state: RobotJointState | None = None):
        left_ee_tgt = None; right_ee_tgt = None
        for i in range(len(target)):
            target_dict = target[i]
            frame_name = list(target_dict.keys())[0]
            value = target_dict[frame_name]
            homo = convert_7D_2_homo(value)
            if 'left' in frame_name: left_ee_tgt = homo
            elif 'right' in frame_name: right_ee_tgt = homo
        joint_target = self._ik.solve_ik(left_ee_tgt, right_ee_tgt, robot_state._positions)
        return True, joint_target, ["position"]*len(target)
