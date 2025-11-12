
# Online Python - IDE, Editor, Compiler, Interpreter
from scipy.spatial.transform import Rotation as R
import numpy as np

def pose_diff(pose1, pose2):
    """
        @ brief: compute the difference between two poses (pose1 - pose2)
    """
    assert pose1.shape == (7,), f"pose1 must be 7D array, got shape {pose1.shape}"
    assert pose2.shape == (7,), f"pose2 must be 7D array, got shape {pose2.shape}"
    
    res = np.zeros(7)
    posi_diff = pose1[:3] - pose2[:3]
    rot1 = R.from_quat(pose1[3:])
    rot2 = R.from_quat(pose2[3:])
    rot2_trans = rot2.inv()
    rot = rot2_trans * rot1
    res[:3] = rot2_trans.apply(posi_diff)
    res[3:] = rot.as_quat()
    return res
    

a = np.array([4,8,3, 0, 1, 0, 0])
b = np.array([4,8,3, 0, 1, 0, 0])
print(f'pose diff: {pose_diff(a, b)}')