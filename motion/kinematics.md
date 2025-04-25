
## Requirements
```bash
# conda install -c conda-forge pinocchio
pip install numpy
pip install pin
pip install mujoco
pip install mink
conda install conda-forge::python-orocos-kdl
pip3 install roboticstoolbox-python # note :rtb was compiled with Numpy 1.x

```

## usage:
- INITIALIZE:
```PYTHON
import os
from pathlib import Path
from kinematics import *
# 获取当前脚本所在的目录
current_dir = Path(__file__).parent.absolute()
# 获取上一级目录
parent_dir = current_dir.parent
# 构建 URDF 文件的相对路径（assets 在上一级目录）
urdf_path = os.path.join(parent_dir, "assets", "franka_fr3", "fr3_franka_hand.urdf")
# 构建pinocchio模型
kin_model = PinocchioKinematicsModel(urdf_path,base_link="base",end_effector_link="fr3_hand_tcp") 
```

- FK:
```PYTHON
T_test = kin_model.forward_kinematics(q_test)
"""
q_test : joint_positions: Joint angles array of shape (n,) where n is the number of joints
T_test : homogeneous transformation matrix of np.array shape (4,4) 
"""
```
- IK:
```python
q_solved = kin_model.inverse_kinematics(target_pose, seed=q_seed)
# you can choose present joint angles as the seed of the joint angles
"""
target_pose : homogeneous transformation matrix of np.array shape (4,4)
q_solved : joint_positions: Joint angles array of shape (n,) where n is the number of joints
"""

```