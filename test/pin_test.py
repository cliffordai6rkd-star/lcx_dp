import pinocchio as pin
import numpy as np

a = np.array([[1, 0, 0, 3],
              [0, 1, 0, 3],
              [0, 0, 1, 3],
              [0, 0, 0, 1]])
b = np.array([[1, 0, 0, 2],
              [0, 1, 0, 2],
              [0, 0, 1, 2],
              [0, 0, 0, 1]])
SE3_1 = pin.SE3(a[:3, :3], a[:3, 3])
print(f'se3 1: {SE3_1}')
SE3_2 = pin.SE3(b[:3, :3], b[:3, 3])
print(f'se3 2: {SE3_2}')

T_err = SE3_2.actInv(SE3_1)
err = pin.log6(T_err).vector
print(f'err: {err}')
