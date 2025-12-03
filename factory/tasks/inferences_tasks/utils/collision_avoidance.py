"""
    ported from umi by Chicheng and modified by zyx
"""
import numpy as np
from scipy.spatial import transform as st
import glog as log
"""
    ee_pose is always represented by 7D pose [x,y,z,qx,qy,qz,qw]
"""

def solve_table_collision(ee_pose, gripper_width, height_threshold,
                          finger_thickness = 25.5/1000):
    keypoints = list()
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            keypoints.append((dx * gripper_width / 2, dy * finger_thickness / 2, 0))
    keypoints = np.asarray(keypoints)
    rot_mat = st.Rotation.from_quat(ee_pose[3:7]).as_matrix()
    transformed_keypoints = np.transpose(rot_mat @ np.transpose(keypoints)) + ee_pose[:3]
    delta = max(height_threshold - np.min(transformed_keypoints[:, 2]), 0)
    log.info(f'beforce collisoon z: {ee_pose[2]}')
    ee_pose[2] += delta
    log.info(f'after collisoon z: {ee_pose[2]}')
    
def prevent_table_collision(ee_pose, height_threshold):
    height = max(height_threshold, ee_pose[2])
    log.info(f'beforce collisoon z: {ee_pose[2]}')
    ee_pose[2] = height
    log.info(f'after collisoon z: {ee_pose[2]}')

# def solve_sphere_collision(ee_poses, robots_config):
#     num_robot = len(robots_config)
#     this_that_mat = np.identity(4)
#     this_that_mat[:3, 3] = np.array([0, 0.89, 0]) # TODO: very hacky now!!!!

#     for this_robot_idx in range(num_robot):
#         for that_robot_idx in range(this_robot_idx + 1, num_robot):
#             this_ee_mat = pose_to_mat(ee_poses[this_robot_idx][:6])
#             this_sphere_mat_local = np.identity(4)
#             this_sphere_mat_local[:3, 3] = np.asarray(robots_config[this_robot_idx]['sphere_center'])
#             this_sphere_mat_global = this_ee_mat @ this_sphere_mat_local
#             this_sphere_center = this_sphere_mat_global[:3, 3]

#             that_ee_mat = pose_to_mat(ee_poses[that_robot_idx][:6])
#             that_sphere_mat_local = np.identity(4)
#             that_sphere_mat_local[:3, 3] = np.asarray(robots_config[that_robot_idx]['sphere_center'])
#             that_sphere_mat_global = this_that_mat @ that_ee_mat @ that_sphere_mat_local
#             that_sphere_center = that_sphere_mat_global[:3, 3]

#             distance = np.linalg.norm(that_sphere_center - this_sphere_center)
#             threshold = robots_config[this_robot_idx]['sphere_radius'] + robots_config[that_robot_idx]['sphere_radius']
#             # print(that_sphere_center, this_sphere_center)
#             if distance < threshold:
#                 print('avoid collision between two arms')
#                 half_delta = (threshold - distance) / 2
#                 normal = (that_sphere_center - this_sphere_center) / distance
#                 this_sphere_mat_global[:3, 3] -= half_delta * normal
#                 that_sphere_mat_global[:3, 3] += half_delta * normal
                
#                 ee_poses[this_robot_idx][:6] = mat_to_pose(this_sphere_mat_global @ np.linalg.inv(this_sphere_mat_local))
#                 ee_poses[that_robot_idx][:6] = mat_to_pose(np.linalg.inv(this_that_mat) @ that_sphere_mat_global @ np.linalg.inv(that_sphere_mat_local))
                
                