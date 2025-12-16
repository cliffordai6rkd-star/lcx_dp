from teleop.base.teleoperation_base import TeleoperationDeviceBase
from teleop.aruco_cube_tracker.cube_tracking import CubeTracker, left_face_corners_3d, right_face_corners_3d
from hardware.sensors.cameras.realsense_camera import RealsenseCamera
from hardware.base.utils import dynamic_load_yaml, convert_homo_2_7D_pose, negate_pose, transform_pose, transform_quat, pose_diff
import os, copy, threading, json
import numpy as np
from typing import Dict, Tuple, Optional

# left_hand:
#     cube_size: 8
#     marker_size: 6.4
#     transformation: [0, 0, 0.10, 0, 0, 0]
#     marker_ids: [null, 46, 44, 45, 47, 43]
#     corner_faces: left

#   right_hand:
#     cube_size: 8
#     marker_size: 6.4
#     transformation: [0, 0, 0.10, 0, 0, 0]
#     marker_ids: [null, 3, 0, 4, 5, 2]
#     corner_faces: right

class CubePoseTracker(TeleoperationDeviceBase):
    # init pose for absolute delta pose calculation, left, right, head
    INIT_TARGET_POSE = [[0,0,0,0,0,0,1], [0,0,0,0,0,0,1], [0,0,0,0,0,0,1]]
    
    def __init__(self, config):
        # cube & marker related config
        all_cube_marker_ids:dict = config["all_cube_marker_ids"]
        cube_size = config.get("cube_size", 8)
        marker_size = config.get("marker_size", 8);
        transformation = config.get("transformation", [0, 0, 0.10, 0, 0, 0])
        face_corners_3d = {"left": left_face_corners_3d, "right": right_face_corners_3d}
        
        # realsense_cameras related config
        cur_path = os.path.dirname(__file__)
        rs_calibration_file = config["camera_calibration_path"]
        rs_calibration_file = os.path.join(cur_path, "../..", rs_calibration_file)
        if not os.path.exists(cur_path):
            raise ValueError(f'Could not find the camera caliration path from {rs_calibration_file}')
        with open(rs_calibration_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert all([key in data for key in ["camera_matrix", "dist_coeffs"]])
        camera_matrix = np.array(data["camera_matrix"], dtype=np.float32)
        distorsion = np.array(data["dist_coeffs"], dtype=np.float32)
        rs_config = config["camera_config"]
        rs_config = os.path.join(cur_path, "../..", rs_config)
        rs_config = dynamic_load_yaml(rs_config)
        self._camera = RealsenseCamera(rs_config)
        
        self._cube_trackers = {}
        for key, marker_ids in all_cube_marker_ids.items():
            self._cube_trackers = CubeTracker(camera_matrix, distorsion, cube_size,
                marker_size, marker_ids, face_corners_3d[key], transformation)
        
        self._output_left = config.get("output_left", True)
        self._output_right = config.get("output_right", True)
        if self._output_left and self._output_right:
            self._index = {"left": 0, "right": 1}
        elif not self._output_left: 
            self._index = {"single": 1}
        elif not self._output_right:
            self._index = {"single": 0}
      
            
        # Initialize offset pose for relative positioning
        self._init_pose = config.get('init_pose', None)
        self._last_quat = [np.array([0,0,0,1]), np.array([0,0,0,1])]
        if not self._init_pose is None:
            self._init_pose_rot = [self._init_pose["initial_pose_left"][3:], 
                                    self._init_pose["initial_pose_right"][3:]]
            self._last_quat = self._init_pose_rot
            self._init_pose_trans = [self._init_pose["initial_pose_left"][:3], 
                                    self._init_pose["initial_pose_right"][:3]]
        self._device_enabled = False
        self._reset_pose_counter = 0
        
        super().__init__(config)
        
    def initialize(self):
        if self._is_initialized:
            return True
        
        
    
    def _process_raw_pose(self, pose, key):
        res_pose = np.zeros(7)
        if key != "head":
            tracker_robot_trans = convert_homo_2_7D_pose(self.T_TRACKER_ROBOT)
        else:
            tracker_robot_trans = convert_homo_2_7D_pose(self.T_TRACKER_HEAD)
        # basis change
        robot_tracker_trans = negate_pose(tracker_robot_trans)
        res_pose = transform_pose(transform_pose(robot_tracker_trans, pose), tracker_robot_trans)
        
        if key != "head":
            # read pose init sync to x forward y left z up 
            static_rot_offset = np.array([0, 0, 1, 0])
            res_pose = self.apply_rotation_offset(res_pose, static_rot_offset)
            # apply rotation based on robot init pose
            res_pose = self.apply_init_offset(res_pose, self._index[key])
        else:
            res_pose = self.apply_rotation_offset(res_pose, np.array([0, 0, 0, 1]))
            # no need to apply for robot sync    
        return res_pose
    
    def parse_data_2_robot_target(self, mode: str) -> Tuple[bool, Optional[Dict], Optional[Dict]]:
        pass
    
    def apply_rotation_offset(self, pose, rot_offset):
        new_pose = copy.deepcopy(pose)
        # apply rotation offset
        new_pose[3:] = transform_quat(pose[3:], rot_offset)
        return new_pose
    
    def apply_init_offset(self, pose, id):
        new_pose = pose
        # basis convertion (mainly for rot)
        if self._init_pose is not None:
            init_rot = self._init_pose_rot[id]
            # log.info(f'id: {id}, init rot: {init_rot}')
            new_pose = self.apply_rotation_offset(new_pose, init_rot)
        return new_pose
    
    def _get_diff_trans(self, cur_pose, index):
        diff_pose = pose_diff(cur_pose, self.INIT_TARGET_POSE[index])
        return diff_pose
    