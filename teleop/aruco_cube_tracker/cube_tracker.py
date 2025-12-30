from teleop.base.teleoperation_base import TeleoperationDeviceBase
from teleop.aruco_cube_tracker.cube_tracking import (
    CubeTracker,
    left_face_corners_3d,
    right_face_corners_3d,
)
from hardware.sensors.cameras.realsense_camera import RealsenseCamera
from hardware.base.utils import (
    dynamic_load_yaml,
    convert_homo_2_7D_pose,
    negate_pose,
    transform_pose,
    transform_quat,
    pose_diff,
)
import os, copy, json, time, cv2
import numpy as np
from typing import Dict, Tuple, Optional
import glog as log
from pynput import keyboard

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
    """ArUco cube tracker teleoperation interface.

    This mirrors the high-level behavior of PikaTracker so that it can be
    dropped into the same teleop pipeline. It reads RGB(D) frames from a
    RealSense, estimates 6D pose(s) of one or two tagged cubes, and exposes
    absolute / absolute_delta pose targets with a keyboard-driven init sync.
    """

    # Fixed transforms and init targets follow PikaTracker conventions
    T_ROBOT_TRACKER = np.eye(4)
    # T_TRACKER_ROBOT = np.eye(4)
    T_TRACKER_ROBOT = np.array(
        [
            [0, -1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]
    )

    # init pose for absolute delta pose calculation, left, right, head
    INIT_TARGET_POSE = [[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1]]
    WORLD_POSE = [None, None, None]
    def __init__(self, config):
        # cube & marker related config
        all_cube_marker_ids: dict = config["all_cube_marker_ids"]
        cube_size = config.get("cube_size", 8)
        marker_size = config.get("marker_size", 8)
        transformation = config.get("transformation", [0, 0, 0.10, 0, 0, 0])
        face_corners_3d = {"left": left_face_corners_3d, "right": right_face_corners_3d}
        
        # realsense_cameras related config
        cur_path = os.path.dirname(os.path.abspath(__file__))
        rs_calibration_file = os.path.join(cur_path, "../..", config["camera_calibration_path"])
        if not os.path.exists(rs_calibration_file):
            raise ValueError(
                f"Could not find the camera calibration file at {rs_calibration_file}"
            )
        with open(rs_calibration_file, "r", encoding="utf-8") as file:
            data = json.load(file)
        assert all([key in data for key in ["camera_matrix", "dist_coeffs"]])
        camera_matrix = np.array(data["camera_matrix"], dtype=np.float32)
        distorsion = np.array(data["dist_coeffs"], dtype=np.float32)

        rs_cfg_path = os.path.join(cur_path, "../..", config["camera_config"])
        rs_cfg_all = dynamic_load_yaml(rs_cfg_path)
        # Expect YAML format: {'realsense_camera': {...}}
        rs_config = rs_cfg_all.get("realsense_camera", rs_cfg_all)
        self._camera = RealsenseCamera(rs_config)
        self._img_visualize = config.get("image_visualization", True)
        if self._img_visualize:
            self._img_visual_window = 'CubeTracker Overlay'
            cv2.namedWindow(self._img_visual_window, cv2.WINDOW_NORMAL)
        
        self._cube_trackers: Dict[str, CubeTracker] = {}
        for key, marker_ids in all_cube_marker_ids.items():
            # One CubeTracker per hand side
            self._cube_trackers[key] = CubeTracker(
                camera_matrix,
                distorsion,
                cube_size,
                marker_size,
                marker_ids,
                face_corners_3d[key],
                transformation,
            )
        
        self._output_left = config.get("output_left", True)
        self._output_right = config.get("output_right", True)
        if self._output_left and self._output_right:
            self._index = {"left": 0, "right": 1}
        elif not self._output_left: 
            self._index = {"single": 1}
        elif not self._output_right:
            self._index = {"single": 0}

        # Position scale for absolute_delta
        self._position_scale = config.get("position_scale", 1.0)

        # Initialize offset pose for relative positioning
        self._device_enabled = False
        self._reset_pose_counter = 0
        # Keyboard and transform caches
        self._keyboard_listener = None
        self._key_pressed = False
        self._tracker_robot_trans = convert_homo_2_7D_pose(self.T_TRACKER_ROBOT)
        self._robot_tracker_trans = negate_pose(self._tracker_robot_trans)

        super().__init__(config)
        
    def initialize(self):
        if self._is_initialized:
            return True
        # Camera is initialized in its constructor, so we only set up key listener
        self._keyboard_listener = keyboard.Listener(on_press=self._on_key_press)
        self._keyboard_listener.start()
        log.info("CubePoseTracker keyboard listener started")

        # Small delay to allow camera stream to warm up
        time.sleep(0.5)
        log.info("CubePoseTracker initialized successfully")
        return True
    
    def _update_init_pose(self, pose):
        if pose is None:
            return False
        
        for i, (key, cur_pose) in enumerate(pose.items()):
            cur_pose = self._process_raw_pose(cur_pose, key)
            if cur_pose is None:
                return False
            self.INIT_TARGET_POSE[self._index[key]] = cur_pose
        return True
    
    def _press_i(self):
        pose, _ = self.read_data(draw=False)
        if pose is None:
            log.warn("Failed to read pose data for update init pose!!!")
            return
        if not self._update_init_pose(pose):
            log.warn("Failed to for update init pose!!!")
            return 
        self._device_enabled = True
        self._key_pressed = True
        log.info("init pose is updated!!!")

    def _press_u(self):
        self._device_enabled = False
        log.info("CubePoseTracker disabled!!!")

    def _on_key_press(self, key):
        try:
            if key.char == "i" and not self._device_enabled:
                self._press_i()
                log.info("CubePoseTracker enabled!!!")
            elif key.char == "u" and self._device_enabled:
                self._press_u()
                log.info("CubePoseTracker disabled!!!")
        except AttributeError:
            # Ignore non-character hotkeys
            pass
    
    def _process_raw_pose(self, pose, key):
        # world coordinate alignment
        if self.WORLD_POSE[self._index[key]] is None:
            self.WORLD_POSE[self._index[key]] = pose
            log.info(f'World pose updated for {key}')                
            return None
        
        # Apply the same basis-change and offsets 
        res_pose = pose_diff(pose, self.WORLD_POSE[self._index[key]])
        # pre-process to correct the coordinate
        tracker_robot_trans = self._tracker_robot_trans
        robot_tracker_trans = self._robot_tracker_trans
        res_pose = transform_pose(transform_pose(robot_tracker_trans, res_pose), tracker_robot_trans)
        
        return res_pose
    
    def parse_data_2_robot_target(self, mode: str) -> Tuple[bool, Optional[Dict], Optional[Dict]]:
        # Support absolute and absolute_delta similar to PikaTracker
        if "absolute" not in mode:
            log.warn("The cube tracker only supports absolute pose related teleoperation")
            raise ValueError(f"{mode} mode is not supported for cube tracker")

        if not self._is_initialized:
            log.warn("Device not initialized")
            return False, None, None

        pose_quat, tool_data = self.read_data()
        if pose_quat is None:
            return -1, None, None

        pose_target: Dict[str, np.ndarray] = {}
        tool_target: Dict[str, np.ndarray] = {}
        for key, value in pose_quat.items():
            cur_pose = self._process_raw_pose(value, key)
            if cur_pose is None: continue
            if mode == "absolute_delta" and self._device_enabled:
                cur_pose = self._get_diff_trans(cur_pose, self._index[key])
                cur_pose[:3] *= self._position_scale
            pose_target[key] = cur_pose
            # Provide a tool vector compatible with teleop pipeline: [position_like, command_like, key_pressed]
            tool_target[key] = np.hstack((tool_data[key], copy.deepcopy(self._key_pressed)))
        
        if any([pose is None for key, pose in pose_target.items()]) or len(pose_target) == 0:
            return -1, None, None
        
        if self._key_pressed:
            # Keep the pressed flag for a few cycles so robot can catch it
            if self._reset_pose_counter > 15:
                self._key_pressed = False
                self._reset_pose_counter = 0
            else:
                self._reset_pose_counter += 1

        if mode == "absolute" or (mode == "absolute_delta" and self._device_enabled):
            return True, pose_target, tool_target
        else:
            return False, None, None

    def read_data(self, draw=True):
        """Capture frames and estimate cube pose(s).

        Returns:
            pose_quat: dict mapping 'left'/'right' (or 'single') to 7D pose
            tool_data: dict with 2D array [normalized_dummy, command_dummy]
        """
        res = self._camera.capture_all_data()
        img = res["image"]
        depth_raw = res.get("depth_map", None)
        assert img is not None
        assert depth_raw is not None

        depth_m = depth_raw.astype(np.float32) * getattr(self._camera, "g_depth_scale", 0.001)

        pose_quat: Dict[str, np.ndarray] = {}
        tool_data: Dict[str, np.ndarray] = {}
        
        if self._img_visualize and draw:
            overlay_img = img.copy()
            cv2.waitKey(1)
        # Read poses according to configuration
        needed = {"left": self._output_left, "right": self._output_right}
        for side, need in needed.items():
            if not need or side not in self._cube_trackers:
                continue
            cur_pose = self._cube_trackers[side].get_pose(img, depth=depth_m)
            if cur_pose is None:
                # If any required side is missing, abort this cycle
                if self._img_visualize and draw:
                    cv2.imshow(self._img_visual_window, overlay_img)
                return None, None
            pose_quat[side] = cur_pose
            if self._img_visualize and draw:
                overlay_img = self._cube_trackers[side].overlay_cube_pose(overlay_img, {side: cur_pose})
            # Dummy tool channel to align with teleop pipeline (position + command)
            tool_data[side] = np.array([0.0, 0.0])
        
        if self._img_visualize and draw:
            cv2.imshow(self._img_visual_window, overlay_img)
        
        # @TODO: check how to know tool data
        if not self._output_left and self._output_right:
            pose_quat = dict(single=pose_quat.get("right"))
            tool_data = dict(single=tool_data.get("right", np.array([0.0, 0.0])))
        elif not self._output_right and self._output_left:
            pose_quat = dict(single=pose_quat.get("left"))
            tool_data = dict(single=tool_data.get("left", np.array([0.0, 0.0])))

        return pose_quat, tool_data
    
    def _get_diff_trans(self, cur_pose, index):
        diff_pose = pose_diff(cur_pose, self.INIT_TARGET_POSE[index])
        return diff_pose

    def print_data(self):
        succ, pose, tool = self.parse_data_2_robot_target("absolute")
        if succ:
            log.info(f"pose: {pose}")
        else:
            log.info("No pose available")

    def close(self):
        if not self._is_initialized:
            return
        self._device_enabled = False
        if hasattr(self, "_camera") and self._camera:
            self._camera.close()
        if self._keyboard_listener:
            self._keyboard_listener.stop()
        if self._img_visualize:
            cv2.destroyAllWindows()
        log.info("CubePoseTracker disconnected successfully")
        return True
    
