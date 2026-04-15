from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import glog as log
from scipy.spatial.transform import Rotation as R
import time

from teleop.pika_tracker.vive_tracker import ViveTracker
from hardware.base.utils import dynamic_load_yaml, transform_pose, transform_quat

@dataclass
class _TrackerCalibration:
    pose_extrinsic: Optional[np.ndarray] = None  # 7D [x, y, z, qx, qy, qz, qw]
    rotation_offset: Optional[np.ndarray] = None  # 4D [qx, qy, qz, qw]
    translation_offset: Optional[np.ndarray] = None  # 3D in calibrated world frame
    use_relative: bool = False # whether position x&y calculated by relative

class CalibrateTracker:
    def __init__(self, config):
        self._config = dict(config)
        poll_interval = float(self._config.get("reading_loop_s", self._config.get("poll_interval", 0.001)))
        auto_start = bool(self._config.get("auto_start", False))
        self.vive_tracker = ViveTracker(poll_interval=poll_interval, auto_start=auto_start)
        self._tracker_timout = config.get("timeout", 5.0)
        self._z_offset = config.get("z_offset", None)

        self._calibration_by_uid: Dict[str, _TrackerCalibration] = {}
        self._tracker_cfg = config["trackers"]
        self._use_relative = False
        self._relative_anchor = None
        
        self._is_initialized = False
        self._is_initialized = self.initialize()
        if not self._is_initialized:
            raise ValueError(f'Calibration tracker init failed')

    def initialize(self) -> bool:
        if self._is_initialized: return True
        
        self.vive_tracker.start()
        # keyboard listening for update init pose
        self._key_pressed = False
        self._keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press
        )
        self._keyboard_listener.start()

        # loading calibration info
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        for tracker_info in self._tracker_cfg:
            cur_uid = None; cur_calib_param = _TrackerCalibration()
            for key, value in tracker_info.items():
                if 'uid' in key: cur_uid = value
                elif 'path' in key:
                    if not os.path.isabs(value):
                        value = os.path.join(cur_dir, "../..", value)
                    assert os.path.isfile(value)
                    with open(value, "r") as f:
                        loaded = json.load(f)
                    loaded_T = np.asarray(loaded.get("T_W_LH"), dtype=float)
                    loaded_rot_offset = np.asarray(loaded.get("rotation_offset"))
                    if loaded_T.shape == (4, 4):
                        pose_extrinsic = np.zeros(7)
                        pose_extrinsic[:3] = loaded_T[:3, 3]
                        pose_extrinsic[3:] = R.from_matrix(loaded_T[:3, :3]).as_quat()
                        cur_calib_param.pose_extrinsic = pose_extrinsic
                    else:raise ValueError(f"[warn] {value} has invalid loaded_T shape: {loaded_T.shape}, expected (4, 4)")
                    if loaded_rot_offset.shape[-1] == 4:
                        cur_calib_param.rotation_offset = loaded_rot_offset
                    else: raise ValueError(f"[warn] {value} has invalid rot offset shape: {loaded_rot_offset.shape}, expected (4,)")
                elif 'translation_offset' in key:
                    cur_calib_param.translation_offset = value
                elif 'use_relative' in key:
                    cur_calib_param.use_relative = value
                    if value: self._use_relative = True
                self._calibration_by_uid[cur_uid] = cur_calib_param

        time_start = time.perf_counter()
        while time.perf_counter() - time_start < self._tracker_timout:
            if all(uid in self.vive_tracker.get_uids() for uid in self._calibration_by_uid.keys()):
                return True
        log.warn(f'Vive tracker could not get all uids {self._calibration_by_uid.keys()}, readed uids: {self.vive_tracker.get_uids()}')
        return False

    def _on_key_press(self, key):
        try:
            if key.char == 'i' and not self._key_pressed:
                self._relative_anchor = self.get_pose_dict(disable_realtive=True)
                self._key_pressed = True
                log.info(f'Calibrated Vive enabled!!!')
            elif key.char == 'u':
                self._key_pressed = False
                log.info(f'Calibrated Vive disabled!!!')
        except AttributeError:
              # ignore some hotkeys
              pass

    def stop(self) -> None:
        self.vive_tracker.stop()

    def is_running(self) -> bool:
        return self.vive_tracker.is_running()

    def get_uids(self) -> list[str]:
        return self.vive_tracker.get_uids()

    def get_pose_by_uid(self, uid: str) -> Optional[np.ndarray]:
        raw_pose = self.vive_tracker.get_pose_by_uid(uid)
        if raw_pose is None:
            return None
        return self._apply_calibration(uid, raw_pose)

    def get_pose_dict(self, disable_realtive=False) -> Dict[str, np.ndarray]:
        raw_pose_dict = self.vive_tracker.get_pose_dict()
        if raw_pose_dict is None: return None

        return {
            uid: self._apply_calibration(uid, pose, disable_realtive)
            for uid, pose in raw_pose_dict.items()
        }

    def _apply_calibration(self, uid: str, raw_pose: np.ndarray, disable_realtive=False) -> np.ndarray:
        uid_cali = self._calibration_by_uid.get(uid)
        if uid_cali is None: return None
        pose = np.asarray(raw_pose, dtype=np.float64)

        pose = transform_pose(uid_cali.pose_extrinsic, pose)
        pose[3:] = transform_quat(uid_cali.rotation_offset, pose[3:])
        if self._z_offset is not None: pose[2] += self._z_offset
        if uid_cali.translation_offset is not None:
            trans_pose_offset = np.array([0,0,0,0,0,0,1])
            trans_pose_offset[:3] = uid_cali.translation_offset
            pose = transform_pose(pose, trans_pose_offset)
        if disable_realtive: return pose

        # relative calculation
        if self._use_relative and self._relative_anchor is None:
            return None
        if self._calibration_by_uid[uid].use_relative:
            pose[:2] -= self._relative_anchor[uid][:2]
        return pose
    