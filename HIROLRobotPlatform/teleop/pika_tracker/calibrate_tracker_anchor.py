from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import glog as log
from scipy.spatial.transform import Rotation as R
import time
from pynput import keyboard

from teleop.pika_tracker.vive_tracker import ViveTracker
from hardware.base.utils import dynamic_load_yaml, transform_pose, transform_quat, negate_pose, negate_quat

class CalibrateTracker:
    def __init__(self, config):
        self._config = dict(config)
        poll_interval = float(self._config.get("reading_loop_s", self._config.get("poll_interval", 0.001)))
        auto_start = bool(self._config.get("auto_start", False))
        self._vive_tracker = ViveTracker(poll_interval=poll_interval, auto_start=auto_start, require_rot_similar_transform=True)
        self._tracker_timout = config.get("timeout", 5.0)
        self._z_offset = config.get("z_offset", None)

        self._tracker_uids = config["tracker_uids"]
        self._anchor_tracker = config["anchor_uid"]
        assert self._anchor_tracker in self._tracker_uids, f"Anchor tracker {self._anchor_tracker} must be in tracker_uids {self._tracker_uids}"        
        self._anchor_pose = None; self._anchor_pose_inv = None
        self._use_relative = config.get("use_relative", False)
        self._relative_anchor = None
        
        self._is_initialized = False
        self._is_initialized = self.initialize()
        if not self._is_initialized:
            raise ValueError(f'Calibration tracker init failed')

    def initialize(self) -> bool:
        if self._is_initialized: return True
        
        self._vive_tracker.start()
        # keyboard listening for update init pose
        self._key_pressed = False
        self._keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press
        )
        self._keyboard_listener.start()


        time_start = time.perf_counter()
        while time.perf_counter() - time_start < self._tracker_timout:
            if all(uid in self._vive_tracker.get_uids() for uid in self._tracker_uids):
                self._anchor_pose = self._vive_tracker.get_pose_by_uid(self._anchor_tracker)    
                self._anchor_pose_inv  = negate_pose(self._anchor_pose)
                return True
        log.warn(f'Vive tracker could not get all uids {self._tracker_uids}, readed uids: {self._vive_tracker.get_uids()}')
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
        self._vive_tracker.stop()

    def is_running(self) -> bool:
        return self._vive_tracker.is_running()

    def get_uids(self) -> list[str]:
        return self._vive_tracker.get_uids()

    def get_pose_by_uid(self, uid: str) -> Optional[np.ndarray]:
        raw_pose = self._vive_tracker.get_pose_by_uid(uid)
        if raw_pose is None:
            return None
        return self._apply_calibration(uid, raw_pose)

    def get_pose_dict(self, disable_realtive=False) -> Dict[str, np.ndarray]:
        raw_pose_dict = self._vive_tracker.get_pose_dict()
        if raw_pose_dict is None: return None

        return {
            uid: self._apply_calibration(uid, pose, disable_realtive)
            for uid, pose in raw_pose_dict.items()
        }

    def _apply_calibration(self, uid: str, raw_pose: np.ndarray, disable_realtive=False) -> np.ndarray:
        pose = np.asarray(raw_pose, dtype=np.float64)
        anchor_pose = self._vive_tracker.get_pose_by_uid(self._anchor_tracker)
        if anchor_pose is None: return None

        # anchor alignment
        pose[3:] = transform_quat(negate_quat(anchor_pose[3:]), pose[3:])
        if self._z_offset is not None: pose[2] += self._z_offset
        if disable_realtive: return pose

        # relative calculation
        if self._use_relative:
            if self._relative_anchor is None: return None
            else: pose[:2] -= self._relative_anchor[uid][:2]
        return pose


def _relative_rotation_report(anchor_pose: np.ndarray, tracker_pose: np.ndarray) -> dict:
    rel_rot = R.from_quat(anchor_pose[3:]).inv() * R.from_quat(tracker_pose[3:])
    rotvec_deg = np.rad2deg(rel_rot.as_rotvec())
    return {
        "quat_xyzw": rel_rot.as_quat(),
        "euler_xyz_deg": rel_rot.as_euler("xyz", degrees=True),
        "rotvec_deg": rotvec_deg,
        "angle_deg": float(np.linalg.norm(rotvec_deg)),
    }


def _format_report(name: str, report: dict) -> str:
    euler = np.array2string(report["euler_xyz_deg"], precision=3, suppress_small=True)
    rotvec = np.array2string(report["rotvec_deg"], precision=3, suppress_small=True)
    quat = np.array2string(report["quat_xyzw"], precision=5, suppress_small=True)
    return f"{name}: angle_deg={report['angle_deg']:.4f} euler_xyz_deg={euler} rotvec_deg={rotvec} quat_xyzw={quat}"


def _wait_for_uids(reader: ViveTracker, uids: list[str], timeout_s: float) -> bool:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout_s:
        if all(uid in reader.get_uids() for uid in uids):
            return True
        time.sleep(0.05)
    return False


def _resolve_target_uid(all_uids: list[str], anchor_uid: str, target_uid: Optional[str]) -> str:
    if target_uid is not None:
        if target_uid not in all_uids:
            raise ValueError(f"Requested target uid {target_uid} not found in tracker_uids {all_uids}")
        if target_uid == anchor_uid:
            raise ValueError("Target uid must be different from anchor uid for relative-rotation diagnosis")
        return target_uid

    for uid in all_uids:
        if uid != anchor_uid:
            return uid
    raise ValueError("Could not infer a target uid because tracker_uids only contains the anchor uid")


def _apply_vive_rotation_alignment(raw_pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(raw_pose, dtype=np.float64).copy()
    quat_offset = np.array([0.7071068, 0.7071068, 0.0, 0.0], dtype=np.float64)
    pose[3:] = transform_quat(transform_quat(negate_quat(quat_offset), pose[3:]), quat_offset)
    return pose


def _main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose relative rotation between an anchor tracker and another tracker")
    parser.add_argument(
        "--config",
        type=str,
        default="common/hardwares/configs/calib_tracker_anchor_cfg.yaml",
        help="Path to tracker-anchor config yaml",
    )
    parser.add_argument("--uid", type=str, default="LHR-E9789F55", help="Tracker uid to compare against the anchor")
    parser.add_argument("--hz", type=float, default=5.0, help="Print frequency")
    parser.add_argument("--timeout", type=float, default=None, help="Override config timeout in seconds")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.getcwd(), config_path)
    config = dynamic_load_yaml(config_path)
    tracker_cfg = dict(config["tracker"] if "tracker" in config else config)
    if args.timeout is not None:
        tracker_cfg["timeout"] = float(args.timeout)

    tracker_uids = list(tracker_cfg["tracker_uids"])
    anchor_uid = tracker_cfg["anchor_uid"]
    target_uid = _resolve_target_uid(tracker_uids, anchor_uid, args.uid)
    poll_interval = float(tracker_cfg.get("reading_loop_s", tracker_cfg.get("poll_interval", 0.001)))
    timeout_s = float(tracker_cfg.get("timeout", 5.0))
    z_offset = tracker_cfg.get("z_offset", None)

    reader = ViveTracker(
        poll_interval=poll_interval,
        auto_start=True,
        require_rot_similar_transform=False,
    )
    sleep_dt = 1.0 / max(args.hz, 1e-6)

    try:
        if not _wait_for_uids(reader, tracker_uids, timeout_s):
            raise TimeoutError(f"reader could not observe all tracker_uids={tracker_uids} within {timeout_s}s")

        anchor_raw_init = reader.get_pose_by_uid(anchor_uid)
        if anchor_raw_init is None:
            raise RuntimeError(f"Could not get initial anchor pose for uid={anchor_uid}")
        anchor_aligned_init = _apply_vive_rotation_alignment(anchor_raw_init)
        anchor_pose_inv = negate_pose(anchor_aligned_init)

        print(f"anchor_uid={anchor_uid} target_uid={target_uid} poll_interval={poll_interval}s")
        print("Press Ctrl+C to stop. Keep both trackers static in the same physical pose for diagnosis.")

        while True:
            raw_anchor = reader.get_pose_by_uid(anchor_uid)
            raw_target = reader.get_pose_by_uid(target_uid)

            if any(pose is None for pose in (raw_anchor, raw_target)):
                print("Waiting for complete pose set...")
                time.sleep(sleep_dt)
                continue

            aligned_anchor = _apply_vive_rotation_alignment(raw_anchor)
            aligned_target = _apply_vive_rotation_alignment(raw_target)
            final_anchor = transform_pose(anchor_pose_inv, aligned_anchor)
            final_target = transform_pose(anchor_pose_inv, aligned_target)
            if z_offset is not None:
                final_anchor[2] += z_offset
                final_target[2] += z_offset

            raw_report = _relative_rotation_report(raw_anchor, raw_target)
            aligned_report = _relative_rotation_report(aligned_anchor, aligned_target)
            final_report = _relative_rotation_report(final_anchor, final_target)

            print("=" * 80)
            print(_format_report("raw_rel", raw_report))
            print(_format_report("aligned_rel", aligned_report))
            print(_format_report("final_rel", final_report))
            print(
                "final_target_pos_xyz="
                f"{np.array2string(final_target[:3], precision=4, suppress_small=True)} "
                "final_anchor_pos_xyz="
                f"{np.array2string(final_anchor[:3], precision=4, suppress_small=True)}"
            )
            time.sleep(sleep_dt)
    except KeyboardInterrupt:
        print("\nStopping tracker-anchor diagnosis...")
    finally:
        reader.stop()


if __name__ == "__main__":
    _main()
