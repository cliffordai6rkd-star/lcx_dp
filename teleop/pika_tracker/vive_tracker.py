#!/usr/bin/env python3
"""Realtime Vive tracker pose reader powered by pysurvive."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Iterable, Optional, Tuple
import argparse
import numpy as np

logger = logging.getLogger(__name__)

try:
    import pysurvive
except ImportError:  # pragma: no cover - depends on runtime environment
    pysurvive = None

try:
    from pysurvive.pysurvive_generated import simple_serial_number as _simple_serial_number
except Exception:  # pragma: no cover - symbol may not exist in all bindings
    _simple_serial_number = None


class PysurviveTrackerReader:
    """
    Read all tracker poses from pysurvive in a background thread.

    Internal cache format:
        {tracker_uid: np.ndarray(7,)} -> [x, y, z, qx, qy, qz, qw]
    """

    def __init__(
        self,
        args: Optional[Iterable[str]] = None,
        poll_interval: float = 0.001,
        auto_start: bool = False,
    ) -> None:
        self._args = list(args) if args else []
        self._poll_interval = max(float(poll_interval), 0.0)

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._context: Any = None
        self._latest_poses: Dict[str, np.ndarray] = {}
        self._running = False

        if auto_start:
            self.start()

    def start(self) -> bool:
        """Start background polling."""
        if self._running:
            return True
        if pysurvive is None:
            raise ImportError("pysurvive is not installed")

        self._context = self._create_context()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="vive-tracker-reader")
        self._running = True
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop background polling."""
        if not self._running:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._context = None
        self._running = False

    def is_running(self) -> bool:
        return self._running

    def get_all_poses(self) -> Dict[str, np.ndarray]:
        """Thread-safe snapshot of all tracker poses."""
        with self._lock:
            return {uid: pose.copy() for uid, pose in self._latest_poses.items()}

    def get_pose(self, uid: str) -> Optional[np.ndarray]:
        """Thread-safe latest pose for a given tracker uid."""
        with self._lock:
            pose = self._latest_poses.get(uid)
            return None if pose is None else pose.copy()

    def get_pose_dict(self) -> Dict[str, np.ndarray]:
        """Alias of get_all_poses()."""
        return self.get_all_poses()

    def get_pose_by_uid(self, uid: str) -> Optional[np.ndarray]:
        """Alias of get_pose(uid)."""
        return self.get_pose(uid)

    def get_uids(self) -> list[str]:
        with self._lock:
            return list(self._latest_poses.keys())

    def clear(self) -> None:
        with self._lock:
            self._latest_poses.clear()

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                updated = self._next_updated()
                if updated is None:
                    time.sleep(self._poll_interval)
                    continue

                parsed_pose = self._extract_pose(updated)
                if parsed_pose is None:
                    time.sleep(self._poll_interval)
                    continue

                uid, pose = parsed_pose
                with self._lock:
                    self._latest_poses[uid] = pose
            except Exception as exc:  # pragma: no cover - hardware/runtime dependent
                logger.exception("pysurvive polling loop error: %s", exc)
                time.sleep(max(self._poll_interval, 0.01))

    def _create_context(self) -> Any:
        for cls_name in ("SimpleContext", "SurviveSimpleContext", "Context", "SurviveContext"):
            cls = getattr(pysurvive, cls_name, None)
            if cls is None:
                continue
            try:
                return cls(self._args) if self._args else cls()
            except TypeError:
                return cls()
        raise RuntimeError("Cannot find a compatible pysurvive context class")

    def _next_updated(self) -> Any:
        if self._context is None:
            return None

        for fn_name in ("NextUpdated", "next_updated"):
            fn = getattr(self._context, fn_name, None)
            if fn is not None:
                return fn()

        for fn_name in ("poll", "Poll", "poll_next_event", "NextEvent"):
            fn = getattr(self._context, fn_name, None)
            if fn is not None:
                return fn()

        raise RuntimeError("No polling method found on pysurvive context")

    def _extract_pose(self, updated: Any) -> Optional[Tuple[str, np.ndarray]]:
        # Typical pysurvive object path: updated.Pose() -> (pose_data, timestamp)
        pose_fn = getattr(updated, "Pose", None)
        if callable(pose_fn):
            pose_obj = pose_fn()
            pose_data, _ = self._split_pose_obj(pose_obj)
            if pose_data is None:
                return None

            pos = getattr(pose_data, "Pos", None)
            rot = getattr(pose_data, "Rot", None)
            if pos is None or rot is None or len(pos) < 3 or len(rot) < 4:
                return None

            uid = self._device_uid(updated)
            tracker_pose = np.array(
                [
                    float(pos[0]),
                    float(pos[1]),
                    float(pos[2]),
                    float(rot[1]),
                    float(rot[2]),
                    float(rot[3]),
                    float(rot[0]),
                ],
                dtype=np.float64,
            )
            return uid, tracker_pose

        # Event path fallback: event.pose may contain [x, y, z, ...]
        event_pose = getattr(updated, "pose", None) or getattr(updated, "Pose", None)
        if event_pose is None:
            return None

        if hasattr(event_pose, "__iter__"):
            arr = list(event_pose)
            if len(arr) < 3:
                return None
            uid = self._device_uid(updated)
            quat = [0.0, 0.0, 0.0, 1.0]
            if len(arr) >= 7:
                quat = [float(arr[3]), float(arr[4]), float(arr[5]), float(arr[6])]
            tracker_pose = np.array(
                [float(arr[0]), float(arr[1]), float(arr[2]), quat[0], quat[1], quat[2], quat[3]],
                dtype=np.float64,
            )
            return uid, tracker_pose

        return None

    def _split_pose_obj(self, pose_obj: Any) -> Tuple[Optional[Any], float]:
        if pose_obj is None:
            return None, time.time()

        if isinstance(pose_obj, (tuple, list)):
            if len(pose_obj) >= 2:
                return pose_obj[0], float(pose_obj[1])
            if len(pose_obj) == 1:
                return pose_obj[0], time.time()

        return pose_obj, time.time()

    def _device_uid(self, updated: Any) -> str:
        if _simple_serial_number is not None and hasattr(updated, "ptr"):
            try:
                uid = _simple_serial_number(updated.ptr)
                if isinstance(uid, (bytes, bytearray)):
                    return uid.decode("utf-8", errors="ignore")
                if uid:
                    return str(uid)
            except Exception:
                pass

        for attr_name in ("serial", "device_serial", "uid", "UID", "name", "obj_name"):
            value = getattr(updated, attr_name, None)
            if value:
                if isinstance(value, (bytes, bytearray)):
                    return value.decode("utf-8", errors="ignore")
                return str(value)

        for method_name in ("SerialNumber", "UID", "Name"):
            fn = getattr(updated, method_name, None)
            if callable(fn):
                try:
                    value = fn()
                    if isinstance(value, (bytes, bytearray)):
                        value = value.decode("utf-8", errors="ignore")
                    if value:
                        return str(value)
                except Exception:
                    pass

        return f"unknown_{id(updated)}"

    def __enter__(self) -> "PysurviveTrackerReader":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        self.stop()


class ViveTracker(PysurviveTrackerReader):
    """Backward-friendly class name."""


def _main() -> None:
    parser = argparse.ArgumentParser(description="Test realtime Vive tracker reader")
    parser.add_argument("--uid", type=str, default="", help="Only print pose for this uid")
    parser.add_argument("--hz", type=float, default=20.0, help="Print frequency")
    parser.add_argument("--poll-interval", type=float, default=0.001, help="Background poll interval")
    parser.add_argument(
        "--survive-arg",
        action="append",
        default=[],
        help="Extra arg passed to pysurvive context. Can be used multiple times.",
    )
    args = parser.parse_args()

    sleep_dt = 1.0 / max(args.hz, 1e-6)
    reader = ViveTracker(args=args.survive_arg, poll_interval=args.poll_interval, auto_start=True)

    print("ViveTracker test started. Press Ctrl+C to stop.")
    try:
        while True:
            if args.uid:
                pose = reader.get_pose_by_uid(args.uid)
                if pose is None:
                    print(f"[uid={args.uid}] no pose yet")
                else:
                    print(f"[uid={args.uid}] {np.array2string(pose, precision=6, suppress_small=True)}")
            else:
                pose_dict = reader.get_pose_dict()
                if not pose_dict:
                    print("no tracker pose yet")
                else:
                    print(f"tracked={len(pose_dict)}")
                    for uid, pose in pose_dict.items():
                        print(f"  {uid}: {np.array2string(pose, precision=6, suppress_small=True)}")
            time.sleep(sleep_dt)
    except KeyboardInterrupt:
        print("\nStopping ViveTracker test...")
    finally:
        reader.stop()


if __name__ == "__main__":
    _main()
