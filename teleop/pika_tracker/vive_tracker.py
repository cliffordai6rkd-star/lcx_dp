#!/usr/bin/env python3
"""Realtime Vive tracker pose reader powered by pysurvive."""

from __future__ import annotations

import ctypes
import logging
import threading
import time
from typing import Any, Dict, Iterable, Optional, Tuple
import argparse
import numpy as np
import sys, os, json
from hardware.base.utils import transform_pose, transform_quat
from scipy.spatial.transform import Rotation as R

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
        poll_interval: float = 0.001,
        auto_start: bool = False,
    ) -> None:
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
        while not self._stop_event.is_set() and self._context_running():
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

    def _context_running(self) -> bool:
        if self._context is None:
            return False
        running_fn = getattr(self._context, "Running", None)
        if not callable(running_fn):
            return True
        try:
            return bool(running_fn())
        except Exception:
            return True

    def _create_context(self) -> Any:
        extra_survive_args = []
        argv = sys.argv[1:]
        if "--globalscenesolver" not in argv:
            argv += ["--globalscenesolver", "0"]

        argv += extra_survive_args
        return pysurvive.SimpleContext(argv)

    def _next_updated(self) -> Any:
        if self._context is None:
            return None

        for fn_name in ("NextUpdated", "next_updated"):
            fn = getattr(self._context, fn_name, None)
            if fn is not None:
                return fn()

        raise RuntimeError("No func method found on pysurvive context update")

    def _extract_pose(self, updated: Any) -> Optional[Tuple[str, np.ndarray]]:
        pose_fn = getattr(updated, "Pose", None)
        if callable(pose_fn):
            pose_obj = pose_fn()
            pose_data, time_stamp = pose_obj
            if pose_data is None:
                return None
            pos = self._as_array3(getattr(pose_data, "Pos", None))
            if pos is None:
                return None
            rot = self._as_quat4(getattr(pose_data, "Rot", None))
            if rot is None:
                return None 

            uid = self._device_uid(updated)
            # quaternion to xyzw
            tracker_pose = np.array([pos[0], pos[1], pos[2],
                rot[1], rot[2], rot[3], rot[0],], dtype=np.float64)
            return uid, tracker_pose
        raise ValueError(f'Could not extract pose from update')
    
    def _as_array3(self, value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None

        try:
            return np.array([value[0], value[1], value[2]], dtype=np.float64)
        except Exception as e:
            raise ValueError(f'Cannot convert pysurvive pose to position {e}')

    def _as_quat4(self, value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None

        try:
            # pysurvive quaternion order is wxyz
            return np.array([value[0], value[1], value[2], value[3]], dtype=np.float64)
        except Exception as e:
            raise ValueError(f'Cannot convert pysurvive pose to rotation quaternion {e}')

    def _device_uid(self, updated: Any) -> str:
        if _simple_serial_number is not None and hasattr(updated, "ptr"):
            try:
                uid = _simple_serial_number(updated.ptr)
                return str(uid, "utf-8")
            except Exception as e:
                raise ValueError(f"Cannot extract device uid from pysurvive update: {e}")
        raise ValueError("No valid device uid found")

    def __enter__(self) -> "PysurviveTrackerReader":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        self.stop()


class ViveTracker(PysurviveTrackerReader):
    """Backward-friendly class name."""


###################### for main entry test ######################
def _set_3d_axis_limits(ax: Any, points: np.ndarray) -> None:
    if points.size == 0:
        center = np.zeros(3, dtype=np.float64)
        span = 1.0
    else:
        lo = points.min(axis=0)
        hi = points.max(axis=0)
        center = (lo + hi) / 2.0
        span = max(float(np.max(hi - lo)), 0.4) + 0.2
    half = span / 2.0
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def _run_visualizer(reader: ViveTracker, uid: str, hz: float, axis_length: float, T_W_LH:None, R_offset: None) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("matplotlib is required for --viz mode") from exc

    fig = plt.figure("Vive Tracker Pose Visualizer")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Tracker Pose (position + orientation)")
    status_text = ax.text2D(0.02, 0.98, "", transform=ax.transAxes, va="top")
    artists: list[Any] = []

    def _clear_artists() -> None:
        while artists:
            artist = artists.pop()
            try:
                artist.remove()
            except Exception:
                pass

    def _get_pose_items() -> list[Tuple[str, np.ndarray]]:
        if uid:
            pose = reader.get_pose_by_uid(uid)
            if pose is None:
                return []
            return [(uid, pose)]
        return sorted(reader.get_pose_dict().items(), key=lambda item: item[0])

    axis_colors = ("tab:red", "tab:green", "tab:blue")

    def _update(_frame: int) -> list[Any]:
        _clear_artists()
        pose_items = _get_pose_items()

        if not pose_items:
            status_text.set_text("No tracker pose yet")
            _set_3d_axis_limits(ax, np.empty((0, 3), dtype=np.float64))
            return [status_text]

        all_points: list[np.ndarray] = []
        for idx, (tracker_uid, pose) in enumerate(pose_items):
            if T_W_LH is not None:
                pose = transform_pose(T_W_LH, pose)
                if R_offset is not None:
                    pose[3:] = transform_quat(R_offset, pose[3:])
            pos = np.asarray(pose[:3], dtype=np.float64)
            rot = R.from_quat(pose[3:]).as_matrix()
            all_points.append(pos)

            marker = ax.scatter(
                [pos[0]],
                [pos[1]],
                [pos[2]],
                s=32,
                color=f"C{idx % 10}",
            )
            artists.append(marker)
            label = ax.text(pos[0], pos[1], pos[2], tracker_uid, fontsize=8)
            artists.append(label)

            for axis_idx, color in enumerate(axis_colors):
                end = pos + rot[:, axis_idx] * axis_length
                line = ax.plot(
                    [pos[0], end[0]],
                    [pos[1], end[1]],
                    [pos[2], end[2]],
                    color=color,
                    linewidth=2.0,
                )[0]
                artists.append(line)
                all_points.append(end)

        stacked = np.vstack(all_points) if all_points else np.empty((0, 3), dtype=np.float64)
        _set_3d_axis_limits(ax, stacked)
        status_text.set_text(
            f"tracked={len(pose_items)} | axis colors: X=red, Y=green, Z=blue"
        )
        return artists + [status_text]

    interval_ms = max(int(1000.0 / max(hz, 1e-6)), 10)
    animation = FuncAnimation(fig, _update, interval=interval_ms, blit=False, cache_frame_data=False)
    _ = animation  # keep reference to avoid GC
    plt.show()


def _main() -> None:
    parser = argparse.ArgumentParser(description="Test realtime Vive tracker reader")
    parser.add_argument("--uid", type=str, default="LHR-0DFD738C", help="Only print pose for this uid")
    parser.add_argument("--out", type=str, default="config/T_W_LH.json")
    parser.add_argument("--hz", type=float, default=20.0, help="Print frequency")
    parser.add_argument("--poll-interval", type=float, default=0.001, help="Background poll interval")
    parser.add_argument("--viz", action="store_true", help="Show realtime 3D pose visualization")
    parser.add_argument(
        "--viz-axis-length",
        type=float,
        default=0.08,
        help="Length (meters) of orientation axes in visualization",
    )
    parser.add_argument(
        "--survive-arg",
        action="append",
        default=[],
        help="Extra arg passed to pysurvive context. Can be used multiple times.",
    )
    args = parser.parse_args()

    sleep_dt = 1.0 / max(args.hz, 1e-6)
    reader = ViveTracker(poll_interval=args.poll_interval, auto_start=True)
    T_W_LH = None; R_offset = None
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = args.out if os.path.isabs(args.out) else os.path.join(cur_dir, args.out)

    # If a previous calibration exists, load it as the current extrinsic.
    if os.path.isfile(out_path):
        try:
            with open(out_path, "r") as f:
                loaded = json.load(f)
            loaded_T = np.asarray(loaded.get("T_W_LH"), dtype=float)
            loaded_R = np.asarray(loaded.get("rotation_offset"))
            if loaded_T.shape == (4, 4):
                T_W_LH = loaded_T
                pose_extrinsic = np.zeros(7)
                pose_extrinsic[:3] = T_W_LH[:3, 3]
                pose_extrinsic[3:] = R.from_matrix(T_W_LH[:3, :3]).as_quat()
                T_W_LH = pose_extrinsic
                print(f"Loaded existing T_W_LH from: {out_path} with 7D repre: {T_W_LH}")
            else:
                print(f"[warn] {out_path} has invalid T_W_LH shape: {loaded_T.shape}, expected (4, 4)")
            if loaded_R.shape[-1] == 4:
                R_offset = loaded_R 
        except Exception as e:
            print(f"[warn] Failed to load existing extrinsic from {out_path}: {e}")

    print(f"ViveTracker test started with extrinsic {T_W_LH}, {R_offset}. Press Ctrl+C to stop.")
    try:
        if args.viz:
            print("Visualization mode enabled. Close the plot window to stop.")
            _run_visualizer(reader, uid=args.uid, hz=args.hz, axis_length=max(args.viz_axis_length, 1e-4), T_W_LH=T_W_LH, R_offset=R_offset)
        else:
            while True:
                if args.uid:
                    pose = reader.get_pose_by_uid(args.uid)
                    if pose is None:
                        print(f"[uid={args.uid}] no pose yet")
                    else:
                        if T_W_LH is not None:
                            pose = transform_pose(T_W_LH, pose)
                        print(f"[uid={args.uid}] {np.array2string(pose, precision=6, suppress_small=True)}")
                else:
                    pose_dict = reader.get_pose_dict()
                    if not pose_dict:
                        print("no tracker pose yet")
                    else:
                        print(f"tracked={len(pose_dict)}")
                        for uid, pose in pose_dict.items():
                            if T_W_LH is not None:
                                pose = transform_pose(T_W_LH, pose)
                            print(f"  {uid}: {np.array2string(pose, precision=6, suppress_small=True)}")
                time.sleep(sleep_dt)
    except KeyboardInterrupt:
        print("\nStopping ViveTracker test...")
    finally:
        reader.stop()


if __name__ == "__main__":
    _main()
