#!/usr/bin/env python3
# calibrate_lh_to_world_viz_ransac.py

import argparse, json, time, math, threading
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import ctypes
import pysurvive, sys
from pysurvive.pysurvive_generated import *
import os
from scipy.spatial.transform import Rotation as SciR

# ----------------- math utils -----------------
def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v

def rodrigues(axis, theta):
    axis = normalize(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=float)
    return np.eye(3) + math.sin(theta)*K + (1-math.cos(theta))*(K@K)

def rot_from_a_to_b(a, b):
    a = normalize(a); b = normalize(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        tmp = np.array([1,0,0]) if abs(a[0]) < 0.9 else np.array([0,1,0])
        axis = normalize(np.cross(a, tmp))
        return rodrigues(axis, math.pi)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]], dtype=float)
    return np.eye(3) + vx + vx@vx * ((1 - c)/(s**2))

def build_T(R, t):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3]  = t
    return T

def apply_T(T, p):
    ph = np.array([p[0], p[1], p[2], 1.0], dtype=float)
    out = T @ ph
    return out[:3]

# ----------------- RANSAC plane fit -----------------
def plane_from_3pts(a, b, c):
    n = np.cross(b - a, c - a)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-10:
        return None, None
    n = n / n_norm
    d = -np.dot(n, a)  # plane: n·x + d = 0
    return n, d

def ransac_plane(points, thresh=0.01, iters=500, min_inliers=12, refine=True, seed=0):
    """
    points: (N,3)
    thresh: inlier distance threshold (m). Typical 0.005~0.015
    returns: (n, d, inlier_mask)
    """
    rng = np.random.default_rng(seed)
    P = np.asarray(points, dtype=float)
    N = P.shape[0]
    if N < 3:
        raise ValueError("Need at least 3 points")

    best_inliers = None
    best_count = -1
    best_model = None

    for _ in range(iters):
        idx = rng.choice(N, size=3, replace=False)
        n, d = plane_from_3pts(P[idx[0]], P[idx[1]], P[idx[2]])
        if n is None:
            continue
        dist = np.abs(P @ n + d)  # since n is unit
        inliers = dist < thresh
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_model = (n, d)
            if best_count > 0.9 * N:
                break

    if best_model is None or best_count < min_inliers:
        raise RuntimeError(f"RANSAC failed: best_inliers={best_count}/{N}")

    n, d = best_model
    inliers = best_inliers

    if refine:
        Pin = P[inliers]
        c = Pin.mean(axis=0)
        X = Pin - c
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        n = normalize(Vt[-1])
        d = -np.dot(n, c)

    return n, d, inliers

def calibrate_LH_to_world_with_inliers(floor_inliers, p0, p1, pz):
    """
    floor_inliers: (M,3) in LH frame, M>=3
    p0, p1: (3,) in LH frame (means of samples)
    pz: (3,) in LH frame, a point sampled above the floor plane
    Build W s.t. origin=P0, +X along P0->P1 (projected to floor), +Z floor normal.
    """
    # Floor normal from inliers
    Pin = np.asarray(floor_inliers, dtype=float)
    c = Pin.mean(axis=0)
    X = Pin - c
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    n = normalize(Vt[-1])

    # Plane fitting only determines the normal up to a sign. Use a sampled
    # point above the floor to keep +Z in W on the physically upward side.
    signed_height = float(np.dot(np.asarray(pz, dtype=float) - c, n))
    if abs(signed_height) < 1e-3:
        raise ValueError("Pz is too close to the floor plane; cannot determine +Z direction reliably.")
    if signed_height < 0:
        n = -n

    # Align normal to +Z
    R_floor = rot_from_a_to_b(n, np.array([0,0,1.0]))
    p0p = R_floor @ p0
    p1p = R_floor @ p1

    # Yaw to align P0->P1 projection to +X
    v = p1p - p0p
    v[2] = 0.0
    v = normalize(v)
    yaw = math.atan2(v[1], v[0])
    R_yaw = rodrigues(np.array([0,0,1.0]), -yaw)

    R = R_yaw @ R_floor
    t = -(R @ p0)  # map P0 to origin
    return build_T(R, t), n

# ----------------- pysurvive adapter (best-effort) -----------------
def make_ctx(extra_survive_args=None):
    extra_survive_args = extra_survive_args or []
    argv = ["calib_py", "--globalscenesolver", "0"] + extra_survive_args
    return pysurvive.SimpleContext(argv)

def _as_array3(v):
    # 1) indexable
    try:
        return np.array([v[0], v[1], v[2]], dtype=float)
    except Exception:
        pass
    # 2) has x/y/z
    if all(hasattr(v, a) for a in ("x","y","z")):
        return np.array([v.x, v.y, v.z], dtype=float)
    if all(hasattr(v, a) for a in ("X","Y","Z")):
        return np.array([v.X, v.Y, v.Z], dtype=float)

    # 3) ctypes: try to read raw memory as 3 floats
    try:
        buf = ctypes.string_at(ctypes.addressof(v), ctypes.sizeof(v))
        arr = np.frombuffer(buf, dtype=np.float32)
        if arr.size >= 3:
            return arr[:3].astype(float)
    except Exception:
        pass
    raise RuntimeError(f"Cannot parse vec3 from type {type(v)}")

def _as_quat4(q):
    try:
        return np.array([q[1], q[2], q[3], q[0]], dtype=float)
    except Exception:
        pass
    if all(hasattr(q, a) for a in ("w","x","y","z")):
        return np.array([q.w, q.x, q.y, q.z], dtype=float)
    if all(hasattr(q, a) for a in ("W","X","Y","Z")):
        return np.array([q.W, q.X, q.Y, q.Z], dtype=float)
    try:
        buf = ctypes.string_at(ctypes.addressof(q), ctypes.sizeof(q))
        arr = np.frombuffer(buf, dtype=np.float32)
        if arr.size >= 4:
            return arr[:4].astype(float)
    except Exception:
        pass
    return None  # rotation optional

def extract_pos_from_updated(updated):
    """
    updated: object with Pose() -> tuple(len=2), tuple[0] is struct_LinmathPose
    Returns pos(3,), quat_wxyz(4,) or None
    """
    pt = updated.Pose()
    pose = pt[0] if isinstance(pt, tuple) else pt  # struct_LinmathPose

    pos = _as_array3(pose.Pos)
    quat = _as_quat4(pose.Rot)  # may be None if fails

    return pos, quat

# ----------------- live pose reader -----------------
class PoseStream:
    def __init__(self, tracker_substr="", maxlen=400):
        self.ctx = make_ctx()
        self.tracker_substr = tracker_substr
        self._quat_offset = SciR.from_quat(np.array([0.7071068, 0.7071068, 0, 0]))
        self._negate_quat_offset = self._quat_offset.inv()
        self.lock = threading.Lock()
        self.buf = deque(maxlen=maxlen)  # (t, pos, quat_wxyz key)
        self.seen = {}
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running and self.ctx.Running():
            updated = self.ctx.NextUpdated()
            if not updated:
                continue

            # key = updated.Name()   # 通常是 "T20" 之类（在你的 log 里就是 T20）
            key = str(simple_serial_number(updated.ptr), 'utf-8')
            self.seen[key] = self.seen.get(key, 0) + 1

            if self.tracker_substr and self.tracker_substr not in key:
                continue

            pos, quat = extract_pos_from_updated(updated)
            # @TODO: need to test
            rot = SciR.from_quat(quat)
            rot = self._negate_quat_offset * rot * self._quat_offset
            quat = rot.as_quat()

            with self.lock:
                self.buf.append((time.time(), pos.copy(), quat.copy(), key))

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)

    def snapshot(self):
        with self.lock:
            return list(self.buf)

# ----------------- stability detector -----------------
def compute_speed(history, window_s=0.25):
    if len(history) < 5:
        return None
    t_now = history[-1][0]
    pts = [(t,p) for (t,p,_,_) in history if (t_now - t) <= window_s]
    if len(pts) < 3:
        return None
    t0, p0 = pts[0]
    t1, p1 = pts[-1]
    dt = max(t1 - t0, 1e-6)
    return float(np.linalg.norm(p1 - p0) / dt)

def robust_mean_pos(history, window_s=0.15):
    if len(history) < 5:
        return None
    t_now = history[-1][0]
    pts = np.array([p for (t,p,_,_) in history if (t_now - t) <= window_s], dtype=float)
    if pts.shape[0] < 3:
        return None
    med = np.median(pts, axis=0)
    d = np.linalg.norm(pts - med, axis=1)
    keep = d <= np.quantile(d, 0.8)  # keep closest 80%
    if keep.sum() < 2:
        keep = np.ones_like(keep, dtype=bool)
    return pts[keep].mean(axis=0)

# ----------------- visualization + interaction -----------------
def autoscale(ax, pts, margin=0.2):
    if pts is None or len(pts) == 0:
        return
    pts = np.array(pts, dtype=float)
    lo = pts.min(axis=0) - margin
    hi = pts.max(axis=0) + margin
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])

# ----------------- calibration zero rotation offset -----------------
def average_quat_xyzw_markley(quats_xyzw: np.ndarray) -> np.ndarray:
    """
    Markley quaternion average. Input/output: xyzw.
    """
    Q = np.asarray(quats_xyzw, dtype=np.float64)
    assert Q.ndim == 2 and Q.shape[1] == 4

    # align hemisphere
    ref = Q[0].copy()
    Q2 = Q.copy()
    for i in range(Q2.shape[0]):
        if np.dot(Q2[i], ref) < 0:
            Q2[i] = -Q2[i]

    A = np.zeros((4, 4), dtype=np.float64)
    for q in Q2:
        A += np.outer(q, q)
    A /= Q2.shape[0]

    vals, vecs = np.linalg.eigh(A)
    q_avg = vecs[:, np.argmax(vals)]
    q_avg = q_avg / np.linalg.norm(q_avg)
    return q_avg  # xyzw


def rot_offset_from_history(
    history,
    T_W_LH: np.ndarray,
    still_speed_thresh: float,
    window_s: float = 0.35,
):
    """
    从 history 里选取“静止帧”，计算 R_tracker<-tool（默认 tool target = world axes）
    返回: R_tracker_tool (SciR) 以及 debug 字典
    """
    # 取最近 window_s 秒
    if len(history) < 10:
        return None, {"reason": "history too short"}

    t_now = history[-1][0]
    seg = [(t, p, q_xyzw, k) for (t, p, q_xyzw, k) in history if (t_now - t) <= window_s]
    if len(seg) < 10:
        return None, {"reason": "not enough samples in window"}

    # 计算窗口速度（端点差分）
    t0, p0 = seg[0][0], seg[0][1]
    t1, p1 = seg[-1][0], seg[-1][1]
    spd = float(np.linalg.norm(p1 - p0) / max(t1 - t0, 1e-6))
    if spd > still_speed_thresh:
        return None, {"reason": f"moving speed={spd:.4f} > {still_speed_thresh}"}

    # 从 seg 中取 quat，变成 W<-tracker
    # seg 里的 quat 是 wxyz（libsurvive），先转 xyzw
    quats_xyzw = []
    for (_, _, q_xyzw, _) in seg:
        # q_wxyz = np.asarray(q_wxyz, dtype=np.float64)
        # q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)

        # 这里假设 libsurvive 给的是 LH<-tracker 旋转（常见），则 W<-tracker = (W<-LH)*(LH<-tracker)
        R_W_LH = SciR.from_matrix(T_W_LH[:3, :3])
        R_LH_tracker = SciR.from_quat(q_xyzw)
        R_W_tracker = R_W_LH * R_LH_tracker
        quats_xyzw.append(R_W_tracker.as_quat())

    if len(quats_xyzw) < 10:
        return None, {"reason": "not enough quat samples"}

    # 为了更稳：如果窗口内点太少，就累计多次触发（见下面主循环的 accumulator）
    # 这里先返回这个窗口的平均
    q_avg_xyzw = average_quat_xyzw_markley(np.vstack(quats_xyzw))
    R_W_tracker_avg = SciR.from_quat(q_avg_xyzw)

    # 目标：tool 在 W 下对齐世界轴（Identity）
    R_W_tool_target = SciR.identity()

    # R_tracker<-tool = (R_W_tracker)^-1 * R_W_tool_target
    R_tracker_tool = R_W_tracker_avg.inv() * R_W_tool_target

    return np.array(R_tracker_tool.as_quat()), {"speed": spd, "n_window": len(quats_xyzw)}

def save_R_tracker_tool_into_out(out_path: str, rotation_offset, key: str = "rotation_offset"):
    """
    Save rotation offset into args.out JSON file by merging into existing dict.
    Does NOT delete other existing fields.
    """
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Load existing data if possible
    data = {}
    if os.path.isfile(out_path):
        try:
            with open(out_path, "r") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                data = loaded
            else:
                # keep a backup, then start fresh dict
                bak = out_path + ".bak"
                with open(bak, "w") as f:
                    json.dump(loaded, f, indent=2)
                data = {"_warning": f"Existing JSON was not an object; backed up to {bak}"}
        except Exception as e:
            # backup raw file contents to avoid losing anything
            bak = out_path + ".bak"
            try:
                with open(out_path, "r") as f:
                    raw = f.read()
                with open(bak, "w") as f:
                    f.write(raw)
            except Exception:
                pass
            data = {"_warning": f"Failed to parse existing JSON; backup attempted to {bak}. Error: {e}"}

    # Merge / append new field
    if not isinstance(rotation_offset, list):
        data[key] = rotation_offset.tolist()  # xyzw

    # Write back
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[OK] Appended {key} into {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="List tracker keys seen (5s)")
    ap.add_argument("--calib_rot_offset", action="store_true", help="Calibration tracker offset")
    ap.add_argument("--tracker-serial", type=str, default="LHR-0DFD738C", help="Substring to select tracker (serial/name)")
    ap.add_argument("--out", type=str, default="configs/T_W_LH.json")
    ap.add_argument("--still-speed", type=float, default=0.02, help="m/s threshold considered STILL")
    ap.add_argument("--ransac-thresh", type=float, default=0.01, help="RANSAC inlier threshold (m)")
    ap.add_argument("--ransac-iters", type=int, default=500)
    args = ap.parse_args()
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = args.out if os.path.isabs(args.out) else os.path.join(cur_dir, args.out)

    import pysurvive  # noqa

    stream = PoseStream(tracker_substr=args.tracker_serial)

    if args.list:
        print("Collecting tracker keys for ~5 seconds...")
        t_end = time.time() + 5.0
        while time.time() < t_end:
            time.sleep(0.05)
        stream.stop()
        print("Seen keys (key -> packets):")
        for k,v in sorted(stream.seen.items(), key=lambda x: -x[1])[:80]:
            print(f"  {k}: {v}")
        return

    floor_pts = []
    p0_samples = []
    p1_samples = []
    pz_samples = []
    T_W_LH = None; rot_offset = None
    ransac_inlier_mask = None

    # If a previous calibration exists, load it as the current extrinsic.
    if os.path.isfile(out_path):
        try:
            with open(out_path, "r") as f:
                loaded = json.load(f)
            loaded_T = np.asarray(loaded.get("T_W_LH"), dtype=float)
            loaded_rot_offset = np.asarray(loaded.get("rotation_offset"))
            if loaded_T.shape == (4, 4):
                T_W_LH = loaded_T
                print(f"Loaded existing T_W_LH from: {out_path}")
            else:
                print(f"[warn] {out_path} has invalid T_W_LH shape: {loaded_T.shape}, expected (4, 4)")
            if loaded_rot_offset.shape[-1] == 4:
                rot_offset = loaded_rot_offset
                print(f'Loaded existing rotation offset from: {out_path} ')
        except Exception as e:
            print(f"[warn] Failed to load existing extrinsic from {out_path}: {e}")

    plt.close("all")
    fig = plt.figure(figsize=(11,5))
    ax1 = fig.add_subplot(121, projection="3d")  # LH
    ax2 = fig.add_subplot(122, projection="3d")  # W

    def set_axes(ax, title):
        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_box_aspect([1,1,0.7])

    set_axes(ax1, "LH frame (libsurvive)")
    set_axes(ax2, "W frame (aligned to floor + P0/P1)")

    # live points
    lh_point, = ax1.plot([], [], [], marker="o", linestyle="", markersize=6)
    w_point,  = ax2.plot([], [], [], marker="o", linestyle="", markersize=6)

    # samples (LH)
    lh_floor_sc = ax1.scatter([], [], [], s=12)
    lh_out_sc   = ax1.scatter([], [], [], s=18, marker="x")  # RANSAC outliers
    lh_p0_sc    = ax1.scatter([], [], [], s=30, marker="x")
    lh_p1_sc    = ax1.scatter([], [], [], s=30, marker="x")
    lh_pz_sc    = ax1.scatter([], [], [], s=30, marker="^")

    # samples (W)
    w_floor_sc  = ax2.scatter([], [], [], s=12)
    w_out_sc    = ax2.scatter([], [], [], s=18, marker="x")
    w_p0_sc     = ax2.scatter([], [], [], s=30, marker="x")
    w_p1_sc     = ax2.scatter([], [], [], s=30, marker="x")
    w_pz_sc     = ax2.scatter([], [], [], s=30, marker="^")

    status_text = fig.text(0.02, 0.95, "", fontsize=11)
    fig.text(0.02, 0.02,
             "Keys: f=floor  0=P0  1=P1  z=Pz-above-floor  c=compute+save  r=reset  q/ESC=quit",
             fontsize=10)

    def refresh_samples():
        nonlocal ransac_inlier_mask

        # LH
        if len(floor_pts) > 0:
            P = np.vstack(floor_pts)
            # show all floor points first
            lh_floor_sc._offsets3d = (P[:,0], P[:,1], P[:,2])

            # if we have a mask, show outliers separately
            if ransac_inlier_mask is not None and len(ransac_inlier_mask)==len(floor_pts):
                out = P[~ransac_inlier_mask]
                lh_out_sc._offsets3d = (out[:,0], out[:,1], out[:,2]) if len(out)>0 else ([],[],[])
            else:
                lh_out_sc._offsets3d = ([],[],[])
        else:
            lh_floor_sc._offsets3d = ([], [], [])
            lh_out_sc._offsets3d   = ([], [], [])

        if len(p0_samples) > 0:
            P = np.vstack(p0_samples)
            lh_p0_sc._offsets3d = (P[:,0], P[:,1], P[:,2])
        else:
            lh_p0_sc._offsets3d = ([], [], [])

        if len(p1_samples) > 0:
            P = np.vstack(p1_samples)
            lh_p1_sc._offsets3d = (P[:,0], P[:,1], P[:,2])
        else:
            lh_p1_sc._offsets3d = ([], [], [])

        if len(pz_samples) > 0:
            P = np.vstack(pz_samples)
            lh_pz_sc._offsets3d = (P[:,0], P[:,1], P[:,2])
        else:
            lh_pz_sc._offsets3d = ([], [], [])

        # W
        if T_W_LH is not None:
            if len(floor_pts) > 0:
                P = np.vstack([apply_T(T_W_LH, p) for p in floor_pts])
                w_floor_sc._offsets3d = (P[:,0], P[:,1], P[:,2])
                if ransac_inlier_mask is not None and len(ransac_inlier_mask)==len(floor_pts):
                    out = P[~ransac_inlier_mask]
                    w_out_sc._offsets3d = (out[:,0], out[:,1], out[:,2]) if len(out)>0 else ([],[],[])
                else:
                    w_out_sc._offsets3d = ([],[],[])
            else:
                w_floor_sc._offsets3d = ([],[],[])
                w_out_sc._offsets3d   = ([],[],[])
            if len(p0_samples) > 0:
                P = np.vstack([apply_T(T_W_LH, p) for p in p0_samples])
                w_p0_sc._offsets3d = (P[:,0], P[:,1], P[:,2])
            else:
                w_p0_sc._offsets3d = ([],[],[])
            if len(p1_samples) > 0:
                P = np.vstack([apply_T(T_W_LH, p) for p in p1_samples])
                w_p1_sc._offsets3d = (P[:,0], P[:,1], P[:,2])
            else:
                w_p1_sc._offsets3d = ([],[],[])
            if len(pz_samples) > 0:
                P = np.vstack([apply_T(T_W_LH, p) for p in pz_samples])
                w_pz_sc._offsets3d = (P[:,0], P[:,1], P[:,2])
            else:
                w_pz_sc._offsets3d = ([],[],[])
        else:
            w_floor_sc._offsets3d = ([],[],[])
            w_out_sc._offsets3d   = ([],[],[])
            w_p0_sc._offsets3d    = ([],[],[])
            w_p1_sc._offsets3d    = ([],[],[])
            w_pz_sc._offsets3d    = ([],[],[])

    def on_key(event):
        nonlocal T_W_LH, floor_pts, p0_samples, p1_samples, pz_samples, ransac_inlier_mask
        key = event.key

        if key in ["q", "escape"]:
            stream.stop()
            plt.close(fig)
            return

        hist = stream.snapshot()
        if len(hist) < 5:
            print("No pose yet. Check tracking / --tracker-serial.")
            return

        p = robust_mean_pos(hist)
        if p is None:
            print("Pose not stable enough yet.")
            return

        if key == "f":
            floor_pts.append(p)
            print(f"floor[{len(floor_pts)}] = {p}")
            ransac_inlier_mask = None
        elif key == "0":
            p0_samples.append(p)
            print(f"P0[{len(p0_samples)}] = {p}")
        elif key == "1":
            p1_samples.append(p)
            print(f"P1[{len(p1_samples)}] = {p}")
        elif key == "z":
            pz_samples.append(p)
            print(f"Pz[{len(pz_samples)}] = {p}")
        elif key == "r":
            floor_pts, p0_samples, p1_samples, pz_samples = [], [], [], []
            T_W_LH = None
            ransac_inlier_mask = None
            print("Reset all samples and calibration.")
        elif key == "c":
            if len(floor_pts) < 12:
                print("Need >= 12 floor points (recommend 20-40).")
                return
            if len(p0_samples) < 10 or len(p1_samples) < 10:
                print("Need >= 10 samples each for P0 and P1 (recommend 20-50).")
                return
            if len(pz_samples) < 10:
                print("Need >= 10 samples for Pz above the floor plane (recommend 20-50).")
                return

            floor_arr = np.vstack(floor_pts)
            p0 = np.mean(np.vstack(p0_samples), axis=0)
            p1 = np.mean(np.vstack(p1_samples), axis=0)
            pz = np.mean(np.vstack(pz_samples), axis=0)

            # RANSAC filter
            n, d, inliers = ransac_plane(
                floor_arr,
                thresh=args.ransac_thresh,
                iters=args.ransac_iters,
                min_inliers=12,
                refine=True,
                seed=0
            )
            floor_in = floor_arr[inliers]
            ransac_inlier_mask = inliers

            # Build transform using inliers
            try:
                T_W_LH, n_ref = calibrate_LH_to_world_with_inliers(
                    floor_in,
                    p0,
                    p1,
                    pz,
                )
            except ValueError as e:
                print(f"[warn] Failed to determine +Z direction: {e}")
                return

            # quality metrics
            dist_in = np.abs(floor_in @ n + d)
            plane_rmse = float(np.sqrt(np.mean(dist_in**2)))

            p0_std = np.std(np.vstack(p0_samples), axis=0).tolist()
            p1_std = np.std(np.vstack(p1_samples), axis=0).tolist()
            pz_std = np.std(np.vstack(pz_samples), axis=0).tolist()

            out = {
                "T_W_LH": T_W_LH.tolist(),
                "floor_normal_LH": n_ref.tolist(),
                "plane_rmse_m_inliers": plane_rmse,
                "ransac_thresh_m": args.ransac_thresh,
                "ransac_inliers": int(inliers.sum()),
                "ransac_total": int(len(inliers)),
                "P0_mean_LH": p0.tolist(),
                "P1_mean_LH": p1.tolist(),
                "Pz_mean_LH": pz.tolist(),
                "P0_std_m": p0_std,
                "P1_std_m": p1_std,
                "Pz_std_m": pz_std,
                "tracker_key": hist[-1][3],
                "notes": "W: origin=P0 on floor; +X is P0->P1 projection on floor; +Z is the floor normal oriented toward the sampled Pz point above the floor."
            }
            
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)

            print(f"Saved {out_path}")
            print(f"RANSAC inliers: {int(inliers.sum())}/{len(inliers)}  thresh={args.ransac_thresh} m")
            print(f"plane_rmse_m(inliers)={plane_rmse:.6f}  (good: <0.003~0.005 if stable)")
            print(f"P0_std_m={p0_std}  P1_std_m={p1_std}  Pz_std_m={pz_std}  (good: each axis <0.002~0.005)")

        refresh_samples()

    fig.canvas.mpl_connect("key_press_event", on_key)
    num_cali_rot_offset = 0
    def update(_):
        hist = stream.snapshot()
        if len(hist) < 5:
            status_text.set_text("Waiting for pose... (check tracking / --tracker-serial)")
            return

        speed = compute_speed(hist, window_s=0.25)
        p = robust_mean_pos(hist, window_s=0.15)
        key = hist[-1][3]

        if args.calib_rot_offset:
            if len(hist) < 20: 
                status_text.set_text(f"Waiting for enough reading to get the rotation offset, num snapshot {len(hist)}")
                return
            
            if T_W_LH is None:
                print(f'The extrinsic parameter is not ready for calculating the rot offset!')
            else:
                status_text.set_text(f"Calculating the rotation offset, num snapshot {len(hist)}")
                rot_offset, _ = rot_offset_from_history(hist, T_W_LH, args.still_speed, window_s=0.5)
                save_R_tracker_tool_into_out(out_path, rot_offset)
                nonlocal num_cali_rot_offset
                num_cali_rot_offset += 1
                if num_cali_rot_offset > 25:
                    stream.stop()
                    plt.close(fig)
                    sys.exit(0)
            return

        if p is None or speed is None:
            status_text.set_text(f"Tracker={key} | pose unstable (need more data)")
            return

        still = speed < args.still_speed
        col = "g" if still else "r"

        # LH live point
        lh_point.set_data([p[0]], [p[1]])
        lh_point.set_3d_properties([p[2]])
        lh_point.set_color(col)

        # W live point
        error = 100
        if T_W_LH is not None:
            pw = apply_T(T_W_LH, p)
            # if at floor origin -> green else red
            error = np.linalg.norm(pw)
            col = "g" if error < 0.04 else "r"
            w_point.set_data([pw[0]], [pw[1]])
            w_point.set_3d_properties([pw[2]])
            w_point.set_color(col)
        else:
            w_point.set_data([], [])
            w_point.set_3d_properties([])

        inlier_info = ""
        if ransac_inlier_mask is not None:
            inlier_info = f" | RANSAC inliers {int(ransac_inlier_mask.sum())}/{len(ransac_inlier_mask)}"

        status_text.set_text(
            f"Tracker={key} | speed={speed:.4f} m/s | {'STILL' if still else 'MOVING'}"
            f" | floor={len(floor_pts)} P0={len(p0_samples)} P1={len(p1_samples)} Pz={len(pz_samples)}"
            f" | calibrated={'YES' if T_W_LH is not None else 'NO'}{inlier_info}"
            f" | With LH WORLD extrinsic tranform - floor world frame color representation: green -> near the world origin else red, error: {error}"
        )

        pts_lh = []
        pts_lh.extend(floor_pts)
        pts_lh.extend(p0_samples)
        pts_lh.extend(p1_samples)
        pts_lh.extend(pz_samples)
        pts_lh.append(p)
        autoscale(ax1, pts_lh, margin=0.2)

        if T_W_LH is not None:
            pts_w = [apply_T(T_W_LH, x) for x in pts_lh]
            autoscale(ax2, pts_w, margin=0.2)

    ani = FuncAnimation(fig, update, interval=50)  # ~20 Hz
    refresh_samples()
    plt.show()
    stream.stop()

if __name__ == "__main__":
    main()
