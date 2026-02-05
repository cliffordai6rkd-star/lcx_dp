"""
Pose low-pass filter for 7D pose: [x, y, z, qx, qy, qz, qw].

Translation uses exponential smoothing.
Rotation uses quaternion slerp (with sign continuity).
"""

from __future__ import annotations
from typing import Optional
import numpy as np

class PoseLpf:
    """
    First-order low-pass filter for 7D pose [x, y, z, qx, qy, qz, qw].

    Args:
        cutoff_hz: Cutoff frequency for 1st-order LPF (Hz). Ignored if alpha is set.
        sample_freq: Sampling frequency (Hz) used when dt is not provided.
        alpha: Direct smoothing factor in [0, 1]. If provided, cutoff_hz is ignored.
        initial_pose: Optional initial pose to prime the filter.
        use_slerp: If True, use slerp for rotation. If False, use lerp+normalize.
    """

    def __init__(
        self,
        cutoff_hz: float = 5.0,
        sample_freq: Optional[float] = None,
        alpha: Optional[float] = None,
        initial_pose: Optional[np.ndarray] = None,
        use_slerp: bool = True,
    ) -> None:
        if alpha is None:
            if cutoff_hz is None or cutoff_hz <= 0:
                raise ValueError("cutoff_hz must be > 0 when alpha is not set")
        else:
            alpha = self._clip_alpha(alpha)

        if sample_freq is not None and sample_freq <= 0:
            raise ValueError("sample_freq must be > 0")

        self._cutoff_hz = cutoff_hz
        self._sample_freq = sample_freq
        self._alpha = alpha
        self._use_slerp = use_slerp
        self._pose: Optional[np.ndarray] = None
        # Jump suppression: clamp per-step motion by max speeds
        self._max_translation_speed = 1.0  # m/s
        self._max_rotation_speed = np.deg2rad(120.0)  # rad/s

        if initial_pose is not None:
            self.reset(initial_pose)

    def reset(self, pose: np.ndarray) -> None:
        """Reset filter state to the given pose."""
        pose = self._validate_pose(pose)
        self._pose = pose

    def get(self) -> Optional[np.ndarray]:
        """Get current filtered pose. Returns None if uninitialized."""
        if self._pose is None:
            return None
        return self._pose.copy()

    def update(self, pose: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        """
        Update filter with a new pose measurement.

        Args:
            pose: New pose measurement [x, y, z, qx, qy, qz, qw].
            dt: Optional timestep (s). If None, sample_freq is used.

        Returns:
            Filtered pose as numpy array with shape (7,).
        """
        pose = self._validate_pose(pose)

        if self._pose is None:
            self._pose = pose
            return self._pose.copy()

        dt_used: Optional[float] = None
        if dt is not None or self._sample_freq is not None:
            dt_used = self._resolve_dt(dt)

        alpha = self._compute_alpha(dt)

        # Translation
        t_prev = self._pose[:3]
        t_new = pose[:3]
        if dt_used is not None:
            t_new = self._clamp_translation(t_prev, t_new, dt_used)
        t_filt = (1.0 - alpha) * t_prev + alpha * t_new

        # Rotation
        q_prev = self._pose[3:]
        q_new = pose[3:]
        q_new = self._ensure_quaternion_continuity(q_new, q_prev)
        if dt_used is not None:
            q_new = self._clamp_rotation(q_prev, q_new, dt_used)

        if self._use_slerp:
            q_filt = self._slerp(q_prev, q_new, alpha)
        else:
            q_filt = self._normalize_quaternion((1.0 - alpha) * q_prev + alpha * q_new)

        self._pose = np.concatenate([t_filt, q_filt])
        return self._pose.copy()

    def set_cutoff(self, cutoff_hz: float) -> None:
        """Update cutoff frequency (Hz)."""
        if cutoff_hz <= 0:
            raise ValueError("cutoff_hz must be > 0")
        self._cutoff_hz = cutoff_hz
        self._alpha = None

    def set_alpha(self, alpha: float) -> None:
        """Set smoothing factor directly (0-1)."""
        self._alpha = self._clip_alpha(alpha)

    def _compute_alpha(self, dt: Optional[float]) -> float:
        if self._alpha is not None:
            return self._alpha

        dt = self._resolve_dt(dt)

        tau = 1.0 / (2.0 * np.pi * self._cutoff_hz)
        alpha = dt / (tau + dt)
        return self._clip_alpha(alpha)

    @staticmethod
    def _clip_alpha(alpha: float) -> float:
        return float(np.clip(alpha, 0.0, 1.0))

    def _resolve_dt(self, dt: Optional[float]) -> float:
        if dt is None:
            if self._sample_freq is None:
                raise ValueError("dt or sample_freq must be provided")
            dt = 1.0 / self._sample_freq
        dt = float(dt)
        if dt <= 0:
            raise ValueError("dt must be > 0")
        return dt

    @staticmethod
    def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(q)
        if n < 1e-12:
            raise ValueError("Quaternion norm too small")
        return q / n

    @classmethod
    def _validate_pose(cls, pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose, dtype=float).reshape(-1)
        if pose.size != 7:
            raise ValueError(f"Pose must have 7 elements, got {pose.size}")

        t = pose[:3].copy()
        q = cls._normalize_quaternion(pose[3:].copy())
        return np.concatenate([t, q])

    @staticmethod
    def _ensure_quaternion_continuity(q: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
        if np.dot(q, q_ref) < 0.0:
            return -q
        return q

    def _clamp_translation(
        self, t_prev: np.ndarray, t_new: np.ndarray, dt: float
    ) -> np.ndarray:
        if self._max_translation_speed <= 0:
            return t_new
        delta = t_new - t_prev
        dist = float(np.linalg.norm(delta))
        max_step = self._max_translation_speed * dt
        if dist > max_step and dist > 1e-12:
            return t_prev + delta / dist * max_step
        return t_new

    def _clamp_rotation(
        self, q_prev: np.ndarray, q_new: np.ndarray, dt: float
    ) -> np.ndarray:
        if self._max_rotation_speed <= 0:
            return q_new
        q_prev = self._normalize_quaternion(q_prev)
        q_new = self._normalize_quaternion(q_new)
        dot = float(np.clip(np.dot(q_prev, q_new), -1.0, 1.0))
        angle = 2.0 * np.arccos(abs(dot))
        max_step = self._max_rotation_speed * dt
        if angle > max_step and max_step > 0:
            ratio = max_step / angle
            return self._slerp(q_prev, q_new, ratio)
        return q_new

    @classmethod
    def _slerp(cls, q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
        q0 = cls._normalize_quaternion(q0)
        q1 = cls._normalize_quaternion(q1)

        dot = np.dot(q0, q1)
        dot = float(np.clip(dot, -1.0, 1.0))

        # If very close, use lerp to avoid numerical issues
        if dot > 0.9995:
            return cls._normalize_quaternion((1.0 - t) * q0 + t * q1)

        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        sin_theta = np.sin(theta)

        s0 = np.sin(theta_0 - theta) / sin_theta_0
        s1 = sin_theta / sin_theta_0
        return cls._normalize_quaternion(s0 * q0 + s1 * q1)

if __name__ == "__main__":
    def _euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)

        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        qw = cr * cp * cy + sr * sp * sy
        return np.array([qx, qy, qz, qw], dtype=float)


    def _quat_to_yaw(q: np.ndarray) -> float:
        qx, qy, qz, qw = q
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return float(np.arctan2(siny_cosp, cosy_cosp))
    
    
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(
            "matplotlib is required for visualization. "
            "Please install it to run this demo."
        ) from exc

    np.random.seed(7)

    dt = 1.0 / 30.0
    t = np.arange(0.0, 10.0, dt)
    n = t.size

    # Ground-truth translation
    pos_gt = np.stack(
        [
            0.3 * np.sin(0.6 * t),
            0.2 * np.cos(0.4 * t),
            0.1 * np.sin(0.9 * t),
        ],
        axis=1,
    )

    # Ground-truth rotation (slowly varying)
    roll_gt = 0.2 * np.sin(0.3 * t)
    pitch_gt = 0.15 * np.sin(0.2 * t + 0.5)
    yaw_gt = 0.5 * np.sin(0.25 * t)
    quat_gt = np.stack(
        [_euler_to_quat(r, p, y) for r, p, y in zip(roll_gt, pitch_gt, yaw_gt)],
        axis=0,
    )

    # Add measurement noise
    pos_noise = 0.01 * np.random.randn(n, 3)
    rot_noise = 0.03 * np.random.randn(n, 3)
    roll_meas = roll_gt + rot_noise[:, 0]
    pitch_meas = pitch_gt + rot_noise[:, 1]
    yaw_meas = yaw_gt + rot_noise[:, 2]

    pos_meas = pos_gt + pos_noise

    # Inject jumpy outliers to simulate ArUco glitches
    jump_idx = (t > 4.0) & (t < 4.3)
    pos_meas[jump_idx] += np.array([0.5, -0.4, 0.3])
    yaw_meas[jump_idx] += 1.2

    quat_meas = np.stack(
        [_euler_to_quat(r, p, y) for r, p, y in zip(roll_meas, pitch_meas, yaw_meas)],
        axis=0,
    )

    pose_meas = np.hstack([pos_meas, quat_meas])

    lpf = PoseLpf(cutoff_hz=3.0, sample_freq=50.0, use_slerp=True)
    pose_filt = np.zeros_like(pose_meas)
    for i in range(n):
        pose_filt[i] = lpf.update(pose_meas[i], dt=dt)

    yaw_meas_series = np.array([_quat_to_yaw(q) for q in quat_meas])
    yaw_filt_series = np.array([_quat_to_yaw(q) for q in pose_filt[:, 3:]])

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    labels = ["x", "y", "z"]
    for i in range(3):
        axes[i].plot(t, pos_meas[:, i], color="tab:orange", alpha=0.4, label="meas")
        axes[i].plot(t, pose_filt[:, i], color="tab:blue", label="filtered")
        axes[i].plot(t, pos_gt[:, i], color="tab:green", linewidth=1.0, label="gt")
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True, alpha=0.3)

    axes[3].plot(t, yaw_meas_series, color="tab:orange", alpha=0.4, label="meas")
    axes[3].plot(t, yaw_filt_series, color="tab:blue", label="filtered")
    axes[3].plot(t, yaw_gt, color="tab:green", linewidth=1.0, label="gt")
    axes[3].set_ylabel("yaw")
    axes[3].set_xlabel("time (s)")
    axes[3].grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    plt.show()
