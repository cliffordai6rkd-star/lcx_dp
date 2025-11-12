import random
from typing import Optional
import numpy as np

class LowPassFilter:
    """
    [IIR, Exponential Moving Average）。

    Equation:
        y[n] = y[n-1] + α * (x[n] - y[n-1])
    and:
        α = 1 - exp(-dt / τ)
        τ = 1 / (2π f_c)

    Parameters
    ----
    cutoff_hz : float
    dt : Optional[float] (sample time)
    initial : Optional[float] initial value if None intial is the first value during calling
    """

    def __init__(self, cutoff_hz: float, dt: Optional[float] = None, initial: Optional[np.ndarray] = None):
        if cutoff_hz <= 0:
            raise ValueError("cutoff_hz must be > 0")
        if dt is not None and dt <= 0:
            raise ValueError("dt must be > 0 when provided")

        self.cutoff_hz = float(cutoff_hz)
        self._dt_fixed = float(dt) if dt is not None else None
        self._y_prev = np.array(initial) if initial is not None else None

    def update(self, x, dt: Optional[float] = None) -> np.ndarray:
        """Get the filtered output"""
        _dt = self._resolve_dt(dt)
        x_arr = np.array(x)

        if self._y_prev is None:
            self._y_prev = x_arr.copy()
            return x_arr.copy()
        
        assert x_arr.shape == self._y_prev.shape, f"x shape {x_arr.shape} not match y prev shape: {self._y_prev.shape}"

        alpha = self._alpha(_dt)
        self._y_prev = self._y_prev + alpha * (x_arr - self._y_prev)
        return self._y_prev.copy()

    def apply(self, xs, dt: float) -> np.ndarray:
        """对序列做离线滤波。xs 可是 (T,D,...) 形状的 ndarray，或可迭代的向量列表。"""
        if dt <= 0:
            raise ValueError("dt must be > 0")

        # 支持直接传入 2D/ND 时间批：shape (T, *feat_shape)
        if isinstance(xs, np.ndarray) and xs.ndim >= 2:
            T = xs.shape[0]
            outs = []
            # 暂存并设置固定 dt
            old_dt = self._dt_fixed
            try:
                self._dt_fixed = dt
                for t in range(T):
                    outs.append(self.update(xs[t]))
            finally:
                self._dt_fixed = old_dt
            return np.stack(outs, axis=0)

        # 一般迭代器路径
        outs = []
        old_dt = self._dt_fixed
        try:
            self._dt_fixed = dt
            for x in xs:
                outs.append(self.update(x))
        finally:
            self._dt_fixed = old_dt
        return np.stack(outs, axis=0)

    def reset(self, value = None) -> None:
        if self._y_prev:
            assert len(self._y_prev) == len(value), f"len y prev is not equal to value during reset"
        self._y_prev = np.array(value) if value is not None else None

    def set_cutoff(self, cutoff_hz: float) -> None:
        if cutoff_hz <= 0:
            raise ValueError("cutoff_hz must be > 0")
        self.cutoff_hz = float(cutoff_hz)

    # ------------ helpers ------------
    def _alpha(self, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * self.cutoff_hz)
        a = 1.0 - np.exp(-dt / tau)
        return 0.0 if a < 0.0 else 1.0 if a > 1.0 else a

    def _resolve_dt(self, dt: Optional[float]) -> float:
        if dt is None:
            if self._dt_fixed is None:
                raise ValueError("dt not provided and no fixed dt set")
            return self._dt_fixed
        if dt <= 0:
            raise ValueError("dt must be > 0")
        return dt

    # for printing
    def __repr__(self) -> str:
        dt = self._dt_fixed if self._dt_fixed is not None else "variable"
        shape = None if self._y_prev is None else tuple(self._y_prev.shape)
        return f"LowPassFilter(cutoff_hz={self.cutoff_hz}, dt={dt}, shape={shape})"

def main():
    # 假设 3 轴传感器（ax, ay, az）
    fs = 200.0
    dt = 1.0 / fs
    T = 1000

    # 构造 (T, 3) 的含噪信号
    t = np.arange(T) * dt
    clean = np.stack(
        [np.sin(2*np.pi*1.5*t), np.cos(2*np.pi*0.8*t), 0.5*np.sin(2*np.pi*0.4*t)],
        axis=1,
    )  # (T,3)
    noise = 0.3 * np.random.randn(T, 3)
    xs = clean + noise

    # 方式 A：离线批量（直接传 (T,3) 数组）
    lpf = LowPassFilter(cutoff_hz=3.0)
    ys = lpf.apply(xs, dt=dt)  # (T,3)
    print("offline out shape:", ys.shape, "first row:", np.round(ys[0], 3))

    # 方式 B：流式（list 或 ndarray 都可）
    lpf2 = LowPassFilter(cutoff_hz=3.0, dt=dt, auto_reset_on_shape_change=True)
    y_last = None
    for i in range(T):
        # 这里传 list，同样生效
        y_last = lpf2.update(xs[i].tolist())
    print("stream last:", np.round(y_last, 3))

if __name__ == "__main__":
    main()