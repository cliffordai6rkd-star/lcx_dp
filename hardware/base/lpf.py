
import math
import random
from typing import Iterable, List, Optional


class LowPassFilter:
    """
    （IIR, Exponential Moving Average）。

    Equation:
        y[n] = y[n-1] + α * (x[n] - y[n-1])
    and：
        α = 1 - exp(-dt / τ)
        τ = 1 / (2π f_c)

    Parameters
    ----
    cutoff_hz : float
    dt : Optional[float] (sample time)
    initial : Optional[float] initial value if None intial is the first value during calling
    """

    def __init__(self, cutoff_hz: float, dt: Optional[float] = None, initial: Optional[float] = None):
        if cutoff_hz <= 0:
            raise ValueError("cutoff_hz must be > 0")
        if dt is not None and dt <= 0:
            raise ValueError("dt must be > 0 when provided")

        self.cutoff_hz = float(cutoff_hz)
        self._dt_fixed = float(dt) if dt is not None else None
        self._y_prev: Optional[float] = float(initial) if initial is not None else None

    # ---- public API ----
    def update(self, x: float, dt: Optional[float] = None) -> float:
        """输入一个新样本，返回滤波输出。dt 不传则使用构造时的固定 dt。"""
        _dt = self._resolve_dt(dt)
        if self._y_prev is None:
            self._y_prev = float(x)
            return self._y_prev

        alpha = self._alpha(_dt)
        y = self._y_prev + alpha * (float(x) - self._y_prev)
        self._y_prev = y
        return y

    def apply(self, xs: Iterable[float], dt: float) -> List[float]:
        """对一个序列做离线滤波（必须提供固定 dt）。"""
        if dt <= 0:
            raise ValueError("dt must be > 0")
        out = []
        # 临时保存、使用完恢复
        old_fixed_dt = self._dt_fixed
        try:
            self._dt_fixed = dt
            for x in xs:
                out.append(self.update(x))
        finally:
            self._dt_fixed = old_fixed_dt
        return out

    def reset(self, value: Optional[float] = None) -> None:
        """重置内部状态；可选地设置新的初值。"""
        self._y_prev = float(value) if value is not None else None

    def set_cutoff(self, cutoff_hz: float) -> None:
        """动态修改截止频率。"""
        if cutoff_hz <= 0:
            raise ValueError("cutoff_hz must be > 0")
        self.cutoff_hz = float(cutoff_hz)

    # ---- helpers ----
    def _alpha(self, dt: float) -> float:
        """根据 dt 和当前截止频率计算平滑系数 α（0~1）。用 exp 形式，数值更准确。"""
        tau = 1.0 / (2.0 * math.pi * self.cutoff_hz)  # 时间常数 τ
        # α = 1 - e^{-dt/τ}
        a = 1.0 - math.exp(-dt / tau)
        # 数值保护，确保在 [0,1] 范围
        if a < 0.0:
            return 0.0
        if a > 1.0:
            return 1.0
        return a

    def _resolve_dt(self, dt: Optional[float]) -> float:
        if dt is None:
            if self._dt_fixed is None:
                raise ValueError("dt is not provided and no fixed dt set in constructor")
            return self._dt_fixed
        if dt <= 0:
            raise ValueError("dt must be > 0")
        return dt

    def __repr__(self) -> str:
        dt = self._dt_fixed if self._dt_fixed is not None else "variable"
        return f"LowPassFilter(cutoff_hz={self.cutoff_hz}, dt={dt}, y_prev={self._y_prev})"


# ------------------------- 使用演示 -------------------------
def main():
    # 示例 1：离线数据（固定采样率）——对含噪正弦信号做低通
    fs = 100.0                    # 采样率 100 Hz
    dt = 1.0 / fs
    t_end = 5.0                   # 5 秒
    n = int(t_end * fs)

    # 生成一个 1 Hz 的正弦+高斯噪声（不依赖 numpy）
    xs = []
    for i in range(n):
        t = i * dt
        clean = math.sin(2 * math.pi * 1.0 * t)  # 1 Hz 正弦
        noise = 0.4 * random.gauss(0.0, 1.0)
        xs.append(clean + noise)

    lpf = LowPassFilter(cutoff_hz=2.0)  # 截止频率 2 Hz（>1 Hz，能保留主体并抑制高频噪声）
    ys = lpf.apply(xs, dt=dt)

    print("示例1：离线滤波完成。前10个输出：")
    print([round(v, 4) for v in ys[:10]])

    # 可选：画图（若未安装 matplotlib，会自动跳过）
    try:
        import matplotlib.pyplot as plt
        ts = [i * dt for i in range(n)]
        plt.figure()
        plt.plot(ts, xs, label="noisy input")
        plt.plot(ts, ys, label="low-pass output")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("First-order Low-pass Filter (fc=2 Hz, fs=100 Hz)")
        plt.legend()
        plt.show()
    except Exception as e:
        # 没装 matplotlib 也没关系
        pass

    # 示例 2：流式、变采样周期（比如实时循环里 dt 不稳定）
    import time
    lpf_stream = LowPassFilter(cutoff_hz=3.0, initial=0.0)  # 初值给 0
    print("\n示例2：流式变 dt（运行 1 秒左右，打印若干步结果）")
    last = time.perf_counter()
    for k in range(20):
        now = time.perf_counter()
        dt_k = max(1e-6, now - last)  # 当前步的 dt
        last = now

        # 伪造一个输入：慢变目标 + 抖动噪声
        t = k * 0.05
        x = 0.5 * math.sin(2 * math.pi * 0.5 * t) + 0.2 * random.gauss(0.0, 1.0)
        y = lpf_stream.update(x, dt=dt_k)
        print(f"step={k:02d} dt={dt_k*1000:6.2f} ms  x={x:+.3f}  y={y:+.3f}")
        time.sleep(0.05)  # 模拟你的实时循环工作负载

    # 示例 3：动态修改截止频率
    lpf_stream.set_cutoff(6.0)  # 提高截止频率以减少滞后
    print("\n示例3：已将流式滤波器截止频率改为 6.0 Hz")


if __name__ == "__main__":
    main()