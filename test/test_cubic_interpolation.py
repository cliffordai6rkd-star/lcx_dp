import numpy as np

def make_minjerk(q0: np.ndarray, qT: np.ndarray, T: float):
    q0 = np.asarray(q0, dtype=float).reshape(-1)
    qT = np.asarray(qT, dtype=float).reshape(-1)
    if q0.shape != qT.shape:
        raise ValueError("q0 and qT must have same shape")
    if T <= 0:
        raise ValueError("T must be > 0")

    dq = (qT - q0)

    def _u(t):
        t = np.asarray(t, dtype=float)
        return np.clip(t / T, 0.0, 1.0)

    def q(t):
        u = _u(t)
        s = 10*u**3 - 15*u**4 + 6*u**5
        return q0 + dq * s[..., None] if np.ndim(u) else q0 + dq * s

    return q

q0 = np.array([1.5, -2.2, 3.3])
qt = np.zeros(3)
T  = 1.5
dt = 0.02
q_fun = make_minjerk(q0, qt, T)

t = np.arange(0.0, T + 1e-9, dt)
# q_traj = q_fun(t)
q_traj = q_fun(0.0)
for cur_t in t[1:]:
    q_traj = np.vstack((q_traj, q_fun(cur_t)))
print(f'q_traj shape: {q_traj.shape}')
dof = q0.size

import matplotlib.pyplot as plt
import math

ncols = int(math.ceil(math.sqrt(dof)))
nrows = int(math.ceil(dof / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 2.8*nrows), sharex=True)
axes = np.atleast_1d(axes).ravel()
for j in range(dof):
    ax = axes[j]
    ax.plot(t, q_traj[:, j])
    ax.set_title(f"joint {j}")
    ax.set_ylabel("q (rad)")
    ax.grid(True)

# 多余的 subplot 隐藏掉
for k in range(dof, len(axes)):
    axes[k].axis("off")

axes[0].set_xlabel("t (s)")
for ax in axes[1:]:
    ax.set_xlabel("t (s)")

fig.suptitle("Min-jerk joint trajectories", y=1.02, fontsize=14)
fig.tight_layout()
plt.show()
