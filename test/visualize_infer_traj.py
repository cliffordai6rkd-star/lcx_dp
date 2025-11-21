import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # 激活 3D 支持


def _quat_xyzw_to_rotmat(q):
    """
    将四元数 (x, y, z, w) 转成 3x3 旋转矩阵.
    q: array-like, [qx, qy, qz, qw]
    """
    x, y, z, w = q
    # 归一化，防止数值误差
    norm = np.linalg.norm([w, x, y, z])
    if norm == 0:
        return np.eye(3)
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.array([
        [1 - 2 * (yy + zz),     2 * (xy - wz),         2 * (xz + wy)],
        [2 * (xy + wz),         1 - 2 * (xx + zz),     2 * (yz - wx)],
        [2 * (xz - wy),         2 * (yz + wx),         1 - 2 * (xx + yy)],
    ])
    return R


def _load_ee_pose_from_json(path):
    """
    从 JSON 文件中读取末端的位姿:
    返回:
        positions: (N, 3) numpy 数组
        quats:     (N, 4) numpy 数组, 四元数顺序为 (x, y, z, w)
    约定结构大致为:
    {
        "data": [
            {
                "ee_states": {
                    "single": {
                        "pose": [x, y, z, ..., qx, qy, qz, qw]
                    }
                }
            },
            ...
        ]
    }
    """
    with open(path, "r") as f:
        content = json.load(f)

    positions = []
    quats = []

    for step in content.get("data", []):
        ee_states = step.get("ee_states")
        if not isinstance(ee_states, dict):
            continue

        state = ee_states.get("single")
        # 如果没有 single，找第一个带 pose 的
        if not isinstance(state, dict):
            for v in ee_states.values():
                if isinstance(v, dict) and "pose" in v:
                    state = v
                    break

        if not isinstance(state, dict):
            continue

        pose = state.get("pose")
        if not (isinstance(pose, (list, tuple)) and len(pose) >= 7):
            continue

        pos = pose[:3]
        quat = pose[-4:]  # 后四位是 (x, y, z, w)
        positions.append(pos)
        quats.append(quat)

    if not positions:
        raise ValueError(f"No valid ee_states.pose found in file: {path}")

    return np.array(positions, dtype=float), np.array(quats, dtype=float)


def _draw_frames(ax, positions, quats,
                 axis_length=0.05,
                 stride=1):
    """
    在给定的点上画姿态坐标轴 (x, y, z 三个方向).
    positions: (N, 3)
    quats:     (N, 4) xyzw
    axis_length: 每个坐标轴长度
    stride: 每隔多少个点画一个坐标轴
    """
    N = len(positions)
    for i in range(0, N, stride):
        p = positions[i]
        q = quats[i]
        R = _quat_xyzw_to_rotmat(q)

        # 基坐标轴单位向量
        ex = R @ np.array([1.0, 0.0, 0.0]) * axis_length
        ey = R @ np.array([0.0, 1.0, 0.0]) * axis_length
        ez = R @ np.array([0.0, 0.0, 1.0]) * axis_length

        # x 轴 – 红色
        ax.quiver(p[0], p[1], p[2],
                  ex[0], ex[1], ex[2],
                  color="r", linewidth=0.8)
        # y 轴 – 绿色
        ax.quiver(p[0], p[1], p[2],
                  ey[0], ey[1], ey[2],
                  color="g", linewidth=0.8)
        # z 轴 – 蓝色
        ax.quiver(p[0], p[1], p[2],
                  ez[0], ez[1], ez[2],
                  color="b", linewidth=0.8)


def parse_files(files:str):
    if files.endswith("data.json"):
        pos, quat = _load_ee_pose_from_json(files)
    else: 
        pos = None; quat = None
        episode_dirs = os.listdir(files)
        for file in episode_dirs:
            cur_file = os.path.join(files, file, "data.json")
            print(f'load file {cur_file}')
            cur_pos, cur_quat = _load_ee_pose_from_json(cur_file)
            if pos is None:
                pos = cur_pos; quat = cur_quat
            else:
                pos = np.concatenate((pos, cur_pos), axis=0)
                quat = np.concatenate((quat, cur_quat), axis=0)
        print(f'pos: {pos.shape}')
    return pos, quat

def plot_ee_trajectories(
    training_files,
    infer_files,
    labels=("training_traj", "infer_traj2"),
    colors=("tab:blue", "tab:orange"),
    enable_rotation=False,
    axis_length=0.02,
    rotation_stride=20,
    show=True,
    save_path=None,
):
    """
    读取两个 JSON 轨迹文件，画出末端轨迹（位置）以及可选的姿态坐标轴.

    参数
    ----
    file1, file2 : str
        两个 JSON 文件路径
    labels : (str, str)
        图例名字
    colors : (str, str)
        两条轨迹的颜色
    enable_rotation : bool
        True  时会在轨迹点上画姿态坐标轴
        False 时只画轨迹
    axis_length : float
        姿态坐标轴的长度（单位与位置一致，通常是米）
    rotation_stride : int
        每隔多少个点画一个坐标轴（防止太挤）。如果想每个点都画，设为 1
    show : bool
        是否 `plt.show()`
    save_path : str | None
        不为 None 时，把图保存到该路径
    """
    pos1, quat1 = parse_files(training_files)
    pos2, quat2 = parse_files(infer_files)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # 画轨迹线
    ax.plot(pos1[:, 0], pos1[:, 1], pos1[:, 2],
            color=colors[0], label=labels[0])
    ax.plot(pos2[:, 0], pos2[:, 1], pos2[:, 2],
            color=colors[1], label=labels[1])

    # 起点/终点标记
    ax.scatter(*pos1[0], color=colors[0], marker="o", s=30)
    ax.scatter(*pos1[-1], color=colors[0], marker="x", s=40)
    ax.scatter(*pos2[0], color=colors[1], marker="o", s=30)
    ax.scatter(*pos2[-1], color=colors[1], marker="x", s=40)

    # 可选：画姿态坐标轴
    if enable_rotation:
        _draw_frames(ax, pos1, quat1,
                     axis_length=axis_length,
                     stride=rotation_stride)
        _draw_frames(ax, pos2, quat2,
                     axis_length=axis_length,
                     stride=rotation_stride)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("End-Effector Trajectories with Orientation")
    ax.legend()

    # 设置三个轴比例一致
    xs = np.concatenate([pos1[:, 0], pos2[:, 0]])
    ys = np.concatenate([pos1[:, 1], pos2[:, 1]])
    zs = np.concatenate([pos1[:, 2], pos2[:, 2]])

    max_range = max(xs.max() - xs.min(),
                    ys.max() - ys.min(),
                    zs.max() - zs.min()) / 2.0
    mid_x = (xs.max() + xs.min()) / 2.0
    mid_y = (ys.max() + ys.min()) / 2.0
    mid_z = (zs.max() + zs.min()) / 2.0

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    if show:
        plt.show()

    return fig, ax

if __name__ == "__main__":
    import os, random
    training_data = "/workspace/dataset/data/1114_left_fr3_insert_pinboard_Generalize_82ep"
    infer_data = "/workspace/dataset/data/infer/insert_pinboard"
    # episode_dirs = os.listdir(training_data)
    # training_episode = random.randint(0, len(episode_dirs)-1)
    # training_episode = episode_dirs[training_episode]
    # print(f'training episode: {training_episode}')
    # training_data = os.path.join(training_data, training_episode, "data.json")
    episode_dirs = os.listdir(infer_data)
    infer_episode = random.randint(0, len(episode_dirs)-1)
    infer_episode = episode_dirs[infer_episode]
    print(f'infer episode: {infer_episode}')
    infer_data = os.path.join(infer_data, infer_episode, "data.json")
    
    plot_ee_trajectories(training_data, infer_data, enable_rotation=False,
                         labels=(f"training_trajs", 
                                 f"infer_traj_{infer_episode}"))
    
    # def serialize_data(data):
    #     if data is None:
    #         return None
    #     print(f'serial calling')
    #     if isinstance(data, dict):
    #         for key, value in data.items():
    #             data[key] = serialize_data(value)
    #     else:
    #         single_number_type = [int, float, bool, complex]
    #         check = [isinstance(data, cur_type) for cur_type in single_number_type]
    #         print(f'check {check}')
    #         if not any(check):
    #             print(f'pass the check for {data}: {data.ndim}')
    #             if isinstance(data, np.ndarray) and data.ndim:
    #                 print(f'pass mp test')
    #                 data = data.tolist()
    #                 print(f'to list: {type(data)}')
    #     return data
    
    # data = dict(a=np.ones(5), b=np.zeros(3))
    # data = serialize_data(data)
    # print(f"data: {data}")
    