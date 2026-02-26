# calibration_world_frame.py 使用说明

本文档说明如何使用 `teleop/pika_tracker/calibration_world_frame.py` 完成世界坐标系标定，并保存 `T_W_LH` 变换矩阵。

## 1. 脚本作用

该脚本会基于手动采样构建世界坐标系 `W`，定义如下：

- 原点：`P0`
- `+Z`：地面法向（由地面采样点 + RANSAC 拟合得到）
- `+X`：`P0 -> P1` 在地面上的投影方向

输出是 `T_W_LH`（把 LH/libsurvive 坐标系下的点变换到世界坐标系）。

## 2. 前置条件

- Lighthouse 与 tracker 已正常工作（`pysurvive` 能读到位姿）。
- 已安装 Python 依赖：
  - `numpy`
  - `matplotlib`
  - `pysurvive`
- 运行时有图形界面（脚本使用 Matplotlib 交互窗口接收按键）。

## 3. 参数说明

```bash
python teleop/pika_tracker/calibration_world_frame.py --help
```

常用参数：

- `--list`：先跑 5 秒，列出当前可见 tracker key（序列号/名称）。
- `--tracker-serial`：按子串过滤 tracker key，避免多设备串扰。
- `--out`：输出 JSON 路径。
- `--still-speed`：静止速度阈值，默认 `0.02` m/s。
- `--ransac-thresh`：地面平面 RANSAC 内点阈值，默认 `0.01` m。
- `--ransac-iters`：RANSAC 迭代次数，默认 `500`。

## 4. 标定流程（推荐）

### 第一步：确认 tracker key

```bash
python teleop/pika_tracker/calibration_world_frame.py --list
```

记下你要用的 tracker key（比如序列号子串）。

### 第二步：启动交互标定

```bash
python teleop/pika_tracker/calibration_world_frame.py \
  --tracker-serial <你的tracker子串> \
  --out teleop/pika_tracker/config/T_W_LH.json
```

窗口打开后，按键如下：

- `f`：记录当前点为地面点
- `0`：记录当前点为 `P0` 采样
- `1`：记录当前点为 `P1` 采样
- `c`：计算并保存标定
- `r`：清空所有采样并重来
- `q` 或 `ESC`：退出

### 第三步：采样建议

- 地面点（`f`）：至少 `12` 个，建议 `20-40` 个，尽量分布在较大区域。
- `P0`（`0`）：至少 `10` 个，建议 `20-50` 个。
- `P1`（`1`）：至少 `10` 个，建议 `20-50` 个。

操作要点：

- 采样时尽量保持 tracker 静止，状态栏显示 `STILL` 再按键。
- `P0` 和 `P1` 需要有足够间距（建议 `>= 20 cm`），且在地面平面上定义方向更稳定。
- `P1` 选在你希望世界坐标 `+X` 指向的位置。

### 第四步：计算与保存

按 `c` 后，终端会打印质量指标并写入 JSON。

主要关注：

- `RANSAC inliers`: 内点比例越高越好。
- `plane_rmse_m(inliers)`: 建议小于 `0.003 ~ 0.005` m。
- `P0_std_m` / `P1_std_m`: 每轴建议小于 `0.002 ~ 0.005` m。

## 5. 输出文件说明

输出 JSON 典型字段：

- `T_W_LH`: 4x4 齐次变换矩阵
- `floor_normal_LH`: LH 坐标系中的地面法向
- `plane_rmse_m_inliers`: 内点平面拟合误差
- `P0_mean_LH` / `P1_mean_LH`: 两个参考点均值
- `P0_std_m` / `P1_std_m`: 参考点采样标准差
- `tracker_key`: 本次使用的 tracker 标识

## 6. 常见问题

### 1) 窗口提示 `No pose yet` 或 `Waiting for pose...`

- 检查 Lighthouse 与 tracker 是否已被 `pysurvive` 识别。
- 先用 `--list` 确认 key，再设置 `--tracker-serial`。
- 若过滤字符串写错，会导致无数据。

### 2) `RANSAC failed`

- 地面点数量不足或离群点太多。
- 增加地面采样点并重采（建议 20+）。
- 适当放宽 `--ransac-thresh`（例如从 `0.01` 到 `0.012~0.015`）。

### 3) 标定后坐标方向不符合预期

- `+X` 方向由 `P0 -> P1` 决定，重新选择 `P1` 并重标定。
- 确保 `P0` 和 `P1` 不是几乎重合。

## 7. 最小可复现命令

```bash
# 1) 查 tracker key
python teleop/pika_tracker/calibration_world_frame.py --list

# 2) 标定并保存
python teleop/pika_tracker/calibration_world_frame.py \
  --tracker-serial WM0 \
  --out teleop/pika_tracker/config/T_W_LH.json
```
