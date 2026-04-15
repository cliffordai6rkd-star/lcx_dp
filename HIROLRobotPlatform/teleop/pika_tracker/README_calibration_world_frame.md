# `calibration_world_frame.py` 使用说明

本文档对应当前版本的 [`teleop/pika_tracker/calibration_world_frame.py`](./calibration_world_frame.py)。

该脚本有两个用途：

1. 标定 `T_W_LH`：建立世界坐标系 `W` 相对于 Lighthouse/libsurvive 坐标系 `LH` 的变换。
2. 标定 `rotation_offset`：在已有 `T_W_LH` 的前提下，估计 tracker 的零位姿旋转补偿四元数。

## 1. 世界坐标系定义

脚本构建的世界坐标系 `W` 定义如下：

- 原点：`P0`
- `+X`：`P0 -> P1` 在地面平面上的投影方向
- `+Z`：地面法向，且朝向由 `Pz` 采样点所在的一侧决定

这里的 `Pz` 不是原点，而是“位于地面上方的一组采样点”，用于消除平面法向的正负号二义性，确保 `+Z` 指向物理意义上的“向上”。

最终输出的 `T_W_LH` 用于把 `LH` 坐标系下的点变换到世界坐标系：

```text
p_W = T_W_LH * p_LH
```

## 2. 脚本能力概览

启动交互窗口后，左图显示 `LH` 坐标系，右图显示变换到 `W` 后的结果。

脚本支持：

- 实时显示 tracker 位置
- 用键盘采样地面点、`P0`、`P1`、`Pz`
- 对地面点做 RANSAC 平面拟合
- 计算并保存 `T_W_LH`
- 从已有标定文件继续加载 `T_W_LH` / `rotation_offset`
- 单独运行旋转零位补偿标定 `--calib_rot_offset`

## 3. 前置条件

需要满足：

- Lighthouse 与 tracker 已经正常工作
- `pysurvive` 能稳定读到目标 tracker 位姿
- 运行环境有图形界面，Matplotlib 窗口可以正常弹出并接收按键
- Python 依赖已安装：
  - `numpy`
  - `matplotlib`
  - `scipy`
  - `pysurvive`

## 4. 命令行参数

查看完整参数：

```bash
python teleop/pika_tracker/calibration_world_frame.py --help
```

当前脚本的主要参数如下：

- `--list`
  - 运行约 5 秒，列出当前看到的 tracker key / serial
- `--calib_rot_offset`
  - 进入旋转补偿标定模式，只估计并保存 `rotation_offset`
- `--tracker-serial`
  - tracker key 的子串过滤，默认是 `LHR-0DFD738C`
- `--out`
  - 输出 JSON 路径
  - 如果传入相对路径，会按脚本所在目录解析，不是按当前 shell 目录解析
- `--still-speed`
  - 静止判定阈值，默认 `0.02` m/s
- `--ransac-thresh`
  - 地面平面 RANSAC 内点阈值，默认 `0.01` m
- `--ransac-iters`
  - RANSAC 迭代次数，默认 `500`

## 5. 推荐标定流程

### 5.1 先确认 tracker key

```bash
python teleop/pika_tracker/calibration_world_frame.py --list
```

终端会输出类似：

```text
Seen keys (key -> packets):
  LHR-0DFD738C: 153
```

记录你要使用的 tracker serial 或其唯一子串。

### 5.2 运行世界坐标系标定

建议命令：

```bash
python teleop/pika_tracker/calibration_world_frame.py \
  --tracker-serial LHR-0DFD738C \
  --out config/T_W_LH.json
```

如果你在本仓库根目录运行，上面的输出实际会写到：

- `teleop/pika_tracker/config/T_W_LH.json`

因为脚本会把相对路径拼到 `calibration_world_frame.py` 所在目录下。

### 5.3 交互按键

窗口内按键定义如下：

- `f`：记录一个地面点
- `0`：记录一个 `P0` 采样
- `1`：记录一个 `P1` 采样
- `z`：记录一个 `Pz` 采样，要求该点明显位于地面上方
- `c`：计算并保存 `T_W_LH`
- `r`：清空当前所有采样与标定结果
- `q` 或 `ESC`：退出

窗口底部也会显示这些快捷键。

### 5.4 采样要求

脚本当前的硬性检查如下：

- 地面点 `floor`：至少 `12` 个
- `P0`：至少 `10` 个
- `P1`：至少 `10` 个
- `Pz`：至少 `10` 个

建议实际采样数量：

- `floor`：`20-40` 个
- `P0`：`20-50` 个
- `P1`：`20-50` 个
- `Pz`：`20-50` 个

采样建议：

- 只在状态栏显示 `STILL` 时按键采样
- 地面点尽量分布在较大的平面区域，不要只采一小块
- `P0` 和 `P1` 要拉开明显距离，建议至少 `20 cm`
- `P1` 应该放在你希望世界坐标 `+X` 指向的位置
- `Pz` 需要明确高于地面，且不要离地面太近，否则脚本可能拒绝计算 `+Z`

## 6. 标定成功后的现象

按 `c` 后，脚本会：

1. 对地面点做 RANSAC 平面拟合
2. 用内点重新估计地面法向
3. 利用 `Pz` 决定法向朝上方向
4. 构建 `T_W_LH`
5. 把结果写入 `--out` 指定的 JSON 文件

终端会打印：

- `RANSAC inliers`
- `plane_rmse_m(inliers)`
- `P0_std_m`
- `P1_std_m`
- `Pz_std_m`

经验上建议关注：

- `RANSAC inliers / total` 越高越好
- `plane_rmse_m(inliers)` 尽量小，稳定情况下通常应小于 `0.003 ~ 0.005` m
- `P0_std_m` / `P1_std_m` / `Pz_std_m` 的每个轴尽量小，通常应小于 `0.002 ~ 0.005` m

## 7. 输出 JSON 字段

执行 `c` 后，JSON 中通常包含：

- `T_W_LH`
  - 4x4 齐次变换矩阵
- `floor_normal_LH`
  - `LH` 坐标系中的地面法向
- `plane_rmse_m_inliers`
  - RANSAC 内点上的平面 RMSE
- `ransac_thresh_m`
  - 本次使用的 RANSAC 阈值
- `ransac_inliers`
  - RANSAC 内点数
- `ransac_total`
  - 地面点总数
- `P0_mean_LH`
- `P1_mean_LH`
- `Pz_mean_LH`
- `P0_std_m`
- `P1_std_m`
- `Pz_std_m`
- `tracker_key`
  - 本次使用的 tracker 标识
- `notes`
  - 对世界坐标系定义的文本说明

## 8. 旋转补偿 `rotation_offset` 标定

### 8.1 用途

`rotation_offset` 是一个四元数补偿项，脚本保存格式为：

- `[qx, qy, qz, qw]`

它用于把 tracker 的实际姿态零位，补到项目期望的工具坐标零位。

### 8.2 前提

先保证输出 JSON 里已经有合法的 `T_W_LH`。常见做法是先完成上一节的世界系标定。

脚本启动时如果发现 `--out` 对应文件中已经有：

- `T_W_LH`
- `rotation_offset`

会自动加载它们。

### 8.3 运行方式

```bash
python teleop/pika_tracker/calibration_world_frame.py \
  --tracker-serial LHR-0DFD738C \
  --out config/T_W_LH.json \
  --calib_rot_offset
```

### 8.4 操作要点

这个模式下不需要再按 `f/0/1/z/c`。

脚本会持续读取最近一段静止数据，并基于当前 `T_W_LH` 计算一个 `rotation_offset`，然后把它追加写回同一个 JSON 文件中。

建议操作：

- 让 tracker 保持静止
- 把 tracker 摆到你定义的“零姿态”
- 这个“零姿态”应与世界坐标轴方向对齐，因为脚本当前的目标姿态是世界系单位旋转 `Identity`

换句话说，标定 `rotation_offset` 时，你手里拿的 tracker 朝向，就会被当作后续使用时的参考零姿态。

### 8.5 写回行为

`rotation_offset` 标定不会删除原有字段，而是把新的 `rotation_offset` 合并写回已有 JSON。

## 9. 常见问题

### 9.1 `No pose yet` / `Waiting for pose...`

说明脚本还没收到目标 tracker 的稳定数据。排查顺序：

- 先用 `--list` 看是否能看到设备
- 检查 `--tracker-serial` 是否写错
- 检查 Lighthouse、tracker、电源、接收器状态

### 9.2 `Pose not stable enough yet`

说明最近窗口内采样不足或姿态抖动较大。做法：

- 暂停移动，等状态栏出现 `STILL`
- 适当降低环境抖动
- 必要时可略微调大 `--still-speed`

### 9.3 `RANSAC failed`

通常是地面点质量不够：

- 地面点太少
- 分布过于集中
- 离群点太多

处理方式：

- 重新采 `20+` 个地面点
- 尽量覆盖更大区域
- 必要时把 `--ransac-thresh` 从 `0.01` 放宽到 `0.012 ~ 0.015`

### 9.4 `Failed to determine +Z direction`

这是当前版本新增的检查，说明 `Pz` 离地面太近，无法可靠判断朝上方向。

处理方式：

- 重新采一组更高的 `Pz`
- 避免 `Pz` 与 floor 平面几乎重合

### 9.5 标定后 `+X` 方向不对

`+X` 完全由 `P0 -> P1` 决定。需要：

- 重新选择 `P1`
- 确保 `P0` 与 `P1` 不要几乎重合

### 9.6 标定文件看起来没有写到当前目录

这是因为 `--out` 的相对路径是相对于脚本目录解析的，不是相对于你执行命令时的当前目录。

例如：

```bash
--out config/T_W_LH.json
```

最终会写到：

- `teleop/pika_tracker/config/T_W_LH.json`

## 10. 推荐命令示例

### 10.1 查看 tracker 列表

```bash
python teleop/pika_tracker/calibration_world_frame.py --list
```

### 10.2 标定世界坐标系

```bash
python teleop/pika_tracker/calibration_world_frame.py \
  --tracker-serial LHR-0DFD738C \
  --out config/T_W_LH.json
```

### 10.3 标定旋转零位补偿

```bash
python teleop/pika_tracker/calibration_world_frame.py \
  --tracker-serial LHR-0DFD738C \
  --out config/T_W_LH.json \
  --calib_rot_offset
```

## 11. 建议的完整流程

1. 用 `--list` 确认 tracker key。
2. 运行交互标定，采集 `floor + P0 + P1 + Pz`。
3. 按 `c` 生成 `T_W_LH` 并检查误差指标。
4. 如需姿态零位补偿，再运行一次 `--calib_rot_offset`。
5. 最终确认 `teleop/pika_tracker/config/T_W_LH.json` 同时包含 `T_W_LH` 和 `rotation_offset`。
