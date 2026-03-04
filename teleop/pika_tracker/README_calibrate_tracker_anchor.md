# `calibrate_tracker_anchor.py` 使用说明

本文档介绍一种基于 `anchor tracker` 的方式来获取 tracker 的参考系下坐标与姿态，对应脚本：

- [`teleop/pika_tracker/calibrate_tracker_anchor.py`](./calibrate_tracker_anchor.py)

这种方法不再依赖地面点、`P0/P1/Pz` 去拟合世界坐标系，而是直接把一个固定不动的 tracker 当作参考锚点 `anchor`，用它的初始位姿定义一个“绝对参考坐标系”。

## 1. 方法概述

anchor tracker 方法的核心思路是：

1. 选一个 tracker 固定安装在场景中，作为 `anchor tracker`
2. 程序启动后，读取这个 anchor 的初始位姿
3. 把 anchor 的这个初始位姿当作参考系原点与方向
4. 其他 tracker 的位姿都转换到这个 anchor 参考系下表达

因此，这里的“绝对坐标”本质上是：

- 相对于 anchor 初始位姿定义的固定参考系
- 不是通过地面拟合得到的全局世界系
- 也不是机器人本体坐标系，除非你把 anchor 本身固定在那个参考位置上

这个方法适合：

- 场景中已经有一个可以长期固定的 tracker
- 不想每次都做 `floor + P0 + P1 + Pz` 的世界系标定
- 更关注“相对某个固定参考点”的稳定坐标

## 2. 与 `calibration_world_frame.py` 的区别

和 [`teleop/pika_tracker/README_calibration_world_frame.md`](./README_calibration_world_frame.md) 对应的方法相比：

- `world_frame` 方法
  - 参考系由地面法向、`P0`、`P1` 人工定义
  - 可以明确指定 `+X`、`+Z` 的物理含义
- `anchor tracker` 方法
  - 参考系直接由 anchor tracker 的初始位姿定义
  - 不需要采地面点
  - 部署更快，但参考轴方向取决于 anchor 安装姿态

简单理解：

- `world_frame` 更像“显式建图”
- `anchor tracker` 更像“拿一个固定 tracker 当基准坐标系”

## 3. 脚本提供的两部分能力

`calibrate_tracker_anchor.py` 里实际上有两部分功能。

### 3.1 `CalibrateTracker` 类

这个类用于在项目代码中读取经过 anchor 校正的 tracker 数据，但当前实现更偏向“姿态对齐 + 可选相对平移零点”，并不是完整地把所有位姿都变换到 anchor 初始坐标系。

它内部会：

- 启动 `ViveTracker`
- 检查配置里的所有 tracker uid 是否在线
- 记录 anchor tracker
- 用当前 anchor 的姿态去对齐其他 tracker 的姿态输出
- 按配置可选地加上 `z_offset`
- 按配置可选地启用相对平移模式 `use_relative`

要特别注意：

- 当前类实现没有把位置整体左乘 `anchor_pose_inv` 做完整坐标变换
- 它主要校正的是姿态
- 位置部分默认仍保留 tracker 原始平移，只能额外做 `z_offset` 和可选的 `x/y` 零点平移

### 3.2 脚本主函数 `_main()`

直接运行该脚本时，默认进入一个诊断模式，用于检查：

- anchor tracker 与目标 tracker 之间的相对姿态是否稳定
- 应用姿态对齐后，两者的相对旋转误差是否足够小
- 在 anchor 初始参考系下，目标 tracker 的位置输出是否符合预期

这个模式更适合用来调试 anchor 安装方式和检查坐标效果。

## 4. 参考系定义

在脚本诊断模式里，参考系的定义是：

- 原点：anchor tracker 启动时的初始位置
- 朝向：anchor tracker 启动时的初始姿态

也就是说，脚本会先读取一次 anchor 的初始 pose，然后求它的逆变换，再把后续所有 tracker pose 都映射到这个参考系中。

因此如果 anchor 在运行过程中发生移动，输出结果就不再代表一个稳定固定的“绝对系”。

结论很直接：

- anchor tracker 必须固定牢
- 程序启动后不要再碰它

## 5. 适用前提

使用该方法前，建议满足：

- Lighthouse 与所有 tracker 均工作正常
- `pysurvive` / `ViveTracker` 可以稳定读到所有目标 uid
- anchor tracker 已固定在场景中，不会晃动或位移
- 你清楚 anchor 自身安装姿态决定了参考坐标轴方向

## 6. 配置项说明

`CalibrateTracker` 类需要的核心配置字段如下：

- `tracker_uids`
  - 参与计算的所有 tracker uid 列表，必须包含 anchor uid
- `anchor_uid`
  - 作为参考锚点的 tracker uid
- `reading_loop_s` 或 `poll_interval`
  - 后台读取周期
- `timeout`
  - 初始化等待所有 tracker 出现的超时时间
- `z_offset`
  - 对输出位置的 `z` 额外加一个固定偏移量
- `use_relative`
  - 是否启用相对平移模式
- `auto_start`
  - 是否在初始化时自动启动读取线程

脚本内部有如下约束：

- `anchor_uid` 必须在 `tracker_uids` 中
- 如果初始化超时还没看到所有 uid，会直接失败

### 6.1 示例配置

可以参考下面这种 YAML 结构：

```yaml
tracker:
  tracker_uids:
    - "LHR-ANCHOR0001"
    - "LHR-TARGET0001"
  anchor_uid: "LHR-ANCHOR0001"
  reading_loop_s: 0.005
  timeout: 5.0
  z_offset: 0.0
  use_relative: false
```

注意：

- 脚本默认的 `--config` 是 `common/hardwares/configs/calib_tracker_anchor_cfg.yaml`
- 当前仓库里未必已有这个文件
- 实际使用时通常需要你自己显式传入一个存在的配置文件路径

## 7. 直接运行脚本进行诊断

### 7.1 基本命令

```bash
python teleop/pika_tracker/calibrate_tracker_anchor.py \
  --config <你的anchor配置yaml> \
  --uid <目标tracker_uid>
```

常用参数：

- `--config`
  - anchor 配置文件路径
- `--uid`
  - 要和 anchor 比较的目标 tracker uid
- `--hz`
  - 终端打印频率，默认 `5.0`
- `--timeout`
  - 覆盖配置中的超时参数

### 7.2 启动后做了什么

脚本会：

1. 加载配置文件
2. 等待 `tracker_uids` 中所有 tracker 出现
3. 读取 anchor 的初始 pose
4. 持续读取 anchor 和目标 tracker 的 pose
5. 打印三组相对旋转报告：
   - `raw_rel`
   - `aligned_rel`
   - `final_rel`
6. 打印 `final_target_pos_xyz` 与 `final_anchor_pos_xyz`

## 8. 输出内容如何理解

脚本诊断模式下会循环输出：

- `raw_rel`
  - 原始 tracker 姿态直接计算的相对旋转
- `aligned_rel`
  - 在做过 Vive 四元数对齐修正后的相对旋转
- `final_rel`
  - 在 anchor 初始参考系下的最终相对旋转
- `final_target_pos_xyz`
  - 目标 tracker 在 anchor 初始参考系下的位置
- `final_anchor_pos_xyz`
  - anchor 自己在该参考系下的位置

理想情况下：

- 如果 anchor 和目标 tracker 被摆成相同朝向
- 且都保持静止

那么你应该看到：

- `final_rel.angle_deg` 接近 `0`
- `final_anchor_pos_xyz` 接近 `[0, 0, 0]`
- `final_target_pos_xyz` 表示目标 tracker 相对 anchor 的三维位置

## 9. `CalibrateTracker` 类里的相对模式 `i/u`

`CalibrateTracker` 类还带了一个键盘交互功能：

- 按 `i`
  - 记录当前所有 tracker 的位置作为“相对零点”
  - 之后如果 `use_relative=true`，输出会减去这个时刻记录的 `x/y`
- 按 `u`
  - 关闭这个相对模式的启用状态

这里要注意脚本当前实现的细节：

- 相对模式只对平移的 `x/y` 生效
- `z` 不会被这一步抵消
- 姿态仍然是通过 anchor 姿态去对齐的

所以它更像是：

- 用 anchor 处理姿态参考系
- 用 `i` 键记录一个平面内的相对平移零点
- 而不是完整构造一个新的三维 anchor 世界坐标系

## 10. 这种方法下“绝对坐标”到底是什么意思

这是最容易混淆的地方。

这里的“绝对坐标”不是地图意义上的全局坐标，而是：

- 在 anchor 固定不动前提下
- 用 anchor 初始 pose 定义的固定参考系坐标

只要下面两个条件成立，这个坐标就可以被当作稳定的绝对参考：

1. anchor 始终固定不动
2. Lighthouse 环境稳定，没有明显跳变

如果 anchor 被碰动、重新安装或启动时姿态不同，那么这个“绝对系”就会变化，需要重新启动并重新建立参考。

## 11. 推荐操作流程

### 11.1 初次部署

1. 选一个 tracker 作为 anchor。
2. 把它牢固固定在场景中。
3. 记录 anchor 和目标 tracker 的 uid。
4. 写好配置文件，确保 `tracker_uids` 包含所有相关设备。
5. 运行诊断脚本检查参考系输出是否符合预期。

### 11.2 诊断检查

建议先让 anchor 与目标 tracker 尽量摆成同一姿态，然后运行：

```bash
python teleop/pika_tracker/calibrate_tracker_anchor.py \
  --config <你的anchor配置yaml> \
  --uid <目标tracker_uid> \
  --hz 5
```

重点看：

- `final_rel.angle_deg` 是否足够小
- `final_target_pos_xyz` 是否和你实际拿着目标 tracker 的位置一致
- 移动目标 tracker 时，坐标变化方向是否符合你的参考轴预期

### 11.3 项目内使用

如果不是做诊断，而是想在程序里持续拿经过 anchor 校正的数据，可以直接使用 `CalibrateTracker` 类。

典型调用逻辑是：

1. 用配置初始化 `CalibrateTracker`
2. 周期性调用 `get_pose_by_uid()` 或 `get_pose_dict()`
3. 读取经过 anchor 姿态校正后的 pose 输出
4. 如果你需要“完整 anchor 初始参考系下的位置”，应参考 `_main()` 里的 `anchor_pose_inv` 变换逻辑，而不是直接把 `CalibrateTracker` 当前输出当成严格意义上的 anchor 世界坐标
5. 如有需要，在 `use_relative=true` 时按 `i` 记录相对平移零点

## 12. 优势与限制

优势：

- 不需要手动采地面点
- 部署快
- 对重复启动较友好
- 适合固定工作站、固定支架、固定场景

限制：

- 参考系方向完全取决于 anchor 安装姿态
- anchor 一旦移动，参考系就失效
- 不像 `world_frame` 方法那样天然具备“地面向上”的物理语义
- 如果需要严格定义 `+Z` 为重力反方向、`+X` 指向某个工位方向，`world_frame` 方法通常更合适

## 13. 常见问题

### 13.1 初始化失败，提示无法看到所有 tracker uid

排查：

- 检查配置里的 `tracker_uids` 是否写对
- 检查 tracker 是否都已上电并被基站看到
- 增大 `timeout`

### 13.2 输出不稳定

通常说明：

- anchor 没固定牢
- Lighthouse 环境不稳定
- tracker 遮挡严重

### 13.3 坐标轴方向和预期不一致

这是 anchor 方法的正常特性，因为参考轴来自 anchor 初始姿态。

解决方式：

- 调整 anchor 的安装姿态后重新启动
- 或改用 `calibration_world_frame.py` 手工定义世界坐标轴

### 13.4 为什么按 `i` 之后只有平移变了，姿态没变成零

因为当前实现里：

- `i` 只记录相对平移参考
- 姿态参考仍然来自 anchor tracker

这不是 bug，而是脚本当前设计如此。

## 14. 最小示例

### 14.1 配置文件示例

```yaml
tracker:
  tracker_uids:
    - "LHR-ANCHOR0001"
    - "LHR-TARGET0001"
  anchor_uid: "LHR-ANCHOR0001"
  reading_loop_s: 0.005
  timeout: 5.0
  use_relative: false
```

### 14.2 启动诊断

```bash
python teleop/pika_tracker/calibrate_tracker_anchor.py \
  --config teleop/pika_tracker/config/your_anchor_cfg.yaml \
  --uid LHR-TARGET0001
```

## 15. 什么时候选这个方法

优先选 anchor tracker 方法的情况：

- 场景里可以长期固定一个 tracker
- 你更关注快速部署
- 你只需要“相对这个固定参考点”的稳定坐标

优先选 `world_frame` 方法的情况：

- 你需要显式定义地面和朝上方向
- 你需要一个更符合物理语义的世界坐标系
- 你希望 `+X/+Z` 的含义可控，而不是由 anchor 安装姿态决定
