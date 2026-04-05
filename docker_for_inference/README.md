这个目录用于构建一个独立的、专门给 FR3 真机运行 diffusion policy 推理的容器。

设计原则：

- 独立镜像，不依赖外部 FR3 base image
- 只安装 FR3 真机推理链路和 DP 推理链路需要的依赖
- 不安装 `dp_hirol-main/requirements/uv-common.txt` 和 `uv-sim-cu116.txt` 那批训练/仿真重依赖

FR3 真机依赖直接在镜像中安装：

- `pin`
- `casadi`
- `pyrealsense2`
- `NetFT`
- `ruckig`
- `sshkeyboard`
- `gymnasium`
- `glog`

`panda_py` / `libfranka` 现在按下面的优先级安装：

1. 如果放了 `docker_for_inference/vendor/libfranka`，镜像会先源码编译 `libfranka`
2. 如果放了 `docker_for_inference/vendor/panda_py` 或 `docker_for_inference/vendor/panda-python`，镜像会源码安装 `panda_py`
3. 如果没有本地源码，默认回退到 `pip install panda-python==0.8.1`

这意味着现在容器可以兼容两种场景：

- 你只想先验证推理链路，直接用 PyPI 版本
- 你已经确认 `panda_py` / `libfranka` 有 ABI 或驱动兼容问题，需要在容器里重建

两个仓库外 SDK 仍然保留本地自动安装入口：

- `docker_for_inference/vendor/dm_robotics_panda`
- `docker_for_inference/vendor/pika_sdk`

如果你需要真机 Pika gripper 全链路运行，把这两个目录放进去后重新 build，Dockerfile 会自动执行 `pip install -e`。

如果你要强制使用源码版 `panda_py`，构建前可以导出：

```bash
export PANDA_PY_INSTALL_MODE=source
```

如果你希望显式回退到 PyPI 版本：

```bash
export PANDA_PY_INSTALL_MODE=pip
export PANDA_PY_VERSION=0.8.1
```

如果后续你的 checkpoint 不是常规 FR3 diffusion-unet 推理模型，而是依赖 `robomimic`、`r3m` 之类额外视觉编码器/策略，再单独补对应依赖。

常用命令：

```bash
cd docker_for_inference
docker compose build --no-cache dp_inference
docker compose up -d
docker compose exec dp_inference python factory/tasks/inferences_tasks/dp/dp_inference.py \
  -c factory/tasks/inferences_tasks/dp/config/fr3_dp_ddim_inference_full_cfg.yaml
```

建议的 vendor 目录结构：

```text
docker_for_inference/vendor/
├── libfranka/
├── panda_py/                # 或 panda-python/
├── dm_robotics_panda/
└── pika_sdk/
```
