这个目录用于构建一个独立的、专门给 FR3 真机运行 diffusion policy 推理的容器。

设计原则：

- 独立镜像，不依赖外部 FR3 base image
- 只安装 FR3 真机推理链路和 DP 推理链路需要的依赖
- 不安装 `dp_hirol-main/requirements/uv-common.txt` 和 `uv-sim-cu116.txt` 那批训练/仿真重依赖

FR3 真机依赖直接在镜像中安装：

- `panda-python`
- `pin`
- `casadi`
- `pyrealsense2`
- `NetFT`
- `ruckig`
- `sshkeyboard`
- `gymnasium`
- `glog`

两个仓库外 SDK 仍然保留本地自动安装入口：

- `docker_for_inference/vendor/dm_robotics_panda`
- `docker_for_inference/vendor/pika_sdk`

如果你需要真机 Pika gripper 全链路运行，把这两个目录放进去后重新 build，Dockerfile 会自动执行 `pip install -e`。

如果后续你的 checkpoint 不是常规 FR3 diffusion-unet 推理模型，而是依赖 `robomimic`、`r3m` 之类额外视觉编码器/策略，再单独补对应依赖。

常用命令：

```bash
cd docker_for_inference
docker compose build --no-cache dp_inference
docker compose up -d
docker compose exec dp_inference python factory/tasks/inferences_tasks/dp/dp_inference.py \
  --config factory/tasks/inferences_tasks/dp/config/fr3_dp_ddim_inference_cfg.yaml
```
