这个目录用于给 `HirolPlatform/factory/tasks/inferences_tasks/dp/dp_inference.py` 构建推理容器。

这个构建默认走“FR3 推理最小依赖集”，不会安装 `dp_hirol-main/requirements/uv-sim-cu116.txt` 里那批仿真/训练附加包，因此可以避开 `r3m` 等 GitHub tarball 依赖。

FR3 真机链路已经在镜像里覆盖了这些依赖：

- `panda-python==0.8.1`
- `pin==3.7.0`
- `casadi==3.7.0`
- `pyrealsense2==2.56.4.9191`
- `NetFT==2.0.1`
- `ruckig==0.14.0`
- `sshkeyboard==2.3.1`

另外有两个仓库外 SDK 无法从当前代码仓直接确定安装源，Dockerfile 预留了本地自动安装入口：

- `docker_for_inference/vendor/dm_robotics_panda`
- `docker_for_inference/vendor/pika_sdk`

如果你在另一台机器上需要真机 FR3 + Pika gripper 全链路运行，把这两个目录放进去后重新 build，Dockerfile 会自动执行 `pip install -e`。

如果后续你的 checkpoint 不是常规 FR3 diffusion-unet 推理模型，而是依赖 `robomimic`、`r3m` 之类额外视觉编码器/策略，再单独补对应依赖。

常用命令：

```bash
cd docker_for_inference
docker compose build --no-cache dp_inference
docker compose up -d
docker compose exec dp_inference python factory/tasks/inferences_tasks/dp/dp_inference.py \
  --config factory/tasks/inferences_tasks/dp/config/fr3_dp_ddim_inference_cfg.yaml
```
