# 请参考[impedance_cfg](./config/impedance_fr3_cfg.yaml)
## 参数说明
1. `stiffness`: 刚度参数(6维向量)
2. `saturation`: 输出关节力矩的上下限
3. `kp_null`: 关节构型零空间的kp参数(向量表达，维度需为主动关节自由度一样)
4. `q_des`: 零空间关节构型的目标位置
5. `dq_damping`: 关节构型零空间的damping系数
6. `enable_acceleration_feedforward`: 是否启用显式加速度前馈
7. `acceleration_feedforward_gain`: 显式加速度前馈增益
8. `compensate_jdot_qdot`: 是否在前馈里补偿 `Jdot*qdot`
9. `max_task_acceleration`: task acceleration 限幅，6维

## target 输入
- 兼容原始格式: `{frame_name: [x,y,z,qx,qy,qz,qw]}`
- 启用显式加速度前馈时可传:
  - `{frame_name: {"pose": [x,y,z,qx,qy,qz,qw], "task_acceleration": [ax,ay,az,alphax,alphay,alphaz]}}`
   
