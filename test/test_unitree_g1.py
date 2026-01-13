# tests/test_g1_mujoco_bridge.py
from __future__ import annotations

import time
import argparse
import numpy as np
import glog as log
import os

from hardware.base.utils import dynamic_load_yaml
from simulation.mujoco.mujoco_sim import MujocoSim
from hardware.unitreeG1.unitree_g1 import UnitreeG1


def _load_subcfg(yaml_path: str, prefer_keys: list[str]):
    """Load yaml and return the first matched sub-config; fallback to full dict."""
    cfg = dynamic_load_yaml(yaml_path)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {yaml_path} is not a dict.")
    for k in prefer_keys:
        if k in cfg:
            return cfg[k]
    return cfg


def _apply_index_map(src: np.ndarray, index_map: list[int] | None, dst_len: int) -> np.ndarray:
    """
    Map src -> dst by index_map (dst[i] = src[index_map[i]]).
    If index_map is None, take prefix min(dst_len, len(src)).
    """
    if index_map is None:
        n = min(dst_len, len(src))
        out = np.zeros(dst_len, dtype=np.float64)
        out[:n] = src[:n]
        return out

    if len(index_map) != dst_len:
        raise ValueError(f"index_map length {len(index_map)} != dst_len {dst_len}")

    out = np.zeros(dst_len, dtype=np.float64)
    for i, j in enumerate(index_map):
        if j < 0 or j >= len(src):
            raise IndexError(f"index_map[{i}]={j} out of range for src len={len(src)}")
        out[i] = src[j]
    return out


def test_hw_state_to_sim(
    mujoco_cfg_path: str,
    g1_cfg_path: str,
    duration_s: float = 10.0,
    rate_hz: float = 200.0,
    index_map_hw_to_sim: list[int] | None = None,
):
    """
    读取 Unitree G1 硬件电机角度(q)，直接设置到仿真(qpos)里。

    - duration_s: 运行时长（秒）
    - rate_hz:    同步频率
    - index_map_hw_to_sim: 可选映射（仿真关节顺序 != 硬件顺序时用）
    """
    cur_path = os.path.dirname(__file__)
    mujoco_cfg_path = os.path.join(cur_path, '..', mujoco_cfg_path)
    # mujoco_cfg_path = os.path.join(cur_path, '..', "simulation/config/mujoco_fr3_pika_ati_torque.yaml")
    g1_cfg_path = os.path.join(cur_path, '..', g1_cfg_path)
    mujoco_cfg = _load_subcfg(mujoco_cfg_path, ["mujoco"])
    g1_cfg = _load_subcfg(g1_cfg_path, ["unitree_g1", "unitreeG1", "g1"])

    sim = MujocoSim(mujoco_cfg)
    g1 = UnitreeG1(g1_cfg)

    try:
        g1.initialize()

        # 推断仿真关节数量（优先用 joint_names）
        sim_dof = len(getattr(sim, "_joint_names", [])) or len(getattr(sim, "_actuator_mode", []))
        if sim_dof <= 0:
            raise RuntimeError("Cannot infer MujocoSim DoF (no _joint_names/_actuator_mode).")
        time.sleep(1.5)

        dt = 1.0 / max(rate_hz, 1e-6)
        t0 = time.perf_counter()
        k = 0
        
        log.info(f"[HW->SIM] start: duration={duration_s}s, rate={rate_hz}Hz, sim_dof={sim_dof}")
        # (time.perf_counter() - t0) < duration_s
        while True:
            # 读硬件关节角
            js = g1.get_joint_states()  # ArmBase 通常提供
            hw_pos = np.asarray(js._positions, dtype=np.float64)
            log.info(f'hw position: {hw_pos}')

            # 映射到仿真关节顺序并写入仿真
            sim_pos = hw_pos
            # sim_pos = _apply_index_map(hw_pos, index_map_hw_to_sim, sim_dof)

            # 尽量用 step_lock 避免和 mj_step 数据竞争（如果 MujocoSim 有这个锁）
            # log.info(f'hw position: {hw_pos}')
            sim.set_joint_command(["position"]*(sim_dof), sim_pos)
            # step_lock = getattr(sim, "_step_lock", None)
            # if step_lock is None:
            #     sim.set_joint_command(["position"]*(sim_dof), sim_pos)
            # else:
            #     with step_lock:
            #         sim.set_joint_command(["position"]*(sim_dof), sim_pos)

            k += 1
            if k % int(max(rate_hz, 1.0)) == 0:
                pass
                # log.info(f"[HW->SIM] synced frames={k}, hw_pos[0:3]={hw_pos[:3]}")

            time.sleep(dt)
            # print(f'sleep for {dt}')

    finally:
        try:
            # sim.close()
            pass
        except Exception as e:
            log.warn(f"sim.close() failed: {e}")
        try:
            # g1.close()
            pass
        except Exception as e:
            log.warn(f"g1.close() failed: {e}")


def test_sim_ctrl_to_hw(
    mujoco_cfg_path: str,
    g1_cfg_path: str,
    duration_s: float = 10.0,
    rate_hz: float = 200.0,
    hw_mode: str | None = "position",  # "position" or "torque"
    index_map_sim_to_hw: list[int] | None = None,
):
    """
    读取仿真中的 ctrl 数值，设置给 Unitree G1 硬件 command 执行。

    - hw_mode: None 时尝试从 mujoco actuator_mode 推断；否则显式指定 "position"/"torque"
    - index_map_sim_to_hw: 可选映射（硬件关节顺序 != 仿真顺序时用）
    """
    cur_path = os.path.dirname(__file__)
    mujoco_cfg_path = os.path.join(cur_path, '..', mujoco_cfg_path)
    g1_cfg_path = os.path.join(cur_path, '..', g1_cfg_path)
    mujoco_cfg = _load_subcfg(mujoco_cfg_path, ["mujoco"])
    g1_cfg = _load_subcfg(g1_cfg_path, ["unitree_g1", "unitreeG1", "g1"])

    sim = MujocoSim(mujoco_cfg)
    g1 = UnitreeG1(g1_cfg)

    try:
        g1.initialize()

        # 读取仿真 actuator_mode 推断 hw_mode
        if hw_mode is None:
            act_modes = getattr(sim, "_actuator_mode", None)
            if isinstance(act_modes, (list, tuple)) and len(act_modes) > 0:
                uniq = set(act_modes)
                if len(uniq) == 1:
                    m = list(uniq)[0]
                    if m in ("position", "torque"):
                        hw_mode = m
            if hw_mode is None:
                hw_mode = "torque"  # 默认更符合 ctrl 的语义（但取决于 xml actuator）

        if hw_mode not in ("position", "torque"):
            raise ValueError(f"hw_mode must be 'position' or 'torque', got {hw_mode}")

        # 硬件 dof（按 UnitreeG1._robot_id 的长度）
        hw_dof = len(getattr(g1, "_robot_id", []))
        if hw_dof <= 0:
            # fallback：从 joint_states 维度推断
            hw_dof = len(g1.get_joint_states()._positions)

        dt = 1.0 / max(rate_hz, 1e-6)
        t0 = time.perf_counter()
        k = 0

        log.info(f"[SIM->HW] start: duration={duration_s}s, rate={rate_hz}Hz, hw_dof={hw_dof}, mode={hw_mode}")

        # (time.perf_counter() - t0) < duration_s
        while True:
            # 线程安全读取 ctrl：优先从 render_data + render_lock（因为仿真线程会把 ctrl copy 过去）
            # render_lock = getattr(sim, "_render_lock", None)
            # if render_lock is None or getattr(sim, "_render_data", None) is None:
            #     # fallback：直接读 data.ctrl（可能有轻微竞争，但一般能用）
            #     step_lock = getattr(sim, "_step_lock", None)
            #     if step_lock is None:
            #         sim_ctrl = np.asarray(sim._data.ctrl, dtype=np.float64).copy()
            #     else:
            #         with step_lock:
            #             sim_ctrl = np.asarray(sim._data.ctrl, dtype=np.float64).copy()
            # else:
            #     with render_lock:
            #         sim_ctrl = np.asarray(sim._render_data.ctrl, dtype=np.float64).copy()
            sim_ctrl = sim.get_joint_states()._positions
            assert len(sim_ctrl) == 14, f"sim len {len(sim_ctrl)} != 14"
            
            # 映射到硬件关节维度（只取需要的 dof）
            hw_cmd = sim_ctrl
            # hw_cmd = _apply_index_map(sim_ctrl, index_map_sim_to_hw, hw_dof)

            # 下发给硬件
            # log.info(f'{hw_mode} -> Sim ctrl value: {sim_ctrl}')
            g1.set_joint_command(hw_mode, hw_cmd)

            k += 1
            if k % int(max(rate_hz, 1.0)) == 0:
                log.info(f"[SIM->HW] mode {hw_mode} sent frames={k}, cmd[0:3]={hw_cmd[:3]}")

            time.sleep(dt)

    finally:
        try:
            sim.close()
        except Exception as e:
            log.warn(f"sim.close() failed: {e}")
        try:
            g1.close()
        except Exception as e:
            log.warn(f"g1.close() failed: {e}")


def _parse_index_map(s: str | None) -> list[int] | None:
    """
    Parse mapping like: "0,1,2,3,4"  (dst_len entries)
    """
    if s is None or s.strip() == "":
        return None
    return [int(x) for x in s.split(",")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mujoco_cfg", type=str, 
                        help="path to mujoco yaml (contains key 'mujoco')",
                        default="simulation/config/mujoco_unitree_g1_only_upper.yaml")
    parser.add_argument("--g1_cfg", type=str, 
                        help="path to unitree g1 yaml (contains unitree_g1/unitreeG1/g1)",
                        default="hardware/unitreeG1/config/unitree_g1_upper.yaml")
    parser.add_argument("--test", type=str, choices=["hw2sim", "sim2hw"], required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--rate", type=float, default=200.0)
    parser.add_argument("--hw_mode", type=str, default=None, help="position or torque (for sim2hw); default: infer or torque")
    parser.add_argument("--map", type=str, default=None, help="comma-separated index map (dst order)")

    args = parser.parse_args()
    idx_map = _parse_index_map(args.map)

    if args.test == "hw2sim":
        test_hw_state_to_sim(
            mujoco_cfg_path=args.mujoco_cfg,
            g1_cfg_path=args.g1_cfg,
            duration_s=args.duration,
            rate_hz=args.rate,
            index_map_hw_to_sim=idx_map,
        )
    else:
        test_sim_ctrl_to_hw(
            mujoco_cfg_path=args.mujoco_cfg,
            g1_cfg_path=args.g1_cfg,
            duration_s=args.duration,
            rate_hz=args.rate,
            hw_mode=args.hw_mode,
            index_map_sim_to_hw=idx_map,
        )
