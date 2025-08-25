"""
Critical Damped Smoother
Implements second-order critical damped tracking
"""

import numpy as np
import threading
import time
import logging
from typing import Dict, Any, Tuple

from .smoother_base import SmootherBase

logger = logging.getLogger(__name__)


class CriticalDampedSmoother(SmootherBase):
    """二阶临界阻尼平滑器"""
    
    def __init__(self, config: Dict[str, Any], dof: int):
        """
        初始化二阶系统平滑器
        
        参数:
            config: 包含 omega_n, control_frequency
            dof: 自由度数量
        """
        super().__init__(config, dof)
        
        # 二阶系统参数
        self._omega_n = config.get("omega_n", 25.0)
        self._zeta = config.get("zeta", 1.0)  # 临界阻尼
        self._control_freq = config.get("control_frequency", 800.0)
        
        # 状态变量
        self._target_joints = np.zeros(dof)
        self._current_joints = np.zeros(dof)
        self._joint_velocity = np.zeros(dof)
        self._joint_acceleration = np.zeros(dof)
        
        # 线程控制
        self._lock = threading.Lock()
        self._pause_flag = False
        self._thread = None
        
        # 性能监控
        self._slow_loop_count = 0
        
        logger.info(f"CriticalDampedSmoother initialized: "
                   f"ωn={self._omega_n:.1f} rad/s, "
                   f"settling_time≈{4.6/self._omega_n:.3f}s")
    
    def start(self, initial_positions: np.ndarray) -> None:
        """启动平滑器线程"""
        if self._is_running:
            raise RuntimeError("Smoother already running")
        
        # 初始化状态
        with self._lock:
            self._current_joints = initial_positions.copy()
            self._target_joints = initial_positions.copy()
            self._joint_velocity = np.zeros(self._dof)
            self._joint_acceleration = np.zeros(self._dof)
        
        # 启动线程
        self._is_running = True
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()
        logger.info(f"Smoother thread started at {self._control_freq}Hz")
    
    def stop(self) -> None:
        """停止平滑器线程"""
        self._is_running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                logger.warning("Smoother thread did not stop cleanly")
        logger.info("Smoother stopped")
    
    def update_target(self, joint_target: np.ndarray, immediate: bool = False) -> None:
        """
        更新目标关节位置
        
        参数:
            joint_target: 目标关节角度
            immediate: True时立即跳转（用于reset）
        """
        assert joint_target.shape == (self._dof,), f"Expected shape ({self._dof},), got {joint_target.shape}"
        
        with self._lock:
            self._target_joints = joint_target.copy()
            
            if immediate:
                # 立即跳转模式
                self._current_joints = joint_target.copy()
                self._joint_velocity = np.zeros(self._dof)
                self._joint_acceleration = np.zeros(self._dof)
                logger.debug("Immediate jump to target")
    
    def get_command(self) -> Tuple[np.ndarray, bool]:
        """获取当前平滑后的关节命令"""
        with self._lock:
            return self._current_joints.copy(), not self._pause_flag
    
    def pause(self) -> None:
        """暂停平滑（保持当前输出）"""
        with self._lock:
            self._pause_flag = True
        logger.debug("Smoother paused")
    
    def resume(self, sync_to_current: bool = True) -> None:
        """
        恢复平滑
        
        参数:
            sync_to_current: 是否同步到当前实际位置
        """
        with self._lock:
            if sync_to_current:
                # 获取实际位置需要外部传入
                # 这里假设保持当前插值位置
                self._target_joints = self._current_joints.copy()
                self._joint_velocity = np.zeros(self._dof)
            self._pause_flag = False
        logger.debug("Smoother resumed")
    
    def get_motion_state(self) -> Dict[str, np.ndarray]:
        """获取完整运动状态"""
        with self._lock:
            return {
                'position': self._current_joints.copy(),
                'velocity': self._joint_velocity.copy(),
                'acceleration': self._joint_acceleration.copy()
            }
    
    def is_trajectory_finished(self, tolerance: float = 0.001) -> bool:
        """检查是否到达目标"""
        with self._lock:
            error = np.linalg.norm(self._target_joints - self._current_joints)
            velocity = np.linalg.norm(self._joint_velocity)
        return error < tolerance and velocity < tolerance
    
    def set_omega_n(self, omega_n: float) -> None:
        """
        动态调整自然频率
        
        参数:
            omega_n: 新的自然频率 (rad/s)
        """
        with self._lock:
            self._omega_n = np.clip(omega_n, 10.0, 50.0)
            logger.info(f"Omega_n updated to {self._omega_n:.1f} rad/s")
    
    def _control_loop(self) -> None:
        """控制循环主体"""
        dt = 1.0 / self._control_freq
        next_time = time.perf_counter()
        
        while self._is_running:
            loop_start = time.perf_counter()
            
            # 获取当前状态（线程安全）
            with self._lock:
                if self._pause_flag:
                    time.sleep(dt)
                    continue
                
                target = self._target_joints.copy()
                current = self._current_joints.copy()
                velocity = self._joint_velocity.copy()
                omega_n = self._omega_n
                zeta = self._zeta
            
            # 二阶系统动力学计算
            # ẍ = ωn²(x_target - x) - 2ζωn·ẋ
            error = target - current
            acceleration = omega_n**2 * error - 2 * zeta * omega_n * velocity
            
            # 欧拉积分
            velocity_new = velocity + acceleration * dt
            position_new = current + velocity_new * dt
            
            # 更新状态（线程安全）
            with self._lock:
                self._current_joints = position_new
                self._joint_velocity = velocity_new
                self._joint_acceleration = acceleration
            
            # 时间管理
            next_time += dt
            sleep_time = next_time - time.perf_counter()
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # 性能警告
                self._slow_loop_count += 1
                if self._slow_loop_count % 1000 == 1:
                    actual_dt = time.perf_counter() - loop_start
                    logger.warning(f"Smoother loop slow: {actual_dt*1000:.1f}ms "
                                 f"(target: {dt*1000:.1f}ms)")
                next_time = time.perf_counter()