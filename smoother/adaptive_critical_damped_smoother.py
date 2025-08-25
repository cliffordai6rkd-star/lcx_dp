"""
Adaptive Critical Damped Smoother
Dynamically adjusts response based on tracking error
"""

import numpy as np
import time
import logging
from typing import Dict, Any

from .critical_damped_smoother import CriticalDampedSmoother

logger = logging.getLogger(__name__)


class AdaptiveCriticalDampedSmoother(CriticalDampedSmoother):
    """自适应二阶临界阻尼平滑器"""
    
    def __init__(self, config: Dict[str, Any], dof: int):
        """
        初始化自适应平滑器
        
        额外参数:
            omega_n_min: 最小自然频率（柔顺）
            omega_n_max: 最大自然频率（快速）
            error_thresholds: 误差阈值字典
            transition: 过渡类型 'linear' 或 'sigmoid'
        """
        super().__init__(config, dof)
        
        # 自适应参数
        self._omega_n_min = config.get("omega_n_min", 15.0)
        self._omega_n_max = config.get("omega_n_max", 40.0)
        
        thresholds = config.get("error_thresholds", {})
        self._error_low = thresholds.get("low", 0.01)   # rad
        self._error_high = thresholds.get("high", 0.1)  # rad
        
        self._transition = config.get("transition", "linear")
        
        # 平滑omega变化
        self._omega_change_limit = config.get("omega_change_limit", 5.0)  # rad/s per update
        self._last_omega = self._omega_n
        
        # 性能指标
        self._adaptation_count = 0
        
        logger.info(f"AdaptiveSmoother: ωn∈[{self._omega_n_min:.1f}, {self._omega_n_max:.1f}], "
                   f"error∈[{self._error_low:.3f}, {self._error_high:.3f}]")
    
    def _compute_adaptive_omega(self, error: np.ndarray) -> float:
        """
        根据误差计算自适应omega_n
        
        参数:
            error: 位置误差向量
        返回:
            自适应的omega_n值
        """
        error_norm = np.linalg.norm(error)
        
        if error_norm <= self._error_low:
            # 小误差：柔顺模式
            omega_n = self._omega_n_min
            
        elif error_norm >= self._error_high:
            # 大误差：快速模式
            omega_n = self._omega_n_max
            
        else:
            # 中间区域：插值
            if self._transition == "linear":
                # 线性插值
                ratio = (error_norm - self._error_low) / (self._error_high - self._error_low)
                omega_n = self._omega_n_min + ratio * (self._omega_n_max - self._omega_n_min)
                
            elif self._transition == "sigmoid":
                # S型曲线插值（更平滑）
                x = (error_norm - self._error_low) / (self._error_high - self._error_low)
                # sigmoid: f(x) = 1 / (1 + exp(-k*(x-0.5)))
                k = 10  # 陡度参数
                ratio = 1 / (1 + np.exp(-k * (x - 0.5)))
                omega_n = self._omega_n_min + ratio * (self._omega_n_max - self._omega_n_min)
            else:
                omega_n = self._omega_n
        
        # 平滑omega变化（避免突变）
        if hasattr(self, '_last_omega'):
            omega_diff = omega_n - self._last_omega
            if abs(omega_diff) > self._omega_change_limit:
                omega_n = self._last_omega + np.sign(omega_diff) * self._omega_change_limit
        
        self._last_omega = omega_n
        return omega_n
    
    def _control_loop(self) -> None:
        """重写控制循环，加入自适应逻辑"""
        dt = 1.0 / self._control_freq
        next_time = time.perf_counter()
        
        # 初始化
        log_counter = 0
        
        while self._is_running:
            loop_start = time.perf_counter()
            
            # 获取状态
            with self._lock:
                if self._pause_flag:
                    time.sleep(dt)
                    continue
                
                target = self._target_joints.copy()
                current = self._current_joints.copy()
                velocity = self._joint_velocity.copy()
                zeta = self._zeta
            
            # 计算误差和自适应omega
            error = target - current
            omega_n = self._compute_adaptive_omega(error)
            
            # 二阶系统动力学（使用自适应omega_n）
            acceleration = omega_n**2 * error - 2 * zeta * omega_n * velocity
            
            # 欧拉积分
            velocity_new = velocity + acceleration * dt
            position_new = current + velocity_new * dt
            
            # 更新状态
            with self._lock:
                self._current_joints = position_new
                self._joint_velocity = velocity_new
                self._joint_acceleration = acceleration
                self._omega_n = omega_n  # 记录当前omega
            
            # 定期日志（每秒一次）
            log_counter += 1
            if log_counter >= self._control_freq:
                log_counter = 0
                error_norm = np.linalg.norm(error)
                logger.debug(f"Adaptive: error={error_norm:.4f}, ωn={omega_n:.1f}")
                self._adaptation_count += 1
            
            # 时间管理
            next_time += dt
            sleep_time = next_time - time.perf_counter()
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                self._slow_loop_count += 1
                if self._slow_loop_count % 1000 == 1:
                    actual_dt = time.perf_counter() - loop_start
                    logger.warning(f"Adaptive loop slow: {actual_dt*1000:.1f}ms")
                next_time = time.perf_counter()
    
    def get_adaptive_state(self) -> Dict[str, float]:
        """获取自适应状态信息"""
        with self._lock:
            error_norm = np.linalg.norm(self._target_joints - self._current_joints)
            velocity_norm = np.linalg.norm(self._joint_velocity)
            
            # 计算当前模式
            if error_norm <= self._error_low:
                mode = "smooth"
            elif error_norm >= self._error_high:
                mode = "fast"
            else:
                mode = "adaptive"
            
            return {
                'current_omega_n': self._omega_n,
                'error_norm': error_norm,
                'velocity_norm': velocity_norm,
                'mode': mode,
                'settling_time': 4.6 / self._omega_n,
                'adaptation_count': self._adaptation_count
            }
    
    def reset_adaptation_stats(self) -> None:
        """重置自适应统计信息"""
        with self._lock:
            self._adaptation_count = 0
            self._last_omega = self._omega_n
        logger.info("Adaptation statistics reset")