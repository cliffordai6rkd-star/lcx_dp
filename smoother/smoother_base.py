"""
Smoother Base Class
Provides abstract interface for joint command smoothers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np


class SmootherBase(ABC):
    """
    关节命令平滑器基类
    支持二阶系统、Ruckig等多种平滑算法
    """
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any], dof: int):
        """
        初始化平滑器
        
        参数:
            config: 平滑器配置字典
            dof: 机器人自由度
        """
        self._config = config
        self._dof = dof
        self._is_running = False
    
    @abstractmethod
    def start(self, initial_positions: np.ndarray) -> None:
        """
        启动平滑器
        
        参数:
            initial_positions: shape=(dof,) 初始关节位置
        异常:
            RuntimeError: 平滑器已在运行
        """
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """停止平滑器线程"""
        pass
    
    @abstractmethod
    def update_target(self, joint_target: np.ndarray, immediate: bool = False) -> None:
        """
        更新目标位置
        
        参数:
            joint_target: shape=(dof,) 目标关节角度
            immediate: 是否立即跳转（用于reset场景）
        返回:
            None（非阻塞）
        """
        pass
    
    @abstractmethod
    def get_command(self) -> Tuple[np.ndarray, bool]:
        """
        获取平滑后的命令
        
        返回:
            (joint_positions, is_active): 关节位置和激活状态
        """
        pass
    
    @abstractmethod
    def pause(self) -> None:
        """暂停平滑（保持当前输出）"""
        pass
    
    @abstractmethod
    def resume(self, sync_to_current: bool = True) -> None:
        """
        恢复平滑
        
        参数:
            sync_to_current: 是否同步到当前实际位置
        """
        pass
    
    # === Ruckig相关接口（可选实现） ===
    
    def set_velocity_limits(self, max_velocity: np.ndarray) -> None:
        """
        设置速度限制（Ruckig需要）
        
        参数:
            max_velocity: shape=(dof,) 最大速度 (rad/s)
        """
        pass
    
    def set_acceleration_limits(self, max_acceleration: np.ndarray) -> None:
        """
        设置加速度限制（Ruckig需要）
        
        参数:
            max_acceleration: shape=(dof,) 最大加速度 (rad/s²)
        """
        pass
    
    def set_jerk_limits(self, max_jerk: np.ndarray) -> None:
        """
        设置急动度限制（Ruckig特有）
        
        参数:
            max_jerk: shape=(dof,) 最大急动度 (rad/s³)
        """
        pass
    
    def get_motion_state(self) -> Dict[str, np.ndarray]:
        """
        获取完整运动状态
        
        返回:
            包含position, velocity, acceleration的字典
        """
        return {
            'position': np.zeros(self._dof),
            'velocity': np.zeros(self._dof),
            'acceleration': np.zeros(self._dof)
        }
    
    def get_expected_duration(self) -> float:
        """
        获取预期到达时间（轨迹规划用）
        
        返回:
            预期时间 (秒)
        """
        return 0.0
    
    def is_trajectory_finished(self, tolerance: float = 0.001) -> bool:
        """
        检查是否到达目标
        
        参数:
            tolerance: 位置容差 (rad)
        返回:
            是否完成
        """
        return True