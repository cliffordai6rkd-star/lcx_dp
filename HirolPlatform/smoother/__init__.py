"""
Smoother Module
Provides joint command smoothing for robotic control
"""

from .smoother_base import SmootherBase
from .critical_damped_smoother import CriticalDampedSmoother
from .adaptive_critical_damped_smoother import AdaptiveCriticalDampedSmoother

__all__ = [
    'SmootherBase',
    'CriticalDampedSmoother', 
    'AdaptiveCriticalDampedSmoother'
]

# Version info
__version__ = '1.0.0'