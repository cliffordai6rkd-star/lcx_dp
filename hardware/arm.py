import abc
import numpy as np
from typing import Sequence, Union

from data_types import se3


class Arm(metaclass=abc.ABCMeta):
  @abc.abstractmethod
  def set_jp(self, jp: Union[Sequence, np.ndarray]) -> None:
    pass

  @abc.abstractmethod
  def set_jv(self, jv: Union[Sequence, np.ndarray]) -> None:
    pass

  @abc.abstractmethod
  def get_jp(self) -> np.ndarray:
    pass

  @abc.abstractmethod
  def get_jv(self) -> np.ndarray:
    pass

  @abc.abstractmethod
  def get_flange_pose(self) -> se3.Transform:
    pass

  @abc.abstractmethod
  def hold_joints(self) -> None:
    """Holds current joint position with zero velocity.
    """
    pass

  def interactive_control(self) -> None:
    """Interactively control the robot, i.e., kinethetic teaching."""
    raise NotImplementedError
