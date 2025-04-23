"""Gripper interface, model, type, and config.

Copyright 2021 Wenzhao Lian. All rights reserved.
"""

import abc
import numpy as np


class Gripper(metaclass=abc.ABCMeta):
  @abc.abstractmethod
  def open(self) -> bool:
    pass

  @abc.abstractmethod
  def close(self) -> bool:
    pass

  @abc.abstractmethod
  def move_to(self, position: np.ndarray) -> bool:
    pass

  @abc.abstractmethod
  def is_gripping(self) -> bool:
    pass
