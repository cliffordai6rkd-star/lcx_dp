import abc
import numpy as np
from typing import Sequence, Union

from data_types import se3

from panda_py.constants import *

class ArmBase(metaclass=abc.ABCMeta):
  
  @abc.abstractmethod
  def get_model(self):
      pass
  
  @abc.abstractmethod
  def get_ee_orientation(self):
      pass
  
  @abc.abstractmethod
  def get_ee_pose(self):
      pass

  @abc.abstractmethod
  def get_ee_position(self):
      pass

  @abc.abstractmethod
  def get_state(self):
      pass

  def get_joint_position_start(self):
      return JOINT_POSITION_START

  def get_joint_limits_lower(self):
      return JOINT_LIMITS_LOWER
  
  def get_joint_limits_upper(self):
      return JOINT_LIMITS_UPPER
