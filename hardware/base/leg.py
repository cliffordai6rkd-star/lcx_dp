import abc
import numpy as np
from typing import Sequence, Union

class LegBase(metaclass=abc.ABCMeta):
  
  @abc.abstractmethod
  def get_model(self):
      pass
  
  @abc.abstractmethod
  def get_state(self):
      pass

 
