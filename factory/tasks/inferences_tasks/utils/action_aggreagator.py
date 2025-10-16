from enum import Enum
import numpy as np
import glog as log
from dataset.utils import ActionType

class WeightMode(Enum):
    UNIFORM = "uniform"
    EXPONENTIAL = "exp_by_age"
    TRIANGULAR =  "triangular_by_offset"
    NO_WEIGHT = "no"

class ActionAggregator:
    def __init__(self, query_frequency, chunk_size, max_timestamps, action_size, k):
        """
            one way: k [0, 1]: weight size, if k is larger, the weight will prefer old action
            another way: k [-1.0 0]: if |k| is larger, the weight will prefer new action
        """
        self._query_frequency = query_frequency
        self._chunk_size = chunk_size
        self._max_timestamps = max_timestamps
        self._action_size = action_size
        self._all_time_action = np.zeros((self._max_timestamps, 
            self._max_timestamps + self._chunk_size, action_size), dtype=np.float32)
        # log.info(f'all time action shape: {self._all_time_action.shape}')
        self._k = k

    def add_action_chunk(self, t, new_action_chunk):
        self._all_time_action[[t], t:t+self._chunk_size] = new_action_chunk
        self._all_action = new_action_chunk
    
    def aggregation_action(self, t, weight_mode):
        if weight_mode != WeightMode.NO_WEIGHT:
            action_for_cur_step = self._all_time_action[:, t]
            # log.info(f'action for cur step: {action_for_cur_step.shape}')
            actions_populated = np.all(action_for_cur_step != 0, axis=1)
            # log.info(f'action populated: {action_for_cur_step.shape}')
            action_for_cur_step = action_for_cur_step[actions_populated]
            # log.info(f'action for cur step: {action_for_cur_step.shape}')
            exp_weights = np.exp(-self._k * np.arange(len(action_for_cur_step)))
            # log.info(f'exp weight: {exp_weights.shape}')
            exp_weights = exp_weights / exp_weights.sum()
            # log.info(f'exp weight: {exp_weights.shape}')
            exp_weights = exp_weights[:, None, ...]
            # log.info(f'exp weight: {exp_weights.shape}')
            aggregated_action = (action_for_cur_step * exp_weights).sum(axis=0, keepdims=True)
            # log.info(f'aggregated action: {aggregated_action.shape}')
            aggregated_action = aggregated_action[0]
        else:
            aggregated_action = self._all_action[t % self._query_frequency]
        return aggregated_action
    
    def reset(self):
        self._all_time_action = np.zeros((self._max_timestamps, 
            self._max_timestamps + self._chunk_size, self._action_size), 
                                         dtype=np.float32)
    