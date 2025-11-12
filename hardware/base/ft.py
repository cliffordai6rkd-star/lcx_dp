import abc, threading, copy, time, os, json
import glog as log
import numpy as np
from hardware.base.lpf import LowPassFilter

class FTBase(abc.ABC, metaclass=abc.ABCMeta):
    def __init__(self, config):
        self._config = config
        self._thread_lock = threading.Lock()
        self._zero_offset = np.zeros(6)
        self._raw_value = np.zeros(6)
        self._ft_data = np.zeros(6)
        self._time_stamp = time.perf_counter()
        self._update_frequency = config.get("frequency", 300)
        self._is_initialized = False
        self._aync_save = config.get('async_save', False)
        if self._aync_save:
            self._save_freq = config.get('save_freq', 200)
            self._ready_save = False
            self._aync_save_thread = None
        cutoff_freq = config.get(f'cutoff_frequency', 40)
        self._lpf = LowPassFilter(cutoff_freq)
        
    @abc.abstractmethod
    def initialize(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def close(self):
        raise NotImplementedError
    
    def get_ft_data(self):
        self._thread_lock.acquire()
        ft_data = copy.copy(self._ft_data)
        time_stamp = copy.copy(self._time_stamp)
        self._thread_lock.release()
        return ft_data, time_stamp
    
    def get_raw_value(self):
        return self._raw_value
    
    def save_ft_data(self, save_dir , key):
        if not  self._aync_save:
            raise ValueError(f'The ft sensor is not enabled for aysnc save mode!!!!')
        
        self.write_data(warnig_msg="Late save for the FT data!!!!")
            
        self._aync_save_thread = threading.Thread(
            target=self._async_save_loop, args=(save_dir, key,))
        self._ready_save = False
        self._aync_save_thread.start()
        
    def write_data(self, warnig_msg=None):
        if not  self._aync_save:
            raise ValueError(f'The ft sensor is not enabled for aysnc save mode!!!!')
        
        if self._aync_save_thread and self._aync_save_thread.is_alive():
            if warnig_msg: log.warn(warnig_msg)
            self._ready_save = True
            self._aync_save_thread.join()
        self._aync_save_thread = None
    
    def _async_save_loop(self, save_dir, key):
        if not os.path.exists(save_dir):
            log.error(f'Could not find the {save_dir} for async save the ft data')
            return False
        else: save_path = os.path.join(save_dir, f"{key}_data.json")
        
        data_list = []
        id = 0
        while not self._ready_save:
            start = time.perf_counter()
            
            ft_data, ts = self.get_ft_data()
            if not isinstance(ft_data, list): ft_data = ft_data.tolist()
            data_list.append(dict(id=id, ft_data=ft_data, time_stamp=ts))
            id += 1
            
            duration = time.perf_counter() - start
            start = time.perf_counter()
            if duration < 1.0 / self._save_freq:
                sleep_time = (1.0 / self._save_freq) - duration
                time.sleep(sleep_time)
            elif duration > 1.2 / self._save_freq:
                log.warn(f'async save ft sensor freq {1.0 / duration}Hz')
        
        log.info(f'Ready to save the async ft data to {save_path}')
        data_dict = {"data": data_list}
        with open(save_path, 'w', encoding='utf-8') as jsonf:
            jsonf.write(json.dumps(data_dict, indent=4, ensure_ascii=False))
        log.info(f"{'=='*4}Finished to save the async {key} ft data to {save_path}{'=='*4}")
        