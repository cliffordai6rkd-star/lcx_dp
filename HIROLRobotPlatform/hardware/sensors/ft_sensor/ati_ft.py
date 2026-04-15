# Try to import NetFT, fall back to mock if not available
try:
    from NetFT import Sensor as ATI
except (ImportError, ModuleNotFoundError):
    import glog as log
    log.warning("NetFT not available, using mock implementation")
    from hardware.mocks.mock_netft import Sensor as ATI

from hardware.base.ft import FTBase
import glog as log
import time, copy
import numpy as np

class AtiFt(FTBase):
    def __init__(self, config):
        super().__init__(config)
        self._ip = config["ip"]
        self._start_times = config.get("test_times", 3000)
        self._ati = ATI(self._ip)
        self._mean = np.zeros(6)
        self._is_initialized = self.initialize()
        
    def initialize(self):
        if self._is_initialized:
            return True
        
        log.info(f'ATI ft data {self._ip} successfully initialized!!!')
        self.update_bias()
        return True
    
    def update_bias(self):
        self._ati.stopStreaming()
        self._mean = np.zeros(6)
        self._ati.zero()
        for i in range(self._start_times):
            self._mean += np.array(self._ati.getMeasurement())
        self._mean /= self._start_times
        log.info(f'new mean value for the ft: {self._mean}')
        self._ati.startStreaming()
    
    def get_ft_data(self):
        if not self._is_initialized:
            return None, None
        
        ft_data = self._ati.measurement()
        ft_data = np.array(ft_data) - self._mean
        ft_data /= 1e6
        self._raw_value = ft_data.copy()
        dt = time.perf_counter() - self._time_stamp
        ft_data = self._lpf.update(ft_data, dt)
        self._time_stamp = time.perf_counter()
        # ft_data[:3] = [data / 1e6 for data in ft_data[:3]]
        # ft_data[3:] = [data / 1e6 for data in ft_data[3:]]
        return copy.deepcopy(ft_data), copy.deepcopy(self._time_stamp)
        
    def close(self):
        self._ati.stopStreaming()
        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    config = {"ip": "192.168.1.130", "test_times": 1500,
              "cutoff_frequency": 25}
    
    ati = AtiFt(config)
    
    # time latency test
    # i = 0
    # time.sleep(3)
    # max_z = 0
    # while True:
    #     print(f'i: {i}')
    #     if i % 60 == 0 and i != 0:
    #         ati.update_bias()
    #     i += 1
    #     start = time.perf_counter()
    #     ft_data, time_stamp = ati.get_ft_data()
    #     used_time = time.perf_counter() - start
    #     if max_z < ft_data[2]: max_z = ft_data[2]
    #     log.info(f'cur ft: {ft_data} type: {type(ft_data)}\n max: {max_z}, used time: {used_time}')
    #     time.sleep(0.1)
    
    # lpf testing
    dt = 0.01
    num_data = 150
    raw_values = np.array([]); filtered_data = np.array([])
    for i in range(num_data):
        cur_data, _ = ati.get_ft_data()
        cur_data = cur_data[None]
        cur_raw = ati.get_raw_value().copy()[None]
        raw_values = cur_raw if len(raw_values) == 0 else np.vstack((raw_values, cur_raw))
        filtered_data = cur_data if len(filtered_data) == 0 else np.vstack((filtered_data, cur_data))
        time.sleep(dt)
    log.info(f'raw shape: {raw_values.shape} data shape: {filtered_data.shape}')
    
    def plot_force_with_lpf(raw_values, filtered_data: np.ndarray, fs: float, labels=None, title=None):
        """data: (N,D)"""
        assert raw_values.shape == filtered_data.shape
        N, D = filtered_data.shape

        t = np.arange(N) 

        # 画图（每个维度一个 subplot）
        fig, axes = plt.subplots(D, 1, figsize=(10, 2.2*D), sharex=True)
        if D == 1:
            axes = [axes]
        if labels is None or len(labels) != D:
            labels = [f"ch{i}" for i in range(D)]

        for d, ax in enumerate(axes):
            ax.plot(t, filtered_data[:, d], label=f"lpf (fc={ati._lpf.cutoff_hz} Hz)",linestyle='-', lw=2.2, alpha=1.0,)
            ax.plot(t, raw_values[:, d], label="raw", linestyle='--', lw=1.0, marker='o', markersize=2)
            ax.set_ylabel(labels[d])
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

        axes[-1].set_xlabel("Time (s)")
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        plt.show()
    
    plot_force_with_lpf(raw_values, filtered_data, dt)
    