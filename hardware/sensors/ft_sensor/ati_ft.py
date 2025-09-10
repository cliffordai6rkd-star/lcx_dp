from NetFT import Sensor as ATI
from hardware.base.ft import FTBase
import glog as log
import time

class AtiFt(FTBase):
    def __init__(self, config):
        super().__init__(config)
        self._ip = config["ip"]
        self._start_times = config.get("test_times", 3000)
        self._ati = ATI(self._ip)
        self._is_initialized = self.initialize()
        
    def initialize(self):
        if self._is_initialized:
            return True
        
        log.info(f'ATI ft data {self._ip} successfully initialized!!!')
        self._ati.tare()
        log.info(f'mean value for the ft: {self._ati.mean}')
        self._ati.startStreaming()
        return True
    
    def get_ft_data(self):
        if not self._is_initialized:
            return None
        
        ft_data = self._ati.measurement()
        ft_data[:3] = [data / 1e6 for data in ft_data[:3]]
        ft_data[3:] = [data / 1e6 for data in ft_data[3:]]
        return ft_data
        
    def close(self):
        self._ati.stopStreaming()
        
if __name__ == '__main__':
    config = {"ip": "192.168.1.1", "test_times": 5000}
    
    ati = AtiFt(config)
    i = 0
    time.sleep(3)
    while True:
        print(f'i: {i}')
        i += 1
        ft_data = ati.get_ft_data()
        log.info(f'cur ft: {ft_data}')
        time.sleep(0.1)
        