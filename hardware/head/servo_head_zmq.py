from hardware.communication.servo_pika_img_interface import G1UmiClient
import warnings, threading, time, copy
import numpy as np
import glog as log

class ZmqDynamixelHead:
    def __init__(self, config):
        self._config = config
        self._server_ip = config["ip"]
        self._ctrl_port = config.get("port", 5555)
        self._update_frequency = config.get("frequency", 50)
        
        self._zmq_interface = G1UmiClient(self._server_ip,
            self._ctrl_port, img_endpoint=None, require_control=True)
        self._curr_positions = None; self._time_stamp = None
        self._lock = threading.Lock()
        self._state_updated = False
        self._thread_running = True
        self._thread_handler = threading.Thread(target=self._update_loop, daemon=True)
        self._head_angles = np.zeros(3)
        
        self._is_initialized = self.initialize()
        log.info(f'Started ZMQ dynamixel head update loop!')
        
    def initialize(self):
        if self._is_initialized:
            return True
        
        if not self._thread_handler.is_alive():
            self._thread_handler.start()
            while not self._state_updated:
                time.sleep(0.001)
                
        log.info("ZMQ dynamixel head is successfully initialized!!!!")
        
    def set_head_command(self, command):
        if not self._is_initialized:
            log.warn("ZMQ dynamixel head not initialized for setting!")
            return 
        
        assert len(command) == 3, f"ZMQ dynamixel hand only contains three values!"
        self._zmq_interface.set_neck_positions(command)
    
    def get_head_positions(self):
        if not self._is_initialized:
            log.warn("ZMQ dynamixel head not initialized for setting!")
            return None, None
        
        with self._lock:
            cur_positions = copy.deepcopy(self._zmq_interface.get_neck_positions())
            time_stamp = copy.deepcopy(self._time_stamp)
        return cur_positions, time_stamp
    
    def _update_loop(self):
        log.info(f'Started ZMQ dynamixel head update loop!')
        
        expected_dt = 1.0 / self._update_frequency
        last_read_time = time.perf_counter()
        while self._thread_running:
            rpy = self._zmq_interface.get_neck_positions()
            with self._lock:
                self._curr_positions = copy.deepcopy(rpy)
                self._time_stamp = time.perf_counter()
            self._state_updated = True
            
            dt = time.perf_counter() - last_read_time
            last_read_time = time.perf_counter()
            if dt < expected_dt:
                sleep_time = expected_dt - dt
                time.sleep(0.95*sleep_time)
            elif dt > 1.2*expected_dt:
                warnings.warn(f'ZMQ dynamixel head could not reach the {self._update_frequency}hz, actual freq: {1.0/dt:.2f}hz')
            
        log.info("Started ZMQ dynamixel head update loop!")
        
    def close(self):
        self._thread_running = False
        if self._thread_handler.is_alive():
            self._thread_handler.join()
        self._zmq_interface.close()
        log.info("Closde ZMQ dynamixel head!!")