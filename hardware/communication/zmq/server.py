import zmq
import glog as log
import json, time, threading

class ZmqServer:
    def __init__(self, cfg):
        self._cfg = cfg
        self._ip = cfg["ip"]
        self._port = cfg["port"]
        self._pub_frequency = cfg["frequency"]
        self._all_data = {}
        self._data_updated = False
        self._thread = None
        self._thread_running = True
        self._thread_started = False
        self._data_lock = threading.Lock()
        self._is_initialized = False
        self._is_initialized = self.initialize()
        
    def initialize(self):
        if self._is_initialized:
            return True
        
        end_point = f"tcp://{self._ip}:{self._port}"
        self._ctx = zmq.Context.instance()
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.bind(end_point)
        
        self._thread = threading.Thread(target=self.server_pub_loop, daemon=True)
        self._thread.start()
        while not self._thread_started:
            time.sleep(0.001)
        log.info(f'Successfully create the zmq socket pub with endpoint {end_point}')
        return True
    
    def update_data(self, topic, data):
        """
            data must be a int/flota/list/dict[list]
        """
        with self._data_lock:
            self._all_data[topic] = data
            self._data_updated = True
    
    def server_pub_loop(self):
        log.info(f"Started zmp server publish with {self._ip} {self._port}!!!")
        dt = 1.0 / self._pub_frequency
        while self._thread_running:
            if not self._thread_started: self._thread_started = True
            start_time = time.perf_counter()
            
            with self._data_lock:
                if self._data_updated:
                    json_data = json.dumps(self._all_data).encode("utf-8")
                    # self._pub.send_multipart(json_data)
                    self._pub.send(json_data)
                    self._data_updated = False
            
            used_time = time.perf_counter() - start_time
            sleep_time = dt
            if used_time < dt:
                sleep_time = dt - used_time
            else:
                log.warn(f'ZMQ server pub frequency slow, expected: {self._pub_frequency}, actual: {1.0 / used_time}')
            time.sleep(sleep_time)
            
        log.info(f'Stopped the ZMQ server publish with {self._ip} {self._port}')
        
    def close(self):
        self._thread_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join()
        self._pub.close()
        self._ctx.destroy()
        log.info(f'ZMQ server publish with {self._ip} {self._port} is already closed!!!')

if __name__ == "__main__":
    from pynput import keyboard
    import random
    class KeyMonitor:
        def __init__(self):
            self.should_exit = False
            
        def on_press(self, key):
            try:
                if key.char == 'q':
                    self.should_exit = True
                    print("检测到 'q' 键，程序将退出")
                    return False  # 停止监听器
            except AttributeError:
                pass
        
        def start_monitoring(self):
            # 在单独的线程中启动监听器
            listener = keyboard.Listener(on_press=self.on_press)
            listener.start()
            return listener
        
        @property
        def should_quit(self):
            return self.should_exit

    cfg = {"ip": "0.0.0.0", "port": 5556, "frequency": 1000}
    zmq_server = ZmqServer(cfg)
    monitor = KeyMonitor()
    listener = monitor.start_monitoring()
    possible_topics = ["data_0", "data_1", "data_2", "data_3", "data_4"]
    
    while True:
        if monitor.should_quit:
            zmq_server.close()
            break
        topic = random.randint(0, len(possible_topics) - 1)
        topic = possible_topics[topic]
        data = random.random()
        log.info(f'update {topic} with data {data}')
        zmq_server.update_data(topic, data)
        log.info(f'cur all data: {zmq_server._all_data}')
        time.sleep(2)
        