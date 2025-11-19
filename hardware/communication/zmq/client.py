import zmq, copy
import json, time, threading
import glog as log

class ZmqClient:
    def __init__(self, cfg):
        self._cfg = cfg
        self._ip = cfg["ip"]
        self._port = cfg["port"]
        self._sub_frequency = cfg["frequency"]
        self._all_data = {}
        self._data_lock = threading.Lock()
        self._thread_started = False
        self._thread_running = True
        self._is_initialized = False
        self._is_initialized = self.initialize()
        
    def initialize(self):
        if self._is_initialized:
            return True
        
        end_point = f"tcp://{self._ip}:{self._port}"
        self._ctx = zmq.Context.instance()
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.connect(end_point)
        self._sub.setsockopt(zmq.SUBSCRIBE, b"")
        # recieve timeout is 100ms
        self._sub.setsockopt(zmq.RCVTIMEO, 100) 
        
        self._thread = threading.Thread(target=self.update_data_loop, daemon=True)
        self._thread.start()
        while not self._thread_started:
            time.sleep(0.001)
    
        log.info(f'Successfully create the zmq socket client with endpoint {end_point}')
        return True
    
    def update_data_loop(self):
        log.info(f"Started zmp client sunscriber with {self._ip} {self._port}!!!")
        dt = 1.0 / self._sub_frequency
        while self._thread_running:
            if not self._thread_started: self._thread_started = True
            start_time = time.perf_counter()
           
            try:
                json_data = self._sub.recv()  
                # json_data = self._sub.recv_multipart()
            except zmq.error.Again:
                # timeout
                continue
            except zmq.error.ZMQError as e:
                # error break
                log.error(f"recv error: {e}")
                break
            
            data = json.loads(json_data.decode("utf-8"))
            with self._data_lock:
                self._all_data = data
                
            used_time = time.perf_counter() - start_time
            sleep_time = dt
            if used_time < dt:
                sleep_time = dt - used_time
            else:
                log.warn(f'ZMQ server pub frequency slow, expected: {self._sub_frequency}, actual: {1.0 / used_time}')
            time.sleep(sleep_time)
        
        log.info(f'Stopped the ZMQ client sub with {self._ip} {self._port}')
         
    def get_topic_data(self, topic):
        if topic in self._all_data:
            with self._data_lock:
                data = copy.deepcopy(self._all_data[topic])
            return data
        else:
            return None
    
    def close(self):
        self._thread_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join()
        self._sub.close()
        self._ctx.term()
        log.info(f'ZMQ clinet subscriber with {self._ip} {self._port} is already closed!!!')
        
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
    zmq_client = ZmqClient(cfg)
    possible_topics = ["data_0", "data_1", "data_2", "data_3", "data_4"]
    monitor = KeyMonitor()
    listener = monitor.start_monitoring()
    
    while True:
        if monitor.should_quit:
            zmq_client.close()
            break
        
        for topic in possible_topics:
            data = zmq_client.get_topic_data(topic)
            # log.info(f'{topic}: {data}')
            if data:
                log.info(f'{topic}: {data}')
        time.sleep(0.01)
        