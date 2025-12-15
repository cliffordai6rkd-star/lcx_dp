from hardware.communication.servo_pika_img_interface import G1UmiClient
from hardware.base.camera import CameraBase
import warnings, threading, time, copy
import numpy as np
import glog as log

class ZmqImgSubscriber(CameraBase):
    def __init__(self, config):
        self._server_ip = config['ip']
        self._cam_name = config["cam_name"]
        self._img_port = config.get("port", 5556)
        
        self._zmq_interface = G1UmiClient(self._server_ip,
            ctrl_endpoint=None, img_endpoint=self._img_port, require_control=False)
        self._thread_running = True
        self._thread_handler = None
        super().__init__(config)
        
    def initialize(self):
        if self._is_initialized:
            return True
        
        # this camera do not support imu & depth
        if self._contain_depth or self._contain_imu:
            raise ValueError("Zmq Image subscriber")
        
        test_num = 0; succ_get_img = False; get_cam_name = None        
        for cam_name, img, meta in self._zmq_interface.subscribe_images([self._cam_name]):
            if img is not None: 
                success_get_img = True
            else: get_cam_name = cam_name
            
            if test_num > 5:
                break
            test_num += 1
        if not succ_get_img or get_cam_name != self._cam_name:
            raise ValueError(f"Could not get the correct cam image from zmq {succ_get_img}, {get_cam_name}")            
        
        # thread
        self._thread_handler = threading.Thread(target=self.update_camera_thread)
        self._thread_handler.start()
        
        while self._image_data is None:
            time.sleep(0.001)
        
        log.info(f'ZMQ camera subscriber with {self._cam_name} is ok to retrive data!!!')
        return True
        
    def update_camera_thread(self):
        print(f'ZMQ camera thread started!!!')
        
        last_read_time = time.perf_counter()
        for cam_name, img, meta in self._zmq_interface.subscribe_images([self._cam_name]):
            if not self._thread_running:
                break
            
            if img is not None:
                with self._lock:
                    self._image_data = copy.deepcopy(img)
                    self._time_stamp = time.perf_counter()

            dt = time.perf_counter() - last_read_time
            last_read_time = time.perf_counter()
            if dt < (1.0 / self._fps):
                sleep_time = (1.0 / self._fps) - dt
                time.sleep(0.95*sleep_time)
            elif dt > 1.35 / self._fps:
                warnings.warn(f'Camera could not reach the {self._fps}hz, '
                              f'actual freq: {1.0/dt:.2f}hz')
        log.info(f'ZMQ camera {self._cam_name} thread is successfully stopped!')
            
    def close(self):
        self._thread_running = False
        if self._thread_handler is not None:
            self._thread_handler.join()
        self._zmq_interface.close()
        log.info(f'ZMQ   camera {self._cam_name} successfully closed!!')
        