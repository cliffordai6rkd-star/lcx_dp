import cv2
from hardware.base.camera import CameraBase
import os, threading, time, copy, re
import numpy as np
import glog as log

class OpencvCamera(CameraBase):
    def __init__(self, config):
        """
            image_shape: [height, width]
            device_id: /dev/video*
        """
        self._device_id = config['device_id']
        
        self._thread_running = True
        self._thread_handler = None
        super().__init__(config)
        
    def initialize(self):
        if self._is_initialized:
            return True
        
        # this camera do not support imu & depth
        if self._contain_depth or self._contain_imu:
            raise ValueError("Opencv camera do not support imu and depth")
        
        self._cap = cv2.VideoCapture(self._device_id)

        # Test if the camera can read frames
        opened = self._cap.isOpened()
        log.info(f'opencv camera {self._device_id} opened or not {opened}')
        success, _ = self._cap.read()
        if not success:
            real_device_id = self.get_real_video_id(self._device_id)
            possible_ids = []
            if real_device_id % 2 != 0:
                possible_ids = [real_device_id+1, real_device_id-1]
            else: possible_ids.append(real_device_id)
            for possible_id in possible_ids:
                device_id = "/dev/video"+str(possible_id)
                self._cap = cv2.VideoCapture(device_id)
                opened = self._cap.isOpened()
                log.info(f'opencv camera for other id {device_id} opened or not {opened}')
                if opened:
                    success = True; self._device_id = device_id
                    break  
                
        if not success:
            self.close()
            return False
        
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._img_shape[0])
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._img_shape[1])
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        # thread
        self._thread_handler = threading.Thread(target=self.update_camera_thread)
        self._thread_handler.start()
        
        while self._image_data is None:
            time.sleep(0.001)
        
        log.info(f'Opencv camera with {self._device_id} is ok to retrive data!!!')
        return True
        
    def update_camera_thread(self):
        log.info(f'opencv camera thread started!!!')
        
        last_read_time = time.perf_counter()
        counter = 0
        while self._thread_running:
            # frame reading
            start0 = time.perf_counter()
            ret, color_image = self._cap.read()
            read_time = time.perf_counter() - start0

            # start = time.perf_counter()
            if ret:
                self._lock.acquire()
                self._image_data = copy.deepcopy(color_image)
                self._time_stamp = time.perf_counter()
                self._lock.release()
            # ret_process = time.perf_counter() - start

            # start = time.perf_counter()
            dt = time.perf_counter() - last_read_time
            last_read_time = time.perf_counter()
            # dt_procs = time.perf_counter() - start
            if dt < (1.0 / self._fps):
                sleep_time = (1.0 / self._fps) - dt
                time.sleep(0.92*sleep_time)
            elif dt > 1.35 / self._fps:
                counter += 1
                # total = time.perf_counter() - start0
                if counter % 500 == 0:
                    log.warn(f'Camera could not reach the {self._fps}hz, '
                                f'actual freq: {1.0 / dt}hz, read time: {1.0/read_time}Hz')
                    counter = 0
                # log.warn(f'percentage: {read_time/dt*100} {ret_process/dt*100} {dt_procs/dt*100}')
           
        log.info(f'Opencv camera {self._device_id} thread is successfully stopped!')
    
    def get_real_video_id(self, dev_path: str) -> int:
        """
            dev_path: /dev/video80 
            return: video id(int)
            eg. realpath(/dev/video80) == /dev/video12 -> return 12
        """
        real = os.path.realpath(dev_path)  
        m = re.search(r"/dev/video(\d+)$", real)
        if not m:
            raise ValueError(f"Resolved path is not a /dev/video*: {real}")
        return int(m.group(1))

    def close(self):
        self._thread_running = False
        if self._thread_handler is not None:
            self._thread_handler.join()
        if self._cap.isOpened():
            self._cap.release()
        log.info(f'Opencv camera {self._device_id} successfully closed!!')
        
if __name__ == "__main__":
    config = ""
    