from pika.gripper import Gripper
from pika.camera import RealSenseCamera, FisheyeCamera
from pika.sense import Sense

from hardware.tools.grippers.pika_gripper import PikaGripper
from hardware.sensors.cameras.pika_cameras import PikaCameras
from teleop.pika_tracker.pika_tracker import PikaTracker
from hardware.sensors.cameras.realsense_camera import RealsenseCamera
from hardware.sensors.cameras.opencv_camera import OpencvCamera
from simulation.mujoco.mujoco_sim import MujocoSim

import time, cv2
import numpy as np
from cfg_handling import get_cfg

# 导入并配置统一日志系统
from hardware.tools.grippers.log_config import setup_unified_logging

# 在程序开始时配置统一日志系统
setup_unified_logging(enable_python_logging=False)
# setup_unified_logging(level="INFO", enable_glog=True, enable_python_logging=False, disable_duplicate_handlers=True)

def pikasense2gripper():
    tracker_cfg = "teleop/pika_tracker/config/right_tracker_fr3_cfg.yaml"
    tracker_cfg = get_cfg(tracker_cfg)["pika_tracker"]
    tracker = PikaTracker(tracker_cfg)
    
    gripper_cfg = "hardware/tools/grippers/config/right_pika_gripper_cfg.yaml"
    gripper_cfg = get_cfg(gripper_cfg)["pika_gripper"]
    gripper = PikaGripper(gripper_cfg)
    
    while True:
        true, pose, gripper_data = tracker.parse_data_2_robot_target("absolute")
        print(f'gripper data from sense: {gripper_data}')
        gripper.set_tool_command(gripper_data["single"][0])
        time.sleep(0.006)

def test_pika_gripper_apis():
    gripper = Gripper()
    status = gripper.connect()
    print(f'Gripper connected: {status}')
    if not status:
        raise ValueError    
    status = gripper.enable()
    print(f'Gripper enabled: {status}')    
    if not status:
        raise ValueError  
    
    test_num = 50
    duration = 3
    test = 60
    posi = 0
    
    for i in range(test_num):
        temp = gripper.get_motor_temp()
        dist = gripper.get_gripper_distance()
        print(f'temp: {temp}, distance: {dist} command: {posi}')
        
        if i % duration == 0:
            if posi == test:
                posi = 0
            else: posi = test
            status = gripper.set_gripper_distance(posi)
            print(f'set the gripper displace: {status}')
        time.sleep(0.5)
            
    status = gripper.disable()
    print(f'Gripper disabled: {status}')    
    gripper.disconnect()

def test_pika_class_apis():
    gripper_cfg = "hardware/tools/grippers/config/left_pika_gripper_cfg.yaml"
    gripper_cfg = get_cfg(gripper_cfg)["pika_gripper"]
    gripper = PikaGripper(gripper_cfg)
    
    # camera_config = "hardware/sensors/cameras/config/right_pika_cameras.yaml"
    # camera_config = get_cfg(camera_config)
    # cameras = PikaCameras(camera_config)
    # if not cameras.initialize():
    #     raise ValueError
    # rs_config = get_cfg("hardware/sensors/cameras/config/left_pika_d405_cfg.yaml")["realsense_camera"]
    # pika_rs = RealsenseCamera(rs_config)
    # fisheye_config = get_cfg("hardware/sensors/cameras/config/left_pika_fisheye_cfg.yaml")["opencv_camera"]
    # pika_fisheye = OpencvCamera(fisheye_config)
    
    counter = 0
    test_value1 = 0.2
    test_value2 = 0.8
    posi = 1
    posi = 1
    while True:
        # rs_img = pika_rs.capture_all_data()["image"]
        # fisheye_img = pika_fisheye.capture_all_data()
        # fisheye_img = fisheye_img["image"]
        # cv2.imshow('realsense', rs_img)
        # cv2.imshow('fish eye', fisheye_img)
        # key = cv2.waitKey(1)

        # if key == ord('q'):
        #     print(f'quit!!!!')
        #     gripper.stop_tool()
        #     pika_rs.close()
        #     pika_fisheye.close()
        #     break
        
        gripper_state = gripper.get_tool_state()
        if counter % 10:
            # if posi == test_value1:
            #     posi = test_value2
            # else: posi = test_value1
            posi -= 0.05 
            gripper.set_tool_command(posi)
            time.sleep(0.2)
        # print(f'{gripper_state._time_stamp} gripper state, posi: {gripper_state._position}, is grasped: {gripper_state._is_grasped} diff: {gripper_state._position - np.clip(posi, 0, 1)*90 }  command: {posi*90}mm')
            
        counter += 1
        time.sleep(0.004)

def test_pika_tracker():
    tracker_cfg = "teleop/pika_tracker/config/left_tracker_fr3_cfg.yaml"
    tracker_cfg = get_cfg(tracker_cfg)["pika_tracker"]
    tracker = PikaTracker(tracker_cfg)
    
    sim_cfg = "simulation/config/mujoco_duo_fr3.yaml"
    mujoco_cfg = get_cfg(sim_cfg)["mujoco"]
    sim = MujocoSim(mujoco_cfg)
    # world2base, base2world = get_sim_base_world_transform(mujoco)
    target_site = ["targetL", "targetR"]
    tcp_mocaps = ["TCPL", "TCPR"]
    
    while True:
        print(f'before read')
        succ, pose, tool = tracker.parse_data_2_robot_target("absolute_delta")
        
        if succ:
            right_pose = pose["single"]
            print(f'succ: {succ}, pose: {pose}, tool: {tool}')
            cur_target_mocap = target_site[1]
            sim.set_target_mocap_pose(cur_target_mocap, right_pose)
        time.sleep(0.001)

if __name__ == "__main__":
    # test_pika_gripper_apis()
    # test_pika_class_apis()
    # test_pika_tracker()
    pikasense2gripper()
    
