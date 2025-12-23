"""
    Contributions to the unitree robotics xr_teleoperation projects
    with modification by ZYX
"""

import os
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import rerun as rr
import rerun.blueprint as rrb
from dataset.lerobot.reader import RerunEpisodeReader, ActionType, ObservationType
from dataset.lerobot.rerun_visualizer import RerunLogger
from datetime import datetime
import logging_mp
import psutil

if "XDG_RUNTIME_DIR" not in os.environ:
    runtime_dir = "/tmp/runtime-user"
    os.makedirs(runtime_dir, exist_ok=True)
    os.chmod(runtime_dir, 0o700)
    os.environ["XDG_RUNTIME_DIR"] = runtime_dir

os.environ["RUST_LOG"] = "error"

# Initialize logger for the module
logger_mp = logging_mp.get_logger(__name__, level=logging_mp.INFO)

def transform_pose(pose1, pose2):
        pose = np.zeros(7)
        t1 = pose1[:3]; t2 = pose2[:3]
                
        rot1 = R.from_quat(pose1[3:])
        rot2 = R.from_quat(pose2[3:])
        pose[:3] = t1 + rot1.apply(t2)
        pose[3:] = (rot1 * rot2).as_quat()
        return pose
    
if __name__ == "__main__":

    data_dir = "dataset/data/1221_duo_unitree_bread_picking_human_114ep"
    start_episode = 1
    end_episode = 180
    fps = 100
    skip_step_nums = 2
    # 根据电脑的运存大小调整运存上限 在终端里free -h查看avaliale内存大小 
    lim_storage = 25

    episode_list = range(start_episode, end_episode + 1)
    action_ori_type = "quaternion"
    umi_rotation_transform = {"right": [0.7071068, 0, 0.7071068, 0]}
    contain_ft = False
    data_type = "human_hand"
    logger_mp.info(f"Dir: {data_dir}")
    mem_gb = 0

    for episode_data_number in episode_list:
        episode_dir = f"episode_{str(episode_data_number).zfill(4)}"
        logger_mp.info(f"episode_{str(episode_data_number).zfill(4)}")
        full_path = os.path.join(data_dir, episode_dir)

        if os.path.exists(full_path):         
            episode_reader = RerunEpisodeReader(
                task_dir=data_dir, 
                action_type=ActionType.END_EFFECTOR_POSE,
                action_prediction_step=1, 
                action_ori_type=action_ori_type,
                observation_type=ObservationType.END_EFFECTOR_POSE,
                rotation_transform=None,
                contain_ft=contain_ft,
                data_type=data_type,
            )
             
            logger_mp.info(f'find Episode {episode_data_number}')  
            episode_data = episode_reader.return_episode_data(episode_data_number, skip_step_nums)

            if episode_data is None or len(episode_data) == 0:
                logger_mp.warning(f'Episode {episode_data_number} has no data')
                # delete this data
                
                continue
              
            # step counter
            step_num = int(skip_step_nums*len(episode_data)) 
            example_data = episode_data[0]
            state_keys = episode_reader._state_keys            
            online_logger = RerunLogger(
                task_dir=data_dir, 
                prefix=f"ep{episode_data_number}/", 
                IdxRangeBoundary=60, 
                memory_limit="20GB",
                example_item_data=example_data, 
                action_type=ActionType.END_EFFECTOR_POSE, 
                state_keys=state_keys
            )

            logger_mp.info(f'Episode {episode_data_number}:  {step_num} step')
            
            # storage waring messages
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            mem_gb += mem_info.rss / (1024 ** 3)
            if mem_gb <= lim_storage:
                 logger_mp.info(f"storage used: {mem_gb:.4f} GB")
            else:
                logger_mp.info(f"storage used: {mem_gb:.4f} GB")
                logger_mp.warning(f'storage used too much')
                logger_mp.warning(f'Close the rerun to release the storage')
                logger_mp.warning(f" Press 'c' to release storage")  
    
            for item_data in episode_data:
                online_logger.log_item_data(item_data)
                time.sleep(1/fps)
            logger_mp.info(f"Episode {episode_data_number} rerun finished。")  
            logger_mp.info(f" Press 'Enter' to continue ,or 'ctrl c' to exit")

            user_input = input() 

            if mem_gb > lim_storage:
                mem_gb = 0
                os.system("pkill -x rerun")
                logger_mp.info(f"Rerun UI has closed, system has enough storage now")

            #     
            # if user_input.lower() == 'q':
            #     os.system("pkill -x rerun")
            #     logger_mp.info("Keyboard interrupt")
            #     break
        else:
            logger_mp.warning(f'Could not find: {full_path}')
            logger_mp.warning(f"Press 'Enter' to  continue, or 'ctrl c' to exit ")
            user_input = input()  
    logger_mp.info("rerun finished.")
        