import os
import time
import copy
import threading
import yaml
import numpy as np
import cv2
import glog as log
from enum import Enum
from typing import Optional, Dict

# 引入你的库组件
from dataset.lerobot.reader import RerunEpisodeReader, Action_Type_Mapping_Dict, ActionType, ObservationType
from dataset.lerobot.delete import Deleter
from simulation.mujoco.mujoco_sim import MujocoSim
from factory.tasks.inferences_tasks.utils.display import display_images
from sshkeyboard import listen_keyboard, stop_listening
from hardware.base.utils import dynamic_load_yaml

class ReplayState(Enum):
    IDLE = "idle"
    REPLAYING = "replaying"
    STOPPED = "stopped"
    WATING_INPUT = "waiting"
    INTERRUPTION = "interruption"

class UmiReplay:
    def __init__(self, config: dict):
        self._config = config
        
    # Setup paths
        self._task_data_dir = config["task_data_dir"]
        cur_path = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(self._task_data_dir):
            self._task_data_dir = os.path.join(cur_path, "../..", self._task_data_dir)
        
        if not os.path.exists(self._task_data_dir):
            raise ValueError(f"Task data directory does not exist: {self._task_data_dir}")

        # 2. UMI 特有的配置 (用于坐标转换)
        self._pika_mode = config.get("pika_mode", "absolute_delta") # 默认对齐采集时的模式
        self._use_mujoco = True 
        
        # 3. Reader 配置
        action_type_str = config.get("action_type", "ee_pose")
        if action_type_str not in Action_Type_Mapping_Dict:
             # 如果配置里写的不是标准映射键，默认设为 EE_POSE，适配 UMI
            self._action_type = ActionType.END_EFFECTOR_POSE
        else:
            self._action_type = Action_Type_Mapping_Dict[action_type_str]

        self._replay_frequency = config.get("replay_frequency", 30)
        self._action_prediction_step = config.get("action_prediction_step", 1)
        self._skip_steps = config.get("skip_steps", 1)
        self._state_keys = config.get("state_keys", ["left", "right"]) # 比如 ["left", "right"]
        
        # 4. 状态管理
        self._state = ReplayState.IDLE
        self._state_lock = threading.Lock()
        self._replay_thread_running = True
        self._current_episode_id: Optional[int] = None
        self._is_initialized = False

        self._episode_reader: Optional[RerunEpisodeReader] = None
        self._mujoco: Optional[MujocoSim] = None

    def create_replay_system(self):
        log.info("Starting UMI Replay system...")

        # 1. 初始化 MuJoCo (逻辑来自 UmiCollection)
        mujoco_cfg_path = self._config.get("mujoco_config", {})
        cur_path = os.path.dirname(os.path.abspath(__file__))
        # 注意：这里路径层级根据实际文件位置调整
        cfg_file = os.path.join(cur_path, "../..", mujoco_cfg_path)
        
        with open(cfg_file, 'r') as stream:
            mujoco_config_content = yaml.safe_load(stream)
        self._mujoco = MujocoSim(mujoco_config_content["mujoco"])

        # 2. 初始化 Data Reader (逻辑来自 DataReplay)
        self._episode_reader = RerunEpisodeReader(
            task_dir=self._task_data_dir,
            action_type=self._action_type,
            action_prediction_step=self._action_prediction_step,
            observation_type=ObservationType.MASK, # 只读 Mask 或 Image
            state_keys=self._state_keys,
            camera_keys=self._config.get("cam_keys", None),
            data_type="human_hand"
        )

        # 3. 键盘监听
        threading.Thread(target=listen_keyboard, kwargs={"on_press": self._keyboard_on_press, 
                        "until": None, "sequential": False}, daemon=True).start()
        
        self._episode_id_str = ''
        self._is_initialized = True
        log.info(f"UMI Replay system initialized. Freq: {self._replay_frequency}Hz")

    def run(self):
        if not self._is_initialized:
            raise RuntimeError("System not initialized")
            
        target_period = 1.0 / self._replay_frequency
        
        while self._replay_thread_running:
            with self._state_lock:
                current_state = self._state

            if current_state == ReplayState.IDLE:
                time.sleep(0.1)
                # 保持渲染以防止窗口卡死
                # self._mujoco.render() 
            elif current_state == ReplayState.REPLAYING:
                self._execute_replay(target_period)
                with self._state_lock:
                    self._state = ReplayState.IDLE
            elif current_state == ReplayState.STOPPED:
                break
                
        self._mujoco.close()
        log.info("System exit.")

    def _execute_replay(self, target_period: float):
        if self._current_episode_id is None:
            log.warn('No episode ID selected.')
            return

        episode_id = self._current_episode_id
        log.info(f"Loading episode {episode_id}...")
        
        # 读取数据 (使用 skip_steps 跳帧)
        episode_data = self._episode_reader.return_episode_data(episode_id, skip_steps_nums=self._skip_steps)
        if not episode_data:
            log.error(f"Failed to load episode {episode_id}")
            return

        log.info(f"Starting replay episode {episode_id} ({len(episode_data)} frames)")
        
        # 映射 Key 用于 MuJoCo Mocap (与 UmiCollection 保持一致)
        target_pose_key = {"single":"targetR", "left": "targetL", 
                           "right": "targetR", "head": "targetH"}

        next_run_time = time.perf_counter()

        for frame_idx, frame_data in enumerate(episode_data):
            # 1. 状态检查 (允许中断)
            with self._state_lock:
                if self._state != ReplayState.REPLAYING:
                    if self._state == ReplayState.INTERRUPTION:
                        log.info("Replay interrupted!")
                    return

            # 2. 获取当前帧数据
            # 注意：EpisodeWriter 通常将 EE 姿态存在 'ee_states' 或 'actions' 中
            # 根据 UmiCollection 的代码，我们找 'ee_states'
            ee_states = frame_data.get("ee_states", {})
            tools = frame_data.get("tools", {})
            colors = frame_data.get("colors", {})
            
            # 链式传参防止None报错   
            # double_gripper_value = config.get("tools",{})
            # left_gripper_value = double_gripper_value.get("left",{})
            # right_gripper_value = double_gripper_value.get("right", {})
            # left_gripper_pos = left_gripper_value.get("position", None)
            # left_time_stamp = left_gripper_value.get("time_stamp", None)
            # right_gripper_pos = right_gripper_value.get("position", None)
            # right_time_stamp = right_gripper_value.get("time_stamp", None)
            # 3. 更新 MuJoCo 状态
            if ee_states:
                for pose_key, pose_dict in ee_states.items():
                    if pose_key not in target_pose_key:
                        continue
                    
                    mocap_name = target_pose_key[pose_key]
                    raw_pose = np.array(pose_dict["pose"]) # 原始录制姿态
                    
                    # pika gripper的位置设定与现实场景对齐
                    visual_pose = copy.deepcopy(raw_pose)
                    if self._pika_mode == "absolute_delta":
                        if 'left' in pose_key:
                            visual_pose[1] += 0.3
                            visual_pose[2] += 0.45
                        elif 'right' in pose_key:
                            visual_pose[1] -= 0.3
                            visual_pose[2] += 0.45
                        else: 
                            visual_pose[2] += 0.95
                    
                    self._mujoco.set_target_mocap_pose(mocap_name, visual_pose)

            # 4. 更新夹爪 (Tools) @TODO:尝试在mocap的xml中将夹爪link到fr3pika上  
            # 逻辑：从umireplay读取tool的值 下发到gym_inference的robot_facotory 在到mujoco_sim中
            # datareplay的 self._convert_to_gym_format
            # if tools:
            #     tool_command_input = {}
            #     # 遍历 Replay 数据中的工具状态
            #     for tool_name, tool_state in tools.items():
            #         if isinstance(tool_state, dict) and 'position' in tool_state:
            #             val = tool_state['position']
            #         else:
            #             val = tool_state
            #     if self._mujoco:
            #          self._mujoco.set_tool_command(tool_command_input)


            # 6. 显示图像 (可选，用于对比)
            if colors:
                display_dict = {}
                for cam, img in colors.items():
                    display_dict[f"Replay_{cam}"] = img
                display_images(display_dict, "Replay Camera View")

            # 7. 频率控制
            next_run_time += target_period
            sleep_time = next_run_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_run_time = time.perf_counter() # 追赶时间

        log.info(f"Episode {episode_id} replay finished.")

    def _keyboard_on_press(self, key):
        with self._state_lock:
            state = copy.deepcopy(self._state)
            
        if key == 's' and state == ReplayState.IDLE:
            log.info(f'You could press a number to replay the episode id now!!!!')
            with self._state_lock:
                self._state = ReplayState.WATING_INPUT
            self._episode_id_str = ''
        if key >= '0' and key <= '9' and state == ReplayState.WATING_INPUT:
            try:
                self._episode_id_str += key
                log.info(f'cur episode id str is {self._episode_id_str}')
            except:
                log.warn(f'catch exception the input {key} is not a single number')
                log.warn(f'Please continue to enter the single number or s \
                         to the get the replay episode, cur episode: {self._current_episode_id}')
        if key == 's' and state == ReplayState.WATING_INPUT:
            try:
                self._current_episode_id = int(self._episode_id_str)
            except Exception as e:
                log.warn(f'Catch the exception {e} when ready to replay data but {self._episode_id_str} is not valid!')
                return
            # valid episode for replay
            with self._state_lock:
                self._state = ReplayState.REPLAYING
            log.info(f'Will start to replay {self._current_episode_id} data from {self._task_data_dir}')

        if key == "d":
            self._current_episode_id = int(self._episode_id_str)
            log.info(f'Deleting episod{self._current_episode_id} from{self._task_data_dir}')
            Deleter.delete_episodes(self._current_episode_id, self._task_data_dir)
            log.info(f'Delete success')

    def close(self):
        self._replay_thread_running = False
        with self._state_lock:
            self._state = ReplayState.STOPPED
        stop_listening()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    from factory.utils import parse_args
    
    # 定义参数
    arguments = {
        "config": {
            "short_cut": "-c",
            "symbol": "--config",
            "type": str,
            # 指向你的配置文件
            "default": "factory/tasks/umi_collections/config/umi_replay_cfg.yaml", 
            #factory/tasks/data_replay_config/right_umi_fr3_data_replay_cfg.yaml
            "help": "Path to the config file"
        },
    }
    
    args = parse_args("umi replay", arguments)
    config = dynamic_load_yaml(args.config)
    
    # 确保 config 中包含 replay 必须的字段，如果 yaml 里没有，可以手动补
    if "task_data_dir" not in config:
        # 假设 task_name 在 yaml 里
        config["task_data_dir"] = config.get("task_name", "default_task")
    
    replay_system = UmiReplay(config)
    replay_system.create_replay_system()
    replay_system.run()