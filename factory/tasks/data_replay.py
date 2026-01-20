from factory.components.gym_interface import GymApi
from dataset.lerobot.reader import RerunEpisodeReader, Action_Type_Mapping_Dict, ActionType, ObservationType
from hardware.base.utils import transform_pose, pose_diff, transform_quat
from factory.tasks.inferences_tasks.utils.display import display_images
from dataset.lerobot.delete import Deleter
import threading
import time, shutil
import numpy as np
import glog as log
from enum import Enum
from typing import Dict, Optional
from scipy.spatial.transform import Rotation as R
import os, copy
from sshkeyboard import listen_keyboard, stop_listening
import glog as log
from dataset.lerobot.data_process import EpisodeWriter, serialize_data

class ReplayState(Enum):
    IDLE = "idle"
    REPLAYING = "replaying"
    STOPPED = "stopped"
    WATING_INPUT = "waiting"
    INTERRUPTION = "interruption"
    WAITING_DATA_SAVING = "waiting_data"

class DataReplay:
    def __init__(self, config: dict):
        self._config = config

        # Validate required keys
        required_keys = ["task_data_dir", "action_type", "action_orientation_type",
            "replay_frequency", "motion_config", "tool_position_dof", "data_type"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: '{key}'")

        # Setup paths
        self._task_data_dir = config["task_data_dir"]
        cur_path = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(self._task_data_dir):
            self._task_data_dir = os.path.join(cur_path, "../..", self._task_data_dir)

        if not os.path.exists(self._task_data_dir):
            raise ValueError(f"Task data directory does not exist: {self._task_data_dir}")

        # Action type validation
        action_type_str = config["action_type"]
        if action_type_str not in Action_Type_Mapping_Dict:
            raise ValueError(f"Invalid action_type: '{action_type_str}'")
        self._action_type = Action_Type_Mapping_Dict[action_type_str]

        self._action_ori_type = config.get("action_orientation_type", "quaternion")
        self._action_prediction_step = config.get("action_prediction_step", 1)
        self._skip_steps = config.get("skip_steps", 1)
        self._replay_frequency = config["replay_frequency"]
        self._tool_position_dof = self._config["tool_position_dof"]
        self._tool_max = config.get("tool_max", 90)
        self._rotation_transform = config.get("rotation_transform", None)
        self._state_keys = config.get("state_keys", None)
        self._cam_keys = config.get("cam_keys", None)
        self._data_type = config.get("data_type", "human_hand")

        # data saving 
        self._data_save_dir = config.get("data_save_dir", None)
        if self._data_save_dir is not None:
            self._data_task = config.get('data_task', None)
            assert self._data_task is not None
            self._task_dir = os.path.join(cur_path, "../..", self._data_save_dir)
            self._episode_writer = EpisodeWriter(self._task_dir)
        
        # State management
        self._state = ReplayState.IDLE
        self._state_lock = threading.Lock()
        self._replay_thread_running = True
        self._current_episode_id: Optional[int] = None

        # Components
        self._episode_reader: Optional[RerunEpisodeReader] = None
        self._gym_api: Optional[GymApi] = None
        self._is_initialized = False
        self._init_pose = None
        self._enable_hardware = False

        log.info(f"DataReplay initialized: {self._task_data_dir}, {action_type_str}, {self._replay_frequency}Hz")

    def create_replay_system(self) -> bool:
        log.info("Creating replay system...")

        # Create GymApi
        gym_config = self._config.copy()
        gym_config["max_step_nums"] = 100

        self._gym_api = GymApi(gym_config)

        # Initialize data reader
        self._episode_reader = RerunEpisodeReader(
            task_dir=self._task_data_dir,
            action_type=self._action_type,
            action_prediction_step=self._action_prediction_step,
            action_ori_type=self._action_ori_type,
            observation_type=ObservationType.MASK,
            rotation_transform=self._rotation_transform,
            state_keys=self._state_keys,
            camera_keys=self._cam_keys,
            data_type=self._data_type,
        )

        # Start keyboard listener
        threading.Thread(target=listen_keyboard, kwargs={"on_press": self._keyboard_on_press,
                        "until": None, "sequential": False}, daemon=True).start()
        self._episode_id_str = ''
        self._is_initialized = True
        log.info("Replay system created successfully")
        return True

    def replay_data(self):
        if not self._is_initialized:
            raise RuntimeError("Replay system not initialized")

        log.info("Replay data loop started")
        target_period = 1.0 / self._replay_frequency

        while self._replay_thread_running:
            with self._state_lock:
                current_state = self._state

            if current_state == ReplayState.IDLE:
                time.sleep(0.1)
            elif current_state == ReplayState.REPLAYING:
                success = self._execute_replay(target_period)
                with self._state_lock:
                    if self._data_save_dir is not None and success:
                        self._episode_writer.save_episode()
                        self._state = ReplayState.WAITING_DATA_SAVING
                    else:
                        self._state = ReplayState.IDLE
                    
                log.info("Replay completed, returning to IDLE")
            elif current_state == ReplayState.STOPPED:
                break

        log.info("Replay data loop stopped")

    def _execute_replay(self, target_period: float):
        if self._current_episode_id is None:
            log.warn(f'Do not retrive the correct episode id for execution')
            return False

        episode_id = self._current_episode_id
        log.info(f"Loading episode {episode_id}")

        episode_data = self._episode_reader.return_episode_data(episode_id, skip_steps_nums=self._skip_steps)
        log.info(f"Finished loading episode {episode_id} from {self._task_data_dir}")
        self._state_keys = self._episode_reader._state_keys
        
        if episode_data is None or len(episode_data) == 0:
            log.warning(f"Episode {episode_id} not found or empty")
            return False

        # Reset robot
        self._gym_api.reset()
        time.sleep(0.5)

        # Replay loop
        log.info(f"Starting replay at {self._replay_frequency}Hz")
        next_run_time = time.perf_counter()

        # 获取episode start的pose
        data_init_pose = episode_data[0]["ee_states"]
        init_episode_pose = {}
        if self._rotation_transform:
            for key, pose in data_init_pose.items():
                if key not in self._state_keys:
                    continue
                pose["pose"][3:] = transform_quat(pose["pose"][3:], self._rotation_transform[key])
                init_episode_pose[key] = {}
                init_episode_pose[key]["pose"] = pose["pose"]
        # iterate over the whole episode
        log.info(f'Ready to execute repaly data for episode {episode_id} for {len(episode_data)} datapoints with replay state {self._state}')
        # log.info(f'episode data: {episode_data}')
        if self._data_save_dir is not None:
            self._episode_writer.create_episode()
        for frame_data in episode_data:
            with self._state_lock:
                if self._state != ReplayState.REPLAYING:
                    if self._state == ReplayState.INTERRUPTION:
                        log.info(f'Data {self._task_data_dir}_{episode_id} replay execution interrupted!!!')
                        self._state = ReplayState.IDLE
                    log.info(f'Exit replay with state {self._state}')
                    return True

            actions = frame_data.get("actions", {})
            if not actions:
                log.warn(f"actions could not find from frame data!!!!")
                continue
            
            # relative pose reprensentation
            if self._rotation_transform and self._action_type == ActionType.END_EFFECTOR_POSE:
                for key, cur_action in actions.items():
                    actions[key] = cur_action 
                        
            gym_action = self._convert_to_gym_format(actions)
            res = self._gym_api.step(gym_action, True)
            display_colors = res[0]["colors"]; data_colors = frame_data.get("colors", {})
            for cam_name, cam_data in data_colors.items():
                cam_name = "data_" + cam_name
                display_colors[cam_name] = cam_data
            display_images(display_colors, "Data replay colors")
            
            if self._data_save_dir is not None:
                # saving step data
                joint_states = self._gym_api.get_joint_state()
                joint_ts = time.perf_counter()
                for key in joint_states.keys():
                    joint_states[key]["time_stamp"] = joint_ts
                ee_states = self._gym_api.get_ee_state()
                ee_ts = time.perf_counter()
                for key in ee_states.keys():
                    ee_states[key]["time_stamp"] = ee_ts
                tool_states = self._gym_api.get_tool_state()
                tool_ts = time.perf_counter()
                for key in tool_states.keys():
                    tool_states[key]["time_stamp"] = tool_ts
                cur_colors = {}
                obs_ts = self._gym_api.get_obs_timestamp()
                for cam_name, img in res[0]["colors"].items():
                    cur_colors[cam_data] = {"data": img, "time_stamp": obs_ts}
                    log.info(f'Got cam key {cam_name}')
                self._episode_writer.add_item(colors=cur_colors, 
                    joint_states=joint_states, ee_states=ee_states, tools=tool_states)
            
            # Timing
            next_run_time += target_period
            sleep_time = next_run_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_run_time = time.perf_counter()
                time.sleep(1.0 / self._replay_frequency)

        log.info(f"Episode {episode_id} completed")
        self._gym_api.reset()
        return True

    def _convert_to_gym_format(self, action_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        arm_actions = []
        tool_actions = []
        
        # @TODO: head info pop out
        # log.info(f'sorted action keys: {sorted(action_dict.keys())} {self._state_keys}')
        for key in self._state_keys:
            action = action_dict[key]
            if len(action) > 0:
                if key != "head":
                    arm_actions.append(action[:-1*self._tool_position_dof])
                    tool_actions.append(action[-1*self._tool_position_dof:] / self._tool_max)
                else:
                    arm_actions.append(action)

        return {
            'arm': np.concatenate(arm_actions) if arm_actions else np.array([]),
            'tool': np.array(tool_actions) if tool_actions else np.array([])
        }

    def _keyboard_on_press(self, key: str):
        with self._state_lock:
            state = copy.deepcopy(self._state)
        if key == 'q':
            self.close()
        if key == 'k':
            with self._state_lock:
                self._state = ReplayState.INTERRUPTION
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
        if key == 'y' and state == ReplayState.WAITING_DATA_SAVING:
            with self._state_lock:
                self._state = ReplayState.IDLE
        if key == 'n' and state == ReplayState.WAITING_DATA_SAVING:
            cur_episode = self._episode_writer.episode_id
            episode_dir = os.path.join(self._task_dir, f"episode_{str(cur_episode).zfill(4)}")
            if os.path.exists(episode_dir):
                shutil.rmtree(episode_dir)
                log.info(f'Deleting recorded robot data {episode_dir}')
            with self._state_lock:
                self._state = ReplayState.IDLE
        if key == 'd':
            self._current_episode_id = int(self._episode_id_str)
            log.info(f'Deleting episode {self._current_episode_id} data from {self._task_data_dir}')
            Deleter.delete_episodes(self._current_episode_id, self._task_data_dir)
            log.info(f'Deleting success')
        
        if key == 'h' and state == ReplayState.IDLE:
            self._enable_hardware = not self._enable_hardware
            self._gym_api.change_hardware_state(self._enable_hardware)
            log.info(f'Data replay set to hw state {self._enable_hardware}!!!')
        
    def close(self):
        log.info("Closing replay system")
        self._replay_thread_running = False

        with self._state_lock:
            self._state = ReplayState.STOPPED

        stop_listening()
        if self._gym_api:
            self._gym_api.close()

        log.info("Replay system closed")

if __name__ == "__main__":
    from factory.utils import parse_args
    from hardware.base.utils import dynamic_load_yaml
    arguments = {
        "config": {
            "short_cut": "-c",
            "symbol": "--config",
            "type": str,
            "default": "factory/tasks/data_replay_config/left_fr3_cfg.yaml",
            "help": "Path to the config file"
        },
    }
    args = parse_args("data replay pipeline", arguments)
    config = dynamic_load_yaml(args.config)
    
    data_replay = DataReplay(config)
    data_replay.create_replay_system()
    
    data_replay.replay_data()
    log.info(f'Finished data collection pipeline!!!')
    