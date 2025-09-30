from dataset.lerobot.reader import ActionType, Action_Type_Mapping_Dict
from dataset.lerobot.data_loader_base import DataLoaderBase
import os, sys
import numpy as np
import tqdm
sys.path.append("/home/yuxuan/Code/hirol/diffusion_policy")
from diffusion_policy.common.replay_buffer import ReplayBuffer
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import glog as log

class DiffusionPolicyLoader(DataLoaderBase):
    def __init__(self, config, task_dir, json_file_name = "data.json", action_type = ActionType.JOINT_POSITION):
        action_type = config.get("action_type", "joint_position")
        if action_type not in Action_Type_Mapping_Dict:
            raise ValueError(f'{action_type} is not supported!!')
        action_type = Action_Type_Mapping_Dict[action_type]
        json_file_name = config.get(f'json_file_name', "data.json")
        super().__init__(config, task_dir, json_file_name, action_type)
        
        self._task_list = config.get('task_list', None)
        if self._task_list is None:
            self._task_list = [task_dir]
        else:
            for i, task_name in enumerate(self._task_list):
                self._task_list[i] = os.path.join(self._task_dir, task_name)
        for task_dir in self._task_list:
            if not os.path.exists(task_dir):
                raise ValueError(f'Could not find {task_dir}')
        log.info(f'task list: {self._task_list}')
        
        self._orientation_representation = 'euler'
        self._output_path_dir = config.get('output_path', './data')
        self._file_name = config.get('data_name', 'dataset')
        self._file_name += '.zarr'
        self._zarr_path = os.path.join(self._output_path_dir, self._file_name)
        self._load_fps = config.get("fps", 30)
        
        # replay buffer
        file_exists = os.path.exists(self._zarr_path)
        mode = 'a' if file_exists else 'w'
        self._replay_buffer = ReplayBuffer.create_from_path(self._zarr_path, mode=mode)
        
    def put_key_value_to_zarr_data(self, zarr_data, key, value):
        if key not in zarr_data:
            zarr_data[key] = np.array(value)[None]
        else:
            appended_data = np.array(value)[None]
            zarr_data[key] = np.vstack((zarr_data[key], appended_data))
    
    def create_zarr_dataset(self, task_dir, episode_dir):
        n_episodes = self._replay_buffer.n_episodes
        curr_episode_id = n_episodes

        # episode data appending
        skip_num_step = int(30 / self._load_fps)
        episode_data, _ = self.load_episode(task_dir, episode_dir, skip_num_step)
        if episode_data is None:
            log.warn(f'{task_dir}/{episode_dir} has problem during loading the episode data')
            return
        
        zarr_episode_data = {}
        episode_step_nums = 0
        for stpes, step_data in tqdm(enumerate(episode_data), desc="processing steps ", unit="steps"):
            # images
            colors_data = step_data.get("colors", {})
            for key, value in colors_data.items():
                self.put_key_value_to_zarr_data(zarr_episode_data, key, value)
            
            # state and action and stages
            joint_states = step_data.get("joint_states", {})
            ee_states = step_data.get("ee_states", {})
            tools = step_data.get("tools", {}); actions = step_data.get("actions", {})
            obs_state = np.array([]); zarr_actions = np.array([]); zarr_stages = np.array([])
            for id, (key, value) in enumerate(joint_states.items()):
                robot_state = np.array(value["position"])
                if self._contain_ee_obs: robot_state = np.hstack((ee_states[key]["pose"], robot_state))
                obs_state = np.hstack((obs_state, robot_state, tools[key]["position"],))
                zarr_actions = np.hstack((zarr_actions, actions[key]))
                zarr_stages = np.hstack((zarr_stages, [1]))
            self.put_key_value_to_zarr_data(zarr_episode_data, "state", obs_state)
            self.put_key_value_to_zarr_data(zarr_episode_data, "action", zarr_actions)
            self.put_key_value_to_zarr_data(zarr_episode_data, "stage", zarr_stages)
            episode_step_nums += 1
        self._replay_buffer.add_episode(zarr_episode_data)
        log.info(f'state shape: {zarr_episode_data["state"].shape}')
        log.info(f'The episode data {curr_episode_id} contains {episode_step_nums} and is successfully save to {self._zarr_path} from {self._task_dir}/{episode_dir}')
        log.debug(f'episode_data: {zarr_episode_data}')
    
    def convert_dataset(self):
        for task_dir in tqdm(self._task_list, desc="processing tasks", unit="tasks"):
            episode_dirs = os.listdir(task_dir)
            for episode_dir in tqdm(episode_dirs, desc="processing episodes ", unit="episodes"):
                self.create_zarr_dataset(task_dir, episode_dir)
            log.info(f'Finished processing the {task_dir}')
        log.info(f'created all episodes in {self._task_list}')
            
if __name__ == "__main__":
    import yaml
    config_file = "config/dp_zarr_config.yaml"
    cur_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(cur_path, config_file)
    with open(cfg_file, 'r') as stream:
        config = yaml.safe_load(stream)
    print(f'config: {config}')
    task_dir = "/home/yuxuan/Code/hirol/teleoperated_trajectory/peg_in_hole"
    
    dp_data_loader = DiffusionPolicyLoader(config, task_dir)
    dp_data_loader.convert_dataset()
    