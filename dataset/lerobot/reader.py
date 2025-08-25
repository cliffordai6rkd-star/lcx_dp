import os
import json
import cv2
import numpy as np
import time, enum
from scipy.spatial.transform import Rotation as R
import rerun as rr
import rerun.blueprint as rrb
from datetime import datetime
os.environ["RUST_LOG"] = "error"
import glog as log

class ActionType(enum.Enum):
    JOINT_POSITION = 0
    JOINT_POSITION_DELTA = 1
    END_EFFECTOR_POSE = 2
    END_EFFECTOR_POSE_DELTA = 3
    JOINT_TORQUE = 4

class RerunEpisodeReader:
    def __init__(self, task_dir = ".", json_file="data.json", action_type: ActionType = ActionType.JOINT_POSITION,
                 action_prediction_step = 2):
        self.task_dir = task_dir
        self.json_file = json_file
        self.action_type = action_type
        self._action_prediction_step = action_prediction_step

    def return_episode_data(self, episode_idx, skip_steps_nums):
        # Load episode data on-demand
        episode_dir = os.path.join(self.task_dir, f"episode_{episode_idx:04d}")
        json_path = os.path.join(episode_dir, self.json_file)

        if not os.path.exists(json_path):
            log.warn(f"Episode {episode_idx} data.json not found.")
            return None

        with open(json_path, 'r', encoding='utf-8') as jsonf:
            json_file = json.load(jsonf)

        episode_data = []

        # Loop over the data entries and process each one
        counter = 0
        skip_steps_nums = int(skip_steps_nums)
        last_state_data = None
        len_json_file = len(json_file['data'])
        json_data = json_file['data']
        # print(f'json data: {json_data}')
        for i, item_data in enumerate(json_file['data']):
            # Process images and other data
            colors = self._process_images(item_data, 'colors', episode_dir)
            if colors is None:
                continue
            depths = self._process_images(item_data, 'depths', episode_dir)
            if depths is None:
                continue
            audios = self._process_audio(item_data, 'audios', episode_dir)

            # Append the data in the item_data list
            cur_actions = {}
            action_state_id = i+self._action_prediction_step
            if action_state_id >= len_json_file:
                continue
            if self.action_type == ActionType.JOINT_POSITION:
                joint_states = item_data.get("joint_states", {})
                cur_actions = self._get_absolute_action(joint_states, 
                                            action_state=json_data[action_state_id]["joint_states"],
                                            attribute_name="position")
            elif self.action_type == ActionType.END_EFFECTOR_POSE:
                cur_actions = self._get_absolute_action(item_data.get("ee_states", {}),
                                            action_state=json_data[action_state_id]["ee_states"])
            elif self.action_type == ActionType.JOINT_POSITION_DELTA:
                joint_states = item_data.get("joint_states", {})
                cur_actions = self._get_delta_action(joint_states, last_state_data, "position")
                last_state_data = joint_states
            elif self.action_type == ActionType.END_EFFECTOR_POSE_DELTA:
                # @TODO: 欧拉角转换 for check
                ee_states = item_data.get("ee_states", {})
                for key, cur_ee_state in ee_states.items():
                    ee_states[key][3:] = R.from_quat(cur_ee_state[3:]).as_euler('xyz')
                cur_actions = self._get_delta_action(ee_states, last_state_data)
                for key, action in cur_actions.items():
                    cur_actions[key][3:] = R.from_euler(action[3:], "xyz").as_quat()
                last_state_data = ee_states
            else:
                raise ValueError(f'The action type {self.action_type} is not supported for reading episode data')
            
            # @TODO: add the gripper states as action for check
            tool_states = item_data.get("tools", {})
            for key, tool_state in tool_states.items():
                cur_actions[key] = np.hstack((cur_actions[key], tool_state["position"]))
            if counter % skip_steps_nums == 0:
                episode_data.append(
                    {
                        'idx': item_data.get('idx', 0),
                        'colors': colors,
                        'depths': depths,
                        'joint_states': item_data.get('joint_states', {}),
                        'ee_states': item_data.get('ee_states', {}),
                        'tools': item_data.get('tools', {}),
                        'imus': item_data.get('imus', {}),
                        'tactiles': item_data.get('tactiles', {}),
                        'audios': audios,
                        'actions': cur_actions
                    }
                )
            counter += 1
        
        return episode_data
    
    def get_episode_text_info(self, episode_id):
        episode_dir = os.path.join(self.task_dir, f"episode_{episode_id:04d}")
        json_path = os.path.join(episode_dir, self.json_file)

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Episode {episode_id} data.json not found.")

        with open(json_path, 'r', encoding='utf-8') as jsonf:
            json_file = json.load(jsonf)

        text_info = json_file["text"]
        text_info = 'description: ' + text_info["desc"] + ' ' \
                + 'steps: ' + text_info["steps"] + ' ' + 'goal: ' + text_info["goal"]        
        return text_info
    
    def _get_absolute_action(self, states, action_state, attribute_name = None):
        cur_action = {}
        for key, state in states.items():
            if attribute_name is not None:
                cur_action[key] = action_state[key][attribute_name]
            else:
                cur_action[key] = action_state[key]
        return cur_action
    
    def _get_delta_action(self, states, last_state_data, attribute_name = None):
        cur_action = {}
        last_state_value = {}
        for key, state in last_state_data.items():
            state_value = state if attribute_name is None else state[attribute_name]
            last_state_value[key] = state_value
        
        for key, state in states.items():
            state_value = state if attribute_name is None else state[attribute_name]
            if last_state_data is None:
                cur_action[key] = [0] * len(state_value)
            else:
                cur_action[key] = state_value - last_state_value[key]
        return cur_action

    def _process_images(self, item_data, data_type, dir_path):
        images = {}

        for key, file_name in item_data.get(data_type, {}).items():
            if file_name:
                file_path = os.path.join(dir_path, file_name)
                if os.path.exists(file_path):
                    image = cv2.imread(file_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images[key] = image
                else:
                    return None
        return images

    def _process_audio(self, item_data, data_type, episode_dir):
        audio_item = item_data.get(data_type, {})
        if audio_item is None:
            return
        
        audio_data = {}
        dir_path = os.path.join(episode_dir, data_type)

        for key, file_name in audio_item.items():
            if file_name:
                file_path = os.path.join(dir_path, file_name)
                if os.path.exists(file_path):
                    pass  # Handle audio data if needed
        return audio_data

if __name__ == "__main__":
    # episode_reader = RerunEpisodeReader(task_dir = unzip_file_output_dir)
    # # TEST EXAMPLE 1 : OFFLINE DATA TEST
    # episode_data6 = episode_reader.return_episode_data(6)
    # logger_mp.info("Starting offline visualization...")
    # offline_logger = RerunLogger(prefix="offline/")
    # offline_logger.log_episode_data(episode_data6)
    # logger_mp.info("Offline visualization completed.")
    
    # data_folder = "./data/episode_0052"
    # episode_reader = RerunEpisodeReader(task_dir='./data', action_type=ActionType.JOINT_POSITION_DELTA)
    # data52 = episode_reader.return_episode_data(52)
    pass
    