from __future__ import annotations

import logging, random
import time, copy, threading
from hardware.base.utils import ToolControlMode
import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy

import glog as log
from factory.tasks.inferences_tasks.inference_base import InferenceBase
from tools.performance_profiler import timer
import cv2
from dataset.utils import ActionType

logger = logging.getLogger(__name__)

class PI0_Inferencer(InferenceBase):
    def __init__(self, config):
        super().__init__(config)
        self._tasks = config["tasks"]
        
        # model loading
        # Create a trained policy.
        host = config.get("host", "192.168.1.234")
        port = config.get("port", 8000)
        api_key = config.get("api_key", None)
        self._pi0_policy = _websocket_client_policy.WebsocketClientPolicy(
                        host=host, port=port, api_key=api_key,)
        logger.info(f"Server metadata: {self._pi0_policy.get_server_metadata()}")
        self._key_mapping = config.get('key_mapping', None)
        
        # Send a few observations to make sure the model is loaded.
        for i in range(2):
            obs = self.convert_from_gym_obs()
            log.info(f'{i}th init obs: {obs}')
            self._pi0_policy.infer(obs)
    
    def policy_reset(self):
        self._pi0_policy.reset()

    def policy_prediction(self, obs):
        action = self._pi0_policy.infer(obs)
        return action["actions"]
    
    def start_inference(self):
        execution_thread = None
        for episode_num in range(self._num_episodes):
            if self._quit: break
            
            self._gym_robot.reset()
            self._last_gripper_open = [True, True]
            self._pi0_policy.reset()
            self._status_ok = True
            log.info(f'Starting the {episode_num} th episodes')
            while self._status_ok:
                with timer("gym_obs", "pi0_inferencer"):
                    pi0_obs = self.convert_from_gym_obs()
                    
                with timer("pi0_inference_time", "pi0_inferencer"):
                    start_time = time.perf_counter()
                    result = self._pi0_policy.infer(pi0_obs)
                    log.info(f'infer used time: {time.perf_counter() - start_time}s')
                    if self._predicted_action_chunks < 0:
                        self._predicted_action_chunks = result["actions"].shape[0]
                    
                    if execution_thread is not None:
                        log.info(f'execute thread is alive: {execution_thread.is_alive()}')
                    else: log.info(f'Execution thread is None')
                    if execution_thread is not None and execution_thread.is_alive():
                        self._execution_interruption = True
                        execution_thread.join()
                        log.info(f'Waiting for finishing the execution thread: {execution_thread.is_alive()}')
                    self._execution_interruption = False
                    execute_action = result["actions"][:self._execution_action_chunk_size]
                   
                    # log.info(f'action shape: {execute_action[0].shape} for {episode_num}th episodes')
                    
                def multi_step_tasks():
                    with timer("gym_step", "pi0_inferencer"):
                        self.convert_to_gym_action(execute_action)
                            
                execution_thread = threading.Thread(target=multi_step_tasks)
                execution_thread.start()
                if not self._async_execution:
                    execution_thread.join()
                # log.info(f'result action chunk shape: {result["actions"].shape}')
                
    def convert_from_gym_obs(self, gym_obs = None):
        gym_obs = super().convert_from_gym_obs(gym_obs)
        self.image_display(gym_obs)
            
        pi0_obs = {}
        # @TODO: coupling solution for testing
        pi0_obs["state"] = np.array([])
        temp_joint_positions = np.array([])
        for key, cur_state in gym_obs['state'].items():
            temp_joint_positions = np.hstack((temp_joint_positions, cur_state))
            pi0_obs["state"] = np.hstack((pi0_obs["state"], cur_state))
        
        self._lock.acquire()
        self._joint_positions = temp_joint_positions
        self._lock.release()
            
        for key, img in gym_obs["colors"].items():
            # log.info(f'{key} , shape: {img.shape}')
            pi0_obs[key] = img
            # pi0_obs[key] = cv2.resize(img, (224, 224))
            # if pi0_obs[key].shape[0] == 3:
            #     pi0_obs[key] = einops.rearrange(pi0_obs[key], "c h w -> h w c")
            # log.info(f'after shape: {pi0_obs[key].shape}')
            
        if isinstance(self._tasks, list):
            selected_index = random.randrange(len(self._tasks))
            pi0_obs["task"] = self._tasks[selected_index]
        else: pi0_obs["task"] = self._tasks
        
        if self._key_mapping:
            mapped_pi0_obs = {}
            for key, value in pi0_obs.items():
                if key in self._key_mapping:
                    mapped_pi0_obs[self._key_mapping[key]] = value
                elif key == 'state':
                    mapped_pi0_obs[self._key_mapping["joint_state"]] = value[:-1]
                    mapped_pi0_obs[self._key_mapping["gripper_state"]] = np.array([value[-1]])
            pi0_obs = mapped_pi0_obs
        
        # if not self._key_mapping:
        #     log.info(f'pi0_obs: {pi0_obs["state"]}')
        # else: log.info(f'pi0_obs: {pi0_obs}')
        return pi0_obs
        
    def close(self):
        """Clean up resources and close display windows."""
        if self._enable_display:
            cv2.destroyWindow(self._display_window_name)
        del self._pi0_policy
          
def main():
    from factory.utils import parse_args
    from hardware.base.utils import dynamic_load_yaml
    # testing gym api
    arguments = {"config": {"short_cut": "-c",
                            "symbol": "--config",
                            "type": str, 
                            "default": "factory/tasks/inferences_tasks/pi0/config/fr3_pi0_cfg_client.yaml",
                            "help": "Path to the config file"}}
    args = parse_args("pi0 inference", arguments)
    
    # Load configuration from the YAML file
    config = dynamic_load_yaml(args.config)
    log.info(f'pi0 config: {config}')
    pi0_executor = PI0_Inferencer(config)
    # pi0_executor.start_inference()
    pi0_executor.start_common_inference()
    
if __name__ == "__main__":
    main()
