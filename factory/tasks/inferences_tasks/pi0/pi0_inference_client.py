from __future__ import annotations

import logging, random
import time, copy, threading

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
        self._execution_steps = config.get("execution_step", 5)
        self._execution_interruption = False
        self._async_execution = config.get("async_execution", False)
        self._last_gripper_open = True
        
        # model loading
        # Create a trained policy.
        host = config.get("host", "0.0.0.0")
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
        
    def start_inference(self):
        execution_thread = None
        for episode_num in range(self._num_episodes):
            if self._quit: break
            
            self._gym_robot.reset()
            self._last_gripper_open = True
            self._pi0_policy.reset()
            self._status_ok = True
            log.info(f'Starting the {episode_num} th episodes')
            while self._status_ok:
                with timer("gym_obs", "pi0_inferencer"):
                    pi0_obs = self.convert_from_gym_obs()
                    
                with timer("pi0_inference_time", "pi0_inferencer"):
                    start_time = time.perf_counter()
                    result = self._pi0_policy.infer(pi0_obs)
                    log.info(f'infer used time: {time.perf_counter() - start_time}')
                    
                    if execution_thread is not None:
                        log.info(f'execute thread: {execution_thread.is_alive()}')
                    else: log.info(f'Execution thread is None')
                    if execution_thread is not None and execution_thread.is_alive():
                        self._execution_interruption = True
                        execution_thread.join()
                        log.info(f'Waiting for finishing the execution thread: {execution_thread.is_alive()}')
                    self._execution_interruption = False
                    execute_action = result["actions"]
                    # log.info(f'action shape: {execute_action[0].shape} for {episode_num}th episodes')
                    
                def multi_step_tasks():
                    with timer("gym_step", "pi0_inferencer"):
                        dofs = self._gym_robot._robot_motion.get_model_dof_list()[1:]
                        log.info(f'dofs: {dofs}')
                        for i in range(self._execution_steps):
                            if self._execution_interruption or not self._status_ok:
                                break
                            
                            # @TODO: hack for fr3, zyx
                            action_index = 0
                            cur_action = execute_action[i]
                            # log.info(f'Executing action for {i}th action: {cur_action}')
                            with self._lock:
                                joint_state = copy.deepcopy(self._joint_positions)
                            # log.info(f"🚀 Calling update_signal: step {i}, joint_state_len={len(joint_state)}, cur_action_len={len(cur_action)}")
                            self._plotter.update_signal(joint_state, cur_action)
                            
                            # iterates with the dof list
                            action = {'arm': np.array([]), 'tool': np.array([])}
                            gripper_position_dof = self._tool_position_dof
                            log.info(f'len dof: {len(dofs)}')
                            for j in range(len(dofs)):
                                if self._action_type in [ActionType.JOINT_POSITION, ActionType.JOINT_POSITION_DELTA]:
                                    index_l = gripper_position_dof*j + action_index
                                    index_r = gripper_position_dof*j + dofs[j] + action_index
                                    action_index = index_r+gripper_position_dof
                                    log.info(f'arm index for joint: {index_l}, {index_r}')
                                    cur_arm_action = cur_action[index_l:index_r]
                                elif self._action_type in [ActionType.END_EFFECTOR_POSE, ActionType.END_EFFECTOR_POSE_DELTA]:
                                    pose_dof = 6 if self._action_ori_type == "euler" else 7
                                    log.info(f'arm index for pose: {action_index}')
                                    cur_arm_action = cur_action[action_index:action_index+pose_dof]
                                    index_r = action_index+pose_dof
                                    action_index = index_r+gripper_position_dof 
                                else:
                                    raise ValueError(f"Unsupported action type: {self._action_type}")
                
                                action["arm"] = np.hstack((action["arm"], cur_arm_action))
                                cur_tool_action = cur_action[index_r:index_r+gripper_position_dof].copy()
                                log.info(f'cur tool action for {j}: {cur_tool_action}, len {len(cur_tool_action)}')
                                # @TODO: 耦合fr3
                                if self._last_gripper_open:
                                    log.info(f'open, action {cur_tool_action[0]}')
                                    cur_tool_action[0] = 1.0 if cur_tool_action[0] > 0.002 else 0.0
                                else:
                                    log.info(f'close, action {cur_tool_action[0]}')
                                    cur_tool_action[0] = 1.0 if cur_tool_action[0] > 0.03 else 0.0
                                if cur_tool_action[0] > 0.0085:
                                    self._last_gripper_open = True
                                else: self._last_gripper_open = False
                                action["tool"] = np.hstack((action["tool"], cur_tool_action))
                            log.info(f'gym action: {action}, is gripper open: {self._last_gripper_open}')
                            self._gym_robot.step(action)
                            if self._async_execution:
                                time.sleep(0.001)
                            
                execution_thread = threading.Thread(target=multi_step_tasks)
                execution_thread.start()
                if self._async_execution:
                    execution_thread.join()
                log.info(f'result action chunk shape: {execute_action.shape}')
                
    def convert_from_gym_obs(self):
        gym_obs = super().convert_from_gym_obs()
        self.image_display(gym_obs)
            
        pi0_obs = {}
        # @TODO: coupling solution for testing
        pi0_obs["state"] = np.array([])
        temp_joint_positions = np.array([])
        for key, joint_state in gym_obs['joint_states'].items():
            robot_state = joint_state["position"]
            temp_joint_positions = np.hstack((temp_joint_positions, robot_state, gym_obs["tools"][key]["position"]))
            if self._obs_contain_ee:
                robot_state = np.hstack((gym_obs["ee_states"][key]["pose"]))
            pi0_obs["state"] = np.hstack((pi0_obs["state"], robot_state, 
                                        gym_obs["tools"][key]["position"]))
        
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
        
        if not self._key_mapping:
            log.info(f'pi0_obs: {pi0_obs["state"]}')
        # else: log.info(f'pi0_obs: {pi0_obs}')
        return pi0_obs
        
    def convert_to_gym_action(self):
        pass
    
    def close(self):
        """Clean up resources and close display windows."""
        if self._enable_display:
            cv2.destroyWindow(self._display_window_name)
        self._gym_robot.close()
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
    pi0_executor.start_inference()
    
if __name__ == "__main__":
    main()
