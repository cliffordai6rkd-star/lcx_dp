import threading, time, cv2, os, random
import glog as log
import numpy as np
from factory.tasks.inferences_tasks.inference_base import InferenceBase
import copy
from dataset.utils import ActionType

# pi0 related
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config

# time statistics
from tools.performance_profiler import timer

class PI0_Inferencer(InferenceBase):
    def __init__(self, config):
        super().__init__(config)
        self._tasks = config["tasks"]
        self._execution_steps = config.get("execution_step", 5)
        self._execution_interruption = False
        
        # model loading
        model_cfg_name = config["model_cfg_name"]
        model_config = _config.get_config(model_cfg_name)
        model_dir = config["model_dir"]
        checkpoint_dir = download.maybe_download(model_dir)

        # Create a trained policy.
        self._pi0_policy = _policy_config.create_trained_policy(model_config, 
                                                    checkpoint_dir, default_prompt=self._tasks[0])
        
    def start_inference(self):
        execution_thread = None
        for episode_num in range(self._num_episodes):
            if self._quit: break
            
            self._gym_robot.reset()
            self._pi0_policy.reset()
            self._status_ok = True
            log.info(f'Starting the {episode_num} th episodes')
            while self._status_ok:
                with timer("gym_obs", "pi0_inferencer"):
                    pi0_obs = self.convert_from_gym_obs()
                    
                with timer("pi0_inference_time", "pi0_inferencer"):
                    result = self._pi0_policy.infer(pi0_obs)
                    
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
                                cur_tool_action = cur_action[index_r:index_r+gripper_position_dof]
                                log.info(f'cur tool action for {j}: {cur_tool_action}')
                                # @TODO: 耦合fr3
                                cur_tool_action[0] = 1.0 if cur_tool_action[0] > 0.055 else 0.0
                                action["tool"] = np.hstack((action["tool"], cur_tool_action))
                            log.info(f'gym action: {action}')
                            self._gym_robot.step(action)
                            time.sleep(0.001)
                            
                execution_thread = threading.Thread(target=multi_step_tasks)
                execution_thread.start()
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
        log.info(f'pi0_obs: {pi0_obs["state"]}')
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
                            "default": "factory/tasks/inferences_tasks/pi0/config/fr3_pi0_cfg.yaml",
                            "help": "Path to the config file"}}
    args = parse_args("pi0 inference", arguments)
    
    # Load configuration from the YAML file
    config = dynamic_load_yaml(args.config)
    log.info(f'pi0 config: {config}')
    pi0_executor = PI0_Inferencer(config)
    pi0_executor.start_inference()
    
if __name__ == "__main__":
    main()
    