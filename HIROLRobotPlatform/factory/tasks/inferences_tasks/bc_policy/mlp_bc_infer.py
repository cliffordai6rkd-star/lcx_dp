from dependencies.BC_Policy.models.mlp_bc import load_model
from dependencies.BC_Policy.bc_utils.utils import normalize, denormalize, load_data_stat, LossType
import glog as log
from factory.tasks.inferences_tasks.inference_base import InferenceBase
import cv2
import numpy as np
import torch as th

class MLP_BC_INFERNCER(InferenceBase):
    def __init__(self, config):
        super().__init__(config)
        self._device = config.get("device", "cuda:0")
        self._device = self._device if th.cuda.is_available() else "cpu"
        self._img_shape = tuple(config.get('img_shape', (224, 224)))
        self._model_dir = config["model_dir"]
        self._data_stata_dir = config["data_dir"]
        self._mlp_bc_policy, args = load_model(self._model_dir, 
                            device=self._device, training=False)
        self._mlp_bc_policy.eval().to(self._device)
        log.info(f'img shape: {self._img_shape}')
        log.info(f'mlp bc policy args: {args}')
        self._loss_type = args["loss"]
        self._data_stata = load_data_stat(self._data_stata_dir)
        log.info(f'loaded data stat: {self._data_stata}')
        
    def policy_reset(self):
        pass

    def policy_prediction(self, obs):
        cam_imgs, obs_state = obs 
        output = self._mlp_bc_policy(cam_imgs, obs_state)
        with th.no_grad():
            if self._loss_type == LossType.LOG_PROB:
                action = self._mlp_bc_policy.sample_action(output[0], output[1])
            else: action = output
        action = action.detach().cpu().numpy()
        action = denormalize(action, self._data_stata["action_mean"], self._data_stata["action_std"])
        log.info(f'bc action: {action}, shape: {action.shape}')
        return action
    
    def start_inference(self):
        pass
    
    def convert_from_gym_obs(self, gym_obs=None):
        gym_obs =  super().convert_from_gym_obs(gym_obs)
        self.image_display(gym_obs)
        
        cam_imgs = {}
        for cam_name, img in gym_obs["colors"].items():
            img = cv2.resize(img, self._img_shape)
            img = np.array(img, dtype=np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))[None]
            cam_imgs[cam_name] = th.tensor(img, dtype=th.float32, device=self._device)
            
        observation = np.array([])
        temp_joint_positions = np.array([])
        for key, state in gym_obs["state"].items():
            observation = np.hstack((observation, state))
            temp_joint_positions = np.hstack((temp_joint_positions, state))
        observation = normalize(observation, self._data_stata["obs_mean"], self._data_stata["obs_std"])
        observation = th.tensor(observation, dtype=th.float32, device=self._device)[None]
        
        self._lock.acquire()
        self._joint_positions = temp_joint_positions
        self._lock.release()
        
        return (cam_imgs, observation)
    
    def close(self):
        """Clean up resources and close display windows."""
        if self._enable_display:
            cv2.destroyWindow(self._display_window_name)
        self._gym_robot.close()
        del self._mlp_bc_policy

def main():
    from factory.utils import parse_args
    from hardware.base.utils import dynamic_load_yaml
    # testing gym api
    arguments = {"config": {"short_cut": "-c",
                            "symbol": "--config",
                            "type": str, 
                            "default": "factory/tasks/inferences_tasks/bc_policy/config/fr3_mlp_bc_config.yaml",
                            "help": "Path to the config file"}}
    args = parse_args("pi0 inference", arguments)
    
    # Load configuration from the YAML file
    config = dynamic_load_yaml(args.config)
    log.info(f'mlp bc config: {config}')
    mlp_bc_executor = MLP_BC_INFERNCER(config)
    mlp_bc_executor.start_common_inference()
    
if __name__ == "__main__":
    main()