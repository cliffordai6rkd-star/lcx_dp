import time, os, copy, threading
import glog as log
import numpy as np
from typing import Dict, Any, List
from collections import deque

# Base class import
from factory.tasks.inferences_tasks.inference_base import InferenceBase
from dataset.utils import ActionType
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# DP related imports
import torch
import dill
import hydra
from omegaconf import OmegaConf
import sys, cv2, random

# Add diffusion_policy to path
dp_path = "/home/yuxuan/Code/hirol/new_dp/dp_hirol"
sys.path.append(dp_path)
sys.path.append("/home/yuxuan/Code/hirol/new_dp/dp_hirol/diffusion_policy")

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.precise_sleep import precise_wait
import torchvision.transforms as transforms

# Time statistics
from tools.performance_profiler import timer

OmegaConf.register_new_resolver("eval", eval, replace=True)
log.info("Successfully imported diffusion_policy modules")

class DP_Inferencer(InferenceBase):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        
        seed = config["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Load DP model and initialize inference parameters
        self._device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        log.info(f"Using device: {self._device}")
        
        self._dp_policy = self._load_dp_model(config["checkpoint_path"], config)
        self._n_obs_steps = getattr(self._dp_policy, 'n_obs_steps', 2)
        self._n_action_steps = getattr(self._dp_policy, 'n_action_steps', 8)
        self._execution_action_chunk_size = self._n_action_steps
        self._action_horizon = getattr(self._dp_policy, 'horizon', 16)
        log.info(f'To: {self._n_obs_steps}, Ta: {self._n_action_steps}, Tp: {self._action_horizon}')
        
        img_w = config.get(f'image_width', 128)
        img_h = config.get(f'image_height', 128)
        self._image_size = config.get(f'image_size', (img_w, img_h))
        self._obs_queue = deque(maxlen=self._n_obs_steps)
        
        log.info(f"DP model loaded. obs_steps: {self._n_obs_steps}, action_horizon: {self._action_horizon}")
    
    def policy_reset(self):
        self._dp_policy.reset()
        self._obs_queue.clear()
        
    def policy_prediction(self, obs):
        with torch.no_grad():
            start_time = time.perf_counter()
            result = self._dp_policy.predict_action(obs)
            log.info(f'infer used time: {time.perf_counter() - start_time}s')
            log.info(f'dp result action: {result["action"].shape}')
            action_np = result['action'][0].detach().cpu().numpy()
        return action_np
    
    def start_inference(self) -> None:
        """Main inference loop following PI0 pattern."""
        execution_thread = None
        for episode_num in range(self._num_episodes):
            if self._quit:
                break
                
            self._gym_robot.reset()
            self._dp_policy.reset()
            self._obs_queue.clear()
            self._status_ok = True
            self._last_gripper_open = [True, True]
            log.info(f'Starting episode {episode_num}')
            
            while self._status_ok:
                with timer("gym_obs", "dp_inferencer"):
                    dp_obs = self.convert_from_gym_obs()
                    
                with timer("dp_inference_time", "dp_inferencer"):
                    with torch.no_grad():
                        start_time = time.perf_counter()
                        result = self._dp_policy.predict_action(dp_obs)
                        log.info(f'infer used time: {time.perf_counter() - start_time}s')
                        log.info(f'dp result action: {result["action"].shape}')
                        action_np = result['action'][0][:self._n_action_steps].detach().cpu().numpy()
                
                if execution_thread and execution_thread.is_alive():
                    self._execution_interruption = True
                    execution_thread.join()
                    self._execution_interruption = False
                    
                def multi_step_tasks():
                    with timer("gym_step", "pi0_inferencer"):
                        self.convert_to_gym_action(action_np)
                            
                execution_thread = threading.Thread(target=multi_step_tasks)
                execution_thread.start()
                if not self._async_execution:
                    execution_thread.join()
                    
    def convert_from_gym_obs(self, gym_obs = None) -> Dict[str, torch.Tensor]:
        """Convert gym observations to DP format.
        
        Returns:
            Dict containing DP-formatted observations as tensors
        """
        gym_obs = super().convert_from_gym_obs(gym_obs = None)
        self.image_display(gym_obs)
        
        # Convert to DP format
        dp_obs_np = self._convert_gym_obs_to_dp_format(gym_obs)
        
        # Add to observation queue
        if len(self._obs_queue) >= self._n_obs_steps:
            self._obs_queue.popleft()
        self._obs_queue.append(dp_obs_np)
        
        # Wait until we have enough observations
        if len(self._obs_queue) < self._n_obs_steps:
            log.info(f"Collecting observations... ({len(self._obs_queue)}/{self._n_obs_steps})")
            time.sleep(0.01)
            return self.convert_from_gym_obs()  # Recursive call until enough obs
        
        # Stack observations across time dimension
        obs_dict_np = {}
        for i, obs in enumerate(self._obs_queue):
            for key, value in obs.items():
                value = value[None] # Add time dimension
                if i == 0:
                    obs_dict_np[key] = value  
                else:
                    obs_dict_np[key] = np.concatenate((obs_dict_np[key], value), axis=0)
                # log.info(f'{i}th key: {key}, dp obs shape: {obs_dict_np[key].shape}')
        
        # Convert to torch tensors and add batch dimension[1, T, C, H, W]
        obs_dict = dict_apply(obs_dict_np, 
            lambda x: torch.from_numpy(x).unsqueeze(0).to(self._device))
        return obs_dict
        
    def _convert_gym_obs_to_dp_format(self, gym_obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Convert gym observations to DP format.
        
        Args:
            gym_obs: Gym observation dictionary
            
        Returns:
            DP-formatted observation dictionary
        """
        dp_obs = {}
        
        # Process robot state
        state_components = np.array([])
        temp_joint_posi = np.array([])
        for key, cur_state in gym_obs.get('state', {}).items():
            # Extract components
            temp_joint_posi = np.hstack((temp_joint_posi, cur_state))
            # Combine state components
            state_components = np.hstack((state_components, cur_state))
        self._lock.acquire()
        self._joint_positions = temp_joint_posi
        self._lock.release()
        
        dp_obs["state"] = np.array(state_components, dtype=np.float32)
        
        # Process camera observations
        for camera_name, img in gym_obs.get('colors', {}).items():
            assert img is not None and len(img.shape) == 3, f"Invalid image for {camera_name}"
            
            #resize
            img = cv2.resize(img, self._image_size) 
            
            # Normalize and format
            if img.dtype == np.uint8:
                processed_img = img.astype(np.float32) / 255.0
            else:
                processed_img = np.clip(img.astype(np.float32), 0.0, 1.0)
            
            # Convert to [1, C, H, W] format
            processed_img = np.transpose(processed_img, (2, 0, 1))
            dp_obs[camera_name] = processed_img
        
        return dp_obs
        
    def _load_dp_model(self, checkpoint_path: str, config: Dict[str, Any]) -> BaseImagePolicy:
        """Load DP model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            config: Configuration dictionary containing inference settings

        Returns:
            Loaded DP model

        Raises:
            ValueError: If checkpoint file not found or invalid
        """
        assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
        
        log.info(f"Loading DP checkpoint: {checkpoint_path}")
        payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        
        # Create workspace and load model
        cls = hydra.utils.get_class(cfg._target_)
        workspace: BaseWorkspace = cls(cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        # Get policy (EMA if available)
        policy: BaseImagePolicy = workspace.ema_model if cfg.training.use_ema else workspace.model
        policy.eval().to(self._device)
        
        # Configure inference parameters
        if hasattr(policy, 'num_inference_steps'):
            log.info(f"policy infer steps: {getattr(policy, 'num_inference_steps', 25)}")

            # Setup DDIM scheduler if requested
            if config.get('inference_scheduler_type', 'ddpm').lower() == 'ddim':
                self._setup_ddim_scheduler(policy, config)
            else:
                policy.num_inference_steps = min(70, getattr(policy, 'num_inference_steps', 16))

            policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
            log.info(f'DP model num infer steps: {policy.num_inference_steps}, n action steps: {policy.n_action_steps}')

        log.info("DP model loaded successfully")
        return policy

    def _setup_ddim_scheduler(self, policy: BaseImagePolicy, config: Dict[str, Any]) -> None:
        """Setup DDIM scheduler for the policy.

        Args:
            policy: The loaded diffusion policy
            config: Configuration dictionary containing DDIM settings
        """
        try:
            # Import DDIM scheduler
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler

            # Get DDIM configuration parameters
            ddim_steps = config.get('ddim_inference_steps', 16)
            ddim_eta = config.get('ddim_eta', 0.0)

            # Get original scheduler parameters if available
            original_scheduler = getattr(policy, 'noise_scheduler', None)
            if original_scheduler is not None:
                # Create DDIM scheduler with original parameters
                ddim_scheduler = DDIMScheduler(
                    num_train_timesteps=getattr(original_scheduler, 'config', {}).get('num_train_timesteps', 100),
                    beta_start=getattr(original_scheduler, 'config', {}).get('beta_start', 0.0001),
                    beta_end=getattr(original_scheduler, 'config', {}).get('beta_end', 0.02),
                    beta_schedule=getattr(original_scheduler, 'config', {}).get('beta_schedule', 'squaredcos_cap_v2'),
                    clip_sample=getattr(original_scheduler, 'config', {}).get('clip_sample', True),
                    set_alpha_to_one=True,
                    steps_offset=0,
                    prediction_type=getattr(original_scheduler, 'config', {}).get('prediction_type', 'epsilon')
                )
            else:
                # Create DDIM scheduler with default parameters
                ddim_scheduler = DDIMScheduler(
                    num_train_timesteps=100,
                    beta_start=0.0001,
                    beta_end=0.02,
                    beta_schedule='squaredcos_cap_v2',
                    clip_sample=True,
                    set_alpha_to_one=True,
                    steps_offset=0,
                    prediction_type='epsilon'
                )

            # Replace the scheduler
            policy.noise_scheduler = ddim_scheduler
            policy.num_inference_steps = ddim_steps

            log.info(f"Successfully setup DDIM scheduler with {ddim_steps} inference steps and eta={ddim_eta}")

        except ImportError as e:
            log.error(f"Failed to import DDIM scheduler: {e}")
            raise ValueError("DDIM scheduler not available. Please install diffusers library.")
        except Exception as e:
            log.error(f"Failed to setup DDIM scheduler: {e}")
            # Fallback to original inference steps setting
            policy.num_inference_steps = min(70, getattr(policy, 'num_inference_steps', 16))
            log.warning("Falling back to original DDPM inference settings")
    
    def close(self) -> None:
        """Clean up DP-specific resources."""
        if hasattr(self, '_dp_policy'):
            del self._dp_policy
        super().close()
          
def main():
    from factory.utils import parse_args
    from hardware.base.utils import dynamic_load_yaml
    # testing gym api
    arguments = {"config": {"short_cut": "-c",
                            "symbol": "--config",
                            "type": str, 
                            "default": "factory/tasks/inferences_tasks/dp/config/fr3_dp_ddim_inference_cfg.yaml",
                            "help": "Path to the config file"}}
    args = parse_args("dp inference", arguments)
    
    # Load configuration from the YAML file
    config = dynamic_load_yaml(args.config)
    print(f'dp config: {config}')
    dp_executor = DP_Inferencer(config)
    # dp_executor.start_inference()
    dp_executor.start_common_inference()
    
if __name__ == "__main__":
    main()
    