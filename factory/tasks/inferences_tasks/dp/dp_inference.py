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
        
        # DP-specific configurations
        self._frequency = config.get("frequency", 800.0)
        self._async_execution = config.get('aync_execution', False)
        self._dt = 1.0 / self._frequency
        self._last_gripper_open = True
        
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
        self._action_horizon = getattr(self._dp_policy, 'horizon', 16)
        log.info(f'To: {self._n_obs_steps}, Ta: {self._n_action_steps}, Tp: {self._action_horizon}')
        
        self._obs_queue = deque(maxlen=self._n_obs_steps)
        self._action_interruption = False
        self._execution_thread = None
        
        log.info(f"DP model loaded. obs_steps: {self._n_obs_steps}, action_horizon: {self._action_horizon}")
        
    def start_inference(self) -> None:
        """Main inference loop following PI0 pattern."""
        for episode_num in range(self._num_episodes):
            if self._quit:
                break
                
            self._gym_robot.reset()
            self._dp_policy.reset()
            self._obs_queue.clear()
            self._status_ok = True
            self._last_gripper_open = True
            log.info(f'Starting episode {episode_num}')
            
            while self._status_ok:
                with timer("gym_obs", "dp_inferencer"):
                    dp_obs = self.convert_from_gym_obs()
                    
                with timer("dp_inference_time", "dp_inferencer"):
                    with torch.no_grad():
                        start_time = time.perf_counter()
                        result = self._dp_policy.predict_action(dp_obs)
                        log.info(f'infer used time: {time.perf_counter() - start_time}')
                        log.info(f'dp result action: {result["action"].shape}')
                        action_np = result['action'][0][:self._n_action_steps].detach().cpu().numpy()
                    
                with timer("gym_step", "dp_inferencer"):
                    if self._execution_thread and self._execution_thread.is_alive() and self._async_execution:
                        self._action_interruption = True
                        self._execution_thread.join()
                        self._action_interruption = False
                    self.convert_to_gym_action(action_np)
                    
    def convert_from_gym_obs(self) -> Dict[str, torch.Tensor]:
        """Convert gym observations to DP format.
        
        Returns:
            Dict containing DP-formatted observations as tensors
        """
        gym_obs = super().convert_from_gym_obs()
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
        
        # Resize tensors if needed
        obs_dict = self._resize_observation_tensors(obs_dict)
        
        return obs_dict
    
    def convert_to_gym_action(self, dp_action: np.ndarray) -> None:
        """Convert DP action to gym format and execute.
        
        Args:
            dp_action: Action array from DP model
        """
        gym_actions = self._convert_dp_action_to_gym_format(dp_action)
        # log.info(f'gym action" {gym_actions}')
        self._execute_action_sequence(gym_actions)
        
    def _execute_action_sequence(self, gym_actions: List[Dict]) -> None:
        """Execute action sequence with timing control.
        
        Args:
            gym_actions: List of gym action dictionaries
        """
        def async_execute():
            log.info(f'start to exeute action!!!!!')
            for action in gym_actions:
                log.info(f'async action: {action["tool"]}')
                if self._action_interruption:
                    log.info("Action execution interrupted")
                    break
                    
                # Process gripper action, @TODO: 耦合状态 for fr3
                if self._last_gripper_open:
                    log.info(f'open action["tool"][0]')
                    action["tool"][0] = 1.0 if action["tool"][0] > 0.076 else 0.0
                else:
                    log.info(f'close action["tool"][0]')
                    action["tool"][0] = 1.0 if action["tool"][0] > 0.0785 else 0.0
                self._last_gripper_open = True if action["tool"][0] > 0.01 else False
                
                start_time = time.perf_counter()
                self._gym_robot.step(action)
                log.info(f'step used time: {time.perf_counter() - start_time}')
                if self._async_execution:
                    time.sleep(self._dt)
            log.info(f'execution action finished!!!!!1')
        
        self._execution_thread = threading.Thread(target=async_execute)
        self._execution_thread.start()
        if not self._async_execution:
            self._execution_thread.join()
        
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

    def _resize_observation_tensors(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Resize observation tensors to match model expectations.
        
        Args:
            obs_dict: Dictionary of observation tensors
            
        Returns:
            Dictionary with resized tensors
        """
        for key, tensor in obs_dict.items():
            if 'camera' in key and hasattr(self._dp_policy, 'obs_encoder'):
                # Find expected shape from model
                for net_key, net in self._dp_policy.obs_encoder.obs_nets.items():
                    if net_key == key and hasattr(net, 'input_shape'):
                        expected_shape = net.input_shape
                        actual_shape = tensor.shape[3:]  # Remove batch and time, channel dims
                        log.info(f'{key} expected shape: {expected_shape[-2:]}, actual: {actual_shape}')
                        if actual_shape != expected_shape[-2:]:
                            target_h, target_w = expected_shape[-2:]
                            obs_dict[key] = self._resize_tensor(tensor, (target_h, target_w), key)
        return obs_dict
    
    def _resize_tensor(self, tensor: torch.Tensor, target_size: tuple, key: str) -> torch.Tensor:
        assert len(target_size) == 2, f"Target size must be (H, W), got {target_size}"
        assert len(tensor.shape) >= 3, f"Tensor must have at least 3 dims, got {tensor.shape}"
        
        target_h, target_w = target_size
        assert 0 < target_h <= 1024 and 0 < target_w <= 1024, f"Invalid size: {target_size}"
        
        original_shape = tensor.shape
        
        # Handle different tensor dimensions
        if len(tensor.shape) == 4:  # [B, C, H, W]
            resized = transforms.Resize(size=target_size)(tensor)
        elif len(tensor.shape) == 5:  # [B, T, C, H, W]
            batch_size, time_steps = tensor.shape[:2]
            # Reshape to [B*T, C, H, W] for resize
            reshaped = tensor.view(-1, *tensor.shape[2:])
            resized_flat = transforms.Resize(size=target_size)(reshaped)
            # Reshape back to [B, T, C, H_new, W_new]
            resized = resized_flat.view(batch_size, time_steps, *resized_flat.shape[1:])
        else:
            raise ValueError(f"Unsupported tensor shape: {tensor.shape}")
            
        log.info(f"Resized {key}: {original_shape} -> {resized.shape}")
        return resized

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
        for key, joint_state in gym_obs.get('joint_states', {}).items():
            # Extract components
            robot_state = np.array(joint_state["position"], dtype=np.float32)
            ee_pose = gym_obs.get('ee_states', {})
            if self._obs_contain_ee: robot_state = np.hstack((robot_state, ee_pose[key]["pose"]))
            tools_data = gym_obs.get("tools", {})[key]["position"]
            temp_joint_posi = np.hstack((temp_joint_posi, joint_state["position"], tools_data))
            
            # Combine state components
            state_components = np.hstack((state_components, robot_state, tools_data))
        self._lock.acquire()
        self._joint_positions = temp_joint_posi
        self._lock.release()
        
        dp_obs["state"] = np.array(state_components, dtype=np.float32)
        
        # Process camera observations
        for camera_name, img in gym_obs.get('colors', {}).items():
            assert img is not None and len(img.shape) == 3, f"Invalid image for {camera_name}"
            
            #resize
            img = cv2.resize(img, (128, 128)) 
            
            # Normalize and format
            if img.dtype == np.uint8:
                processed_img = img.astype(np.float32) / 255.0
            else:
                processed_img = np.clip(img.astype(np.float32), 0.0, 1.0)
            
            # Convert to [1, C, H, W] format
            processed_img = np.transpose(processed_img, (2, 0, 1))
            dp_obs[camera_name] = processed_img
        
        return dp_obs
    
    def _convert_dp_action_to_gym_format(self, dp_action: np.ndarray) -> List[Dict[str, Any]]:
        """Convert DP action to gym format.
        
        Args:
            dp_action: DP action array
            
        Returns:
            List of gym action dictionaries
        """
        log.info(f'dp action shape: {dp_action.shape}')
        gym_actions = []
        
        # @TODO: hack, zyx
        gripper_position_dof = self._tool_position_dof
        dofs = self._gym_robot._robot_motion.get_model_dof_list()[1:]
        for i in range(dp_action.shape[0]):
            with self._lock:
                joint_state = copy.deepcopy(self._joint_positions)
            # self._plotter.update_signal(joint_state, dp_action[i])
            
            arm_action = np.array([]); gripper_action = np.array([])
            action_index = 0
            for j in range(len(dofs)):
                if self._action_type in [ActionType.JOINT_POSITION, ActionType.JOINT_POSITION_DELTA]:
                    index_l = gripper_position_dof*j + action_index
                    index_r = gripper_position_dof*j + dofs[j] + action_index
                    action_index = index_r+gripper_position_dof
                    cur_arm_action = dp_action[i][index_l:index_r]
                elif self._action_type in [ActionType.END_EFFECTOR_POSE, ActionType.END_EFFECTOR_POSE_DELTA]:
                    pose_dof = 6 if self._action_ori_type == "euler" else 7
                    cur_arm_action = dp_action[i][action_index:action_index+pose_dof]
                    index_r = action_index+pose_dof
                    action_index = index_r+gripper_position_dof 
                else:
                    raise ValueError(f"Unsupported action type: {self._action_type}")
                
                arm_action = np.hstack((arm_action, cur_arm_action))
                cur_tool_action = dp_action[i][index_r:index_r+gripper_position_dof]
                gripper_action = np.hstack((gripper_action, cur_tool_action))

            gym_actions.append({
                'arm': arm_action.astype(np.float32),
                'tool': gripper_action.astype(np.float32)
            })
        return gym_actions
    
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
    dp_executor.start_inference()
    
if __name__ == "__main__":
    main()
    