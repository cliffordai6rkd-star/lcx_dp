from factory.components.gym_interface import GymApi
import threading, time, cv2, os, random
from sshkeyboard import listen_keyboard, stop_listening
import glog as log
import numpy as np

# pi0 related
from openpi.policies import hirol_fr3_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config

# time statistics
from tools.performance_profiler import timer

class PI0_Inferencer:
    def __init__(self, config):
        self._gym_robot = GymApi(config)
        self._status_ok = True
        self._tasks = config["tasks"]
        
        # Display configuration
        self._enable_display = config.get("enable_display", True)
        self._display_window_name = "PI0 Camera Views"
        self._target_image_size = (240, 320)  # (height, width)
        
        listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                        kwargs={"on_press": self._keyboard_on_press, 
                                                "until": None, "sequential": False,}, 
                                        daemon=True)
        listen_keyboard_thread.start()
        
        # model loading
        model_cfg_name = config["model_cfg_name"]
        model_config = _config.get_config(model_cfg_name)
        model_dir = config["model_dir"]
        # "/home/yuxuan/Code/hirol/pi0/checkpoints/pi0_fast_hirol_fr3_peg_in_hole/hirol_test/29999"
        checkpoint_dir = download.maybe_download(model_dir)

        # Create a trained policy.
        self._pi0_policy = _policy_config.create_trained_policy(model_config, checkpoint_dir)
        
    def start_inference(self):
        while self._status_ok:
            with timer("gym_obs", "pi0_inferencer"):
                obs_dict = self._gym_robot.get_observation()
                pi0_obs = {}
                # @TODO: coupling solution for testing
                pi0_obs["state"] = np.array([])
                for key, joint_state in obs_dict['joint_states'].items():
                    pi0_obs["state"] = np.hstack((pi0_obs["state"], joint_state['position'], 
                                                  obs_dict["tools"][key]._position))
                for key, img in obs_dict["colors"].items():
                    pi0_obs[key] = img
                
                # Display images if enabled
                if self._enable_display and obs_dict["colors"]:
                    self._display_images(obs_dict["colors"])
            if isinstance(self._tasks, list):
                selected_index = random.randrange(len(self._tasks))
                pi0_obs["task"] = self._tasks[selected_index]
            else: pi0_obs["task"] = self._tasks
            # print(f'pio_obs: {pi0_obs}')
            with timer("pi0_inference_time", "pi0_inferencer"):
                result = self._pi0_policy.infer(pi0_obs)
                execute_action = result["actions"][0]
                print(f'execute action: {execute_action}')
                action = {'arm': execute_action[:-1], 'tool': dict(single=np.array([execute_action[-1]]))}
            # execute
            with timer("gym_step", "pi0_inferencer"):
                self._gym_robot.step(action=action)
            
    def _calculate_grid_layout(self, num_images: int) -> tuple[int, int]:
        """Calculate optimal grid layout for images.
        
        Args:
            num_images: Number of images to display
            
        Returns:
            (rows, cols): Grid dimensions as close to square as possible
        """
        if num_images <= 0:
            return (0, 0)
        
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
        return (rows, cols)
    
    def _create_image_grid(self, images_dict: dict[str, np.ndarray], 
                          target_size: tuple[int, int] = None) -> np.ndarray:
        """Create a grid of images for display.
        
        Args:
            images_dict: Dictionary of camera names to image arrays
            target_size: Target size for each image (height, width)
            
        Returns:
            Combined grid image
            
        Raises:
            ValueError: If images cannot be processed
        """
        if not images_dict:
            # Return a blank placeholder image
            placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No Images", (80, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            return placeholder
        
        if target_size is None:
            target_size = self._target_image_size
        
        num_images = len(images_dict)
        rows, cols = self._calculate_grid_layout(num_images)
        
        # Standardize all images
        processed_images = []
        for name, img in images_dict.items():
            try:
                # Ensure image is 3-channel BGR
                if len(img.shape) == 2:  # Grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:  # RGBA
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                
                # Resize to target size
                resized_img = cv2.resize(img, (target_size[1], target_size[0]), 
                                       interpolation=cv2.INTER_LINEAR)
                
                # Add text label
                cv2.putText(resized_img, name, (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                processed_images.append(resized_img)
                
            except cv2.error as e:
                log.warning(f"Failed to process image {name}: {e}")
                # Create placeholder
                placeholder = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
                cv2.putText(placeholder, f"Error: {name}", (10, target_size[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                processed_images.append(placeholder)
        
        # Fill remaining grid positions with blank images if needed
        total_positions = rows * cols
        while len(processed_images) < total_positions:
            blank = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
            processed_images.append(blank)
        
        # Create grid
        grid_height = rows * target_size[0]
        grid_width = cols * target_size[1]
        grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        for i, img in enumerate(processed_images):
            row = i // cols
            col = i % cols
            y_start = row * target_size[0]
            y_end = y_start + target_size[0]
            x_start = col * target_size[1]
            x_end = x_start + target_size[1]
            grid_image[y_start:y_end, x_start:x_end] = img
        
        return grid_image
    
    def _display_images(self, images_dict: dict[str, np.ndarray]) -> None:
        """Display images in a unified OpenCV window.
        
        Args:
            images_dict: Dictionary of camera names to image arrays
        """
        try:
            grid_image = self._create_image_grid(images_dict)
            cv2.imshow(self._display_window_name, grid_image)
            cv2.waitKey(1)  # Non-blocking update
            
        except cv2.error as e:
            log.warning(f"OpenCV display error: {e}")
        except ValueError as e:
            log.error(f"Image processing error: {e}")
        except Exception as e:
            log.error(f"Unexpected error in image display: {e}")
    
    def close(self):
        """Clean up resources and close display windows."""
        if self._enable_display:
            cv2.destroyWindow(self._display_window_name)
        self._gym_robot.close()
        
    def _keyboard_on_press(self, key):           
        # quit
        if key == 'q':
            print(f"{'='*15}Closing the teleoperation thread!!!{'='*15}")
            stop_listening()
            self._status_ok = False
            time.sleep(1.5)
            del self._pi0_policy
            self.close()
            cv2.destroyAllWindows()
        # reset
        elif key == 'r':
            self._gym_robot.reset()
          
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
    print(f'pi0 config: {config}')
    pi0_executor = PI0_Inferencer(config)
    pi0_executor.start_inference()
    
if __name__ == "__main__":
    main()
    