from factory.components.robot_factory import RobotFactory
from factory.components.motion_factory import MotionFactory
import glog as log
import abc, time
import gymnasium as gym
from dataset.utils import ActionType, dict_str2action_type
import numpy as np
from factory.components.motion_factory import Robot_Space

class GymApi(gym.Env):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self._max_step_nums = config["max_step_nums"]
        self._action_type = config.get("action_type", "joint_position")
        self._action_type = dict_str2action_type[self._action_type]
        robot_motion_cfg = config["motion_config"]
        self._reset_space = config.get("reset_space", "joint")
        self._reset_space = Robot_Space.JOINT_SPACE if self._reset_space== 'joint' else Robot_Space.CARTESIAN_SPACE
        self._reset_joint_position = config.get("reset_position", None)
        self._robot_system = RobotFactory(robot_motion_cfg)
        self._robot_motion = MotionFactory(robot_motion_cfg, self._robot_system)
        self._robot_motion.create_motion_components()
        self._robot_motion.update_execute_hardware(True)
        log.info("The robot motion component is successfully created in gym api!")
        
        # variable used for gym api
        self._step_counter = 0
        self.reset()
        
    def set_action_type(self, action_type: ActionType):
        self._action_type = action_type
        
    def step(self, action):
        # action execution
        arm_action = action['arm']
        cur_joint_state = self._robot_system.get_joint_states()
        if self._action_type == ActionType.END_EFFECTOR_POSE:
            # pose in 7d represented as [xyz, qxyzw]
            self._robot_motion.update_high_level_command(arm_action)
        elif self._action_type == ActionType.JOINT_POSITION:
            # @TODO: implement this 
            self._robot_motion.set_joint_positions(arm_action, True)
        elif self._action_type == ActionType.JOINT_POSITION_DELTA:
            arm_action += cur_joint_state._positions
            self._robot_motion.set_joint_positions(arm_action, True)
        elif self._action_type == ActionType.END_EFFECTOR_POSE_DELTA:
            pass
        else:
            raise ValueError(f'Not support for the action type: {self._action_type}')
        
        # tool execution, @TODO:
        tool_action = action['tool']
        self._robot_system.set_tool_command(tool_action)
        
        # obs
        observation = self.get_observation()
        reward, done = self.compute_rewards()
        done = done or (self._step_counter >= self._max_step_nums)
        info = self.get_info()
        
        return observation, reward, done, False, info
        
    def reset(self, *, seed = None, options = None):
        self._robot_motion.reset_robot_system(arm_command=None,
                                              space=Robot_Space.JOINT_SPACE,
                                              tool_command=dict(single=np.array([1])))
    
    @abc.abstractmethod
    def get_observation(self):
        # joints
        current_joint_state = self._robot_system.get_joint_states()
        
        # end effectors + tools 
        end_effector_names = self._robot_motion.get_model_end_effector_link_list()
        ee_index = ['left', 'right'] if len(end_effector_names) > 1 else ['single']
        model_types = self._robot_motion.get_model_types()
        poses = {}; joint_states = {}
        tools_dict = self._robot_system.get_tool_dict_state()
        for i, ee_name in enumerate(end_effector_names):
            cur_model_type = model_types[i] if len(model_types) > 1 else model_types[0]
            frame_pose = self._robot_motion.get_frame_pose(ee_name, 
                                                           cur_model_type)
            key = ee_index[i]
            poses[key] = frame_pose
            joint_states[key] = {} 
            sliced_joint_states = self._robot_motion.get_type_joint_state(current_joint_state, key)
            joint_states[key]["position"] = sliced_joint_states._positions
            joint_states[key]["velocity"] = sliced_joint_states._velocities
            joint_states[key]["acceleration"] = sliced_joint_states._accelerations
            
        # cameras
        cameras_data = self._robot_system.get_cameras_infos()
        if cameras_data is not None:
            cur_colors = {}
            cur_depths = {}
            cur_imus = {}
            for cam_data in cameras_data:
                name = cam_data['name']
                if 'color' in name:
                    cur_colors[name] = cam_data['img']
                if 'depth' in name:
                    cur_depths[name] = cam_data['img']
                if 'imu' in name:
                    cur_imus[name] = cam_data['imu']
    
        # other sensors: @TODO: zyx
        
        obs_dict = {'joint_states': joint_states, 'ee_states': poses, 
                    'tools': tools_dict, 'colors': cur_colors, 
                    'depths': cur_depths}
        return obs_dict
    
    @abc.abstractmethod
    def compute_rewards(self):
        """
            Please implement your own reward logistics
        """
        return 0.0, False

    @abc.abstractmethod
    def get_info(self):
        """
            Please implement your own reward logistics
        """
        return {}
    

def main():
    from factory.utils import parse_args
    from hardware.base.utils import dynamic_load_yaml
    import os, random, cv2
    # testing gym api
    arguments = {"config": {"short_cut": "-c",
                            "symbol": "--config",
                            "type": str, 
                            "default": "factory/tasks/inferences_tasks/pi0/config/fr3_pi0_cfg.yaml",
                            "help": "Path to the config file"}}
    args = parse_args("pi0 inference", arguments)
    
    # Load configuration from the YAML file
    config = dynamic_load_yaml(args.config)
    
    fr3_gym = GymApi(config)
    i = 0
    # position_test = [0.013841196342567696, -0.5922128618960336, -0.023749771665655426, -2.4364204955424196, -0.016736788670502524, 1.921816076394133, 0.7890845040801358] # measure 
    position_test = [-0.17956917997172686, 0.31556980687824737, 
                     0.11859001558572345, -2.032536791881043, 
                     -0.06755871904469697, 2.4087075313408386, 
                     0.8794838978929802] # measure
    start_position = [-0.0016708960744338654, -0.7861180279768628, 
                      0.0011242960248884306, -2.3515749674770583, 
                      -0.0019059956649008665, 1.581831610375827, 
                      0.7904221443917989]
    is_start = False
    while True:
        i += 1
        if i % 1000 == 0:
            position = position_test if is_start else start_position
            is_start = not is_start
            action = {"arm": position, "tool": dict(single=np.array([random.choice([0, 1])]))}
            log.info(f'action: {action}')
            res = fr3_gym.step(action)
            obs = res[0]
            for key, image in obs["colors"].items():
                # log.info(f'key: {key}, iamge: {image}')
                cv2.imshow(key, image)
                cv2.waitKey(1)
            log.info(f'joint command: {position}')
        time.sleep(0.001)
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()
    