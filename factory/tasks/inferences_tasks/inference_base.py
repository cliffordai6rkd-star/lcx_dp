import abc, json
from factory.components.gym_interface import GymApi
from hardware.base.utils import ToolControlMode, rot6d_to_quat, transform_pose
from factory.tasks.inferences_tasks.utils.display import display_images
from factory.tasks.inferences_tasks.utils.plotter import AnimationPlotter
from factory.tasks.inferences_tasks.utils.interpolation import PoseTrajectoryInterpolator
from teleop.aruco_cube_tracker.cube_tracker import CubePoseTracker
from dataset.utils import ActionType, Action_Type_Mapping_Dict, ObservationType
import threading, time, cv2, os, copy
from sshkeyboard import listen_keyboard, stop_listening
import glog as log
import torch as th
import numpy as np
from datetime import datetime
from collections import deque
from factory.tasks.inferences_tasks.utils.action_aggreagator import ActionAggregator, WeightMode
from scipy.spatial.transform import Rotation as R
from dataset.lerobot.data_process import EpisodeWriter, serialize_data

def json_saving(json_file, data):
    dir_path = os.path.dirname(json_file)
    os.makedirs(dir_path, exist_ok=True)
    with open(json_file, "w", encoding="utf-8") as f:
        data = serialize_data(data)
        json.dump(data, f, ensure_ascii=False, indent=4)

class InferenceBase(abc.ABC, metaclass=abc.ABCMeta):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self._gym_robot = GymApi(config)
        self._status_ok = True
        self._episode_start = False
        self._quit = False
        self._data_record = False
        if "data_save_path" in config:
            self._data_record = True
            log.info(f'Enable data record in inference!!!!')
        else:
            log.info(f'Not using the data recorder for inference!!!!')
        
        self._action_type = config["action_type"]
        self._action_type = Action_Type_Mapping_Dict[self._action_type]
        self._action_ori_type = config.get("action_orientation_type", "euler")
        self._obs_type = config.get("observation_type", "jonit_position")
        self._obs_type = ObservationType(self._obs_type)
        # relative or delta
        self._chunk_anchor_mode = config.get("chunk_action_mode", None)
        self._use_relative_pose = config.get("use_relative_pose", False)
        
        # tool related config
        self._tool_position_dof = config.get("tool_position_dof", 1)
        self._tool_min: float | list = config.get(f'tool_min', 0)
        self._tool_max: float | list = config.get(f'tool_max', 90)
        self._tool_control_mode = config.get(f'tool_control_mode', ToolControlMode.BINARY)
        self._last_gripper_open = [True, True]; self._gripper_close_times = [0, 0]
        if self._tool_control_mode == ToolControlMode.BINARY:
            self._tool_open_threshold = config.get(f'tool_open_thresh', 45)
            self._tool_close_threshold = config.get(f'tool_close_thresh', 45)
        self._num_episodes = config.get("num_episodes", 10)
        
        # used for latter aggregation 
        self._max_timestamps = config.get(f"max_timestamps", 1500)
        self._predicted_action_chunks = -1 # policy output action chunk size
        self._infer_frequency = config.get(f'infer_frequency', 10)
        self._weight_mode = WeightMode(config.get('action_weight_mode', "no"))
        self._weight_gain = config.get("action_weight_gain", 0.01)
        self._action_aggregation = None
        self._execution_interruption = False
        self._async_execution = config.get("async_execution", False)
        
        # display
        self._enable_display = config.get("enable_display", True)
        self._display_window_name = config["display_window_name"]
        
        # data saving
        self._save_chunk_dir = config.get("save_chunk_dir", None)
        cur_path = os.path.dirname(os.path.abspath(__file__))
        if self._save_chunk_dir:
            self._task_dir = os.path.join(cur_path, "../../../dataset/data", self._save_chunk_dir)
            self._action_chunk_writer = EpisodeWriter(
                            task_dir=self._task_dir, rerun_log=False)
            self._episode_id = self._action_chunk_writer.episode_id
        
        # model inference stats saving
        self._infer_stats_saving_dir = config.get("infer_stats_saving_dir", None)
        self._infer_stats = {}; self._waiting_infer_stats = False; self._user_feedback = []
        self._grasp_frame = config.get("grasp_check_frame", 15)
        if self._infer_stats_saving_dir is not None:
            now_json = datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + ".json"
            self._infer_stats_saving_dir = os.path.join(cur_path, "../../..", self._infer_stats_saving_dir, now_json)
        
        # @TODO: zyx: model inference teleoperation intervention
        self._teleoperation_cfg = config.get("tele_cfg", None)
        self._teleop_device_type = config.get("tele_device", "cube_tracker")
        self._intervention = False
        if self._teleoperation_cfg:
            self._teleop_device = CubePoseTracker(self._teleoperation_cfg[self._teleop_device_type])
            if not self._teleop_device.initialize():
                raise ValueError(f'{self._teleop_device_type} failed to initialize!')
            self._intervention_init_anchor = {}
            self._tele_init_pose_rot = {}; self._tele_init_pose_rot_inv = {}
        # keyboard listening
        listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                kwargs={"on_press": self._keyboard_on_press, 
                        "until": None, "sequential": False,}, daemon=True)
        listen_keyboard_thread.start()
        
        # animation plotter
        self._joint_positions = None
        self._lock = threading.Lock()
        # Check if plotting should be enabled (can be disabled for performance)
        self._enable_plotting = config.get("enable_plotting", False)
        self._plotter = None
        
    def update_plotter(self, state, action):
        if self._enable_plotting:
            if self._plotter is None:
                joint_state_names = [f'state_{i}' for i in range(len(state))]
                action_names = [f'action_{i}' for i in range(len(action))]
                if len(joint_state_names) < len(action_names):
                    joint_state_names += [f'state_mask_{i}' for i in range(len(action) - len(state))]
                if len(joint_state_names) > len(action_names):
                    action_names += [f'action_mask_{i}' for i in range(len(state) - len(action))]
                self._plotter = AnimationPlotter(joint_state_names, action_names)
                self._plotter.start_animation()
                self._plotter.start_main_thread_updater()
                log.info(f'Plotter is successfully settled and ready for draw plots')
            else:
                if len(state) < len(action):
                    state = np.hstack((state, [0] * (len(action) - len(state))))
                if len(state) > len(action):
                    action = np.hstack((action, [0] * (len(state) - len(action))))
                self._plotter.update_signal(state, action)
    
    @abc.abstractmethod
    def convert_from_gym_obs(self, gym_obs = None):
        if gym_obs is None:
            gym_obs = self._gym_robot.get_observation()
        return gym_obs
    
    def convert_to_gym_action_single_step(self, action, raw_action=None, chunk_anchor=None):
        dofs = self._gym_robot._robot_motion.get_model_dof_list()[1:]
        cur_action = action
        with self._lock:
            joint_state = copy.deepcopy(self._joint_positions)
        self.update_plotter(joint_state, cur_action)
        
        # iterates with the dof list
        action = {'arm': np.array([]), 'tool': np.array([])}
        action_index = 0
        gripper_position_dof = self._tool_position_dof
        ee_links = self._gym_robot._robot_motion.get_model_end_effector_link_list()
        ee_keys = ["single"] if len(ee_links) == 1 else ["left", "right"]
        ee_states = self._gym_robot.get_ee_state()
        if self._gym_robot._contain_head: 
            assert self._action_type in [ActionType.END_EFFECTOR_POSE, ActionType.COMMAND_END_EFFECTOR_POSE]
            dofs.append(7) # hack 占位府
            ee_keys.append("head")
        # log.info(f'len dof: {len(dofs)}')
        for j in range(len(ee_keys)):
            if self._action_type in [ActionType.JOINT_POSITION, ActionType.JOINT_POSITION_DELTA]:
                index_l = action_index
                index_r = action_index + dofs[j]
                action_index = index_r+gripper_position_dof
                # log.info(f'arm index for joint: {index_l}, {index_r}')
                cur_arm_action = cur_action[index_l:index_r]
            elif self._action_type in [ActionType.END_EFFECTOR_POSE, ActionType.END_EFFECTOR_POSE_DELTA]:
                if self._action_ori_type == "euler":
                    pose_dof = 6
                elif self._action_ori_type == "6d_rotation":
                    pose_dof = 9
                else: pose_dof = 7
                index_r = action_index+pose_dof
                # log.info(f'arm index for pose: {action_index}')
                cur_arm_action = cur_action[action_index:action_index+pose_dof].copy()
                cur_arm_action[3:] = raw_action[action_index+3:action_index+pose_dof]
                action_index = index_r+gripper_position_dof 
                if self._action_ori_type == "euler":
                    cur_arm_action = np.hstack((cur_arm_action, [0]))
                    cur_arm_action[3:] = R.from_euler("xyz", cur_arm_action[3:6]).as_quat()
                elif self._action_ori_type == "6d_rotation":
                    rot_6d = cur_arm_action[3:9]
                    cur_arm_action = cur_arm_action[:7]
                    cur_arm_action[3:] = rot6d_to_quat(rot_6d)
                # obs anchor   
                if self._chunk_anchor_mode:
                    if ee_keys[j] == "head": continue
                    assert chunk_anchor is not None
                    log.info(f'chunk anchor len: {chunk_anchor[ee_keys[j]]["pose"].shape}')
                    # cur_obs_anchor = chunk_anchor[j*(7+gripper_position_dof):j*(7+gripper_position_dof)+7]
                    cur_obs_anchor = chunk_anchor[ee_keys[j]]["pose"]
                    cur_arm_action = transform_pose(cur_obs_anchor, cur_arm_action)
            else:
                raise ValueError(f"Unsupported action type: {self._action_type}")

            action["arm"] = np.hstack((action["arm"], cur_arm_action))

            if ee_keys[j] == "head": continue # skip head tool assignment
            cur_tool_action = cur_action[index_r:index_r+gripper_position_dof].copy()
            log.info(f'cur tool action from model action for {j}: {cur_tool_action}, len {len(cur_tool_action)}')
            
            if self._tool_control_mode == ToolControlMode.BINARY:
                if self._last_gripper_open[j]:
                    log.info(f'open, action {cur_tool_action[0]}')
                    cur_tool_action[0] = 1.0 if cur_tool_action[0] > self._tool_open_threshold else 0.0
                else:
                    log.info(f'close, action {cur_tool_action[0]}')
                    cur_tool_action[0] = 1.0 if cur_tool_action[0] > self._tool_close_threshold else 0.0
            else: cur_tool_action /= self._tool_max
            # @TODO: 检查抓取点时机 
            if gripper_position_dof == 1:
                if cur_tool_action[0] > 0.625:
                    self._last_gripper_open[j] = True
                    self._gripper_close_times[j] = 0
                else: 
                    self._last_gripper_open[j] = False
                    self._gripper_close_times[j] += 1
                    log.info(f'gripper close value {cur_tool_action[0]}. close time: {self._gripper_close_times[j]}')
                if self._gripper_close_times[j] == self._grasp_frame and self._infer_stats_saving_dir:
                    # 从开到闭合
                    self._infer_stats[len(self._infer_stats)-1][j].append(ee_states[ee_keys[j]]["pose"])
                    log.info(f'Detected the grasp time: {len(self._infer_stats[len(self._infer_stats)-1][j])} for {j}th ee id and {len(self._infer_stats)-1}th episode!')
            action["tool"] = np.hstack((action["tool"], cur_tool_action))
            # log.info(f'tranformed tool action for {j}: {cur_tool_action}')
        return action
    
    def convert_to_gym_action(self, model_action: np.ndarray):
        """
            @brief: convert action type and execute it one by one
            model_action: dim (chunk_size, action_size)
            assuming all chunk size is need to be executed
        """
        dofs = self._gym_robot._robot_motion.get_model_dof_list()[1:]
        log.info(f'dofs: {dofs}')
        for i in range(model_action.shape[0]):
            if self._execution_interruption or not self._status_ok:
                log.info(f"Action execution interrupted with interruption {self._execution_interruption}, status {self._status_ok}")
                break
            
            start_time = time.perf_counter()
            cur_action = model_action[i]
            with self._lock:
                joint_state = copy.deepcopy(self._joint_positions)
            self.update_plotter(joint_state, cur_action)
            
            # iterates with the dof list
            action = {'arm': np.array([]), 'tool': np.array([])}
            action_index = 0
            gripper_position_dof = self._tool_position_dof
            # log.info(f'len dof: {len(dofs)}')
            convert_start = time.perf_counter()
            for j in range(len(dofs)):
                if self._action_type in [ActionType.JOINT_POSITION, ActionType.JOINT_POSITION_DELTA]:
                    index_l = gripper_position_dof*j + action_index
                    index_r = gripper_position_dof*j + dofs[j] + action_index
                    action_index = index_r+gripper_position_dof
                    # log.info(f'arm index for joint: {index_l}, {index_r}')
                    cur_arm_action = cur_action[index_l:index_r]
                elif self._action_type in [ActionType.END_EFFECTOR_POSE, ActionType.END_EFFECTOR_POSE_DELTA]:
                    pose_dof = 6 if self._action_ori_type == "euler" else 7
                    # log.info(f'arm index for pose: {action_index}')
                    cur_arm_action = cur_action[action_index:action_index+pose_dof]
                    index_r = action_index+pose_dof
                    action_index = index_r+gripper_position_dof 
                else:
                    raise ValueError(f"Unsupported action type: {self._action_type}")

                action["arm"] = np.hstack((action["arm"], cur_arm_action))
                cur_tool_action = cur_action[index_r:index_r+gripper_position_dof].copy()
                log.info(f'cur tool action from model action for {j}: {cur_tool_action}, len {len(cur_tool_action)}')
                
                if self._tool_control_mode == ToolControlMode.BINARY:
                    if self._last_gripper_open[j]:
                        log.info(f'open, action {cur_tool_action[0]}')
                        cur_tool_action[0] = 1.0 if cur_tool_action[0] > self._tool_open_threshold else 0.0
                    else:
                        log.info(f'close, action {cur_tool_action[0]}')
                        cur_tool_action[0] = 1.0 if cur_tool_action[0] > self._tool_close_threshold else 0.0
                    if cur_tool_action[0] > 0.0085:
                        self._last_gripper_open[j] = True
                    else: self._last_gripper_open[j] = False
                else: cur_tool_action /= self._tool_max
                action["tool"] = np.hstack((action["tool"], cur_tool_action))
                # log.info(f'tranformed tool action for {j}: {cur_tool_action}')
            loop_time = time.perf_counter() - convert_start    
            
            # log.info(f'gym action: {action}')
            # time.sleep(0.01)
            step_start = time.perf_counter()
            res = self._gym_robot.step(action)
            step_time = time.perf_counter() - step_start
            dt = time.perf_counter() - start_time
            if dt < 1.0 / 50.0:
                sleep_time = (1.0 / 50) -  dt
                time.sleep(sleep_time)
            else: log.warn(f"{'=='*15} Execution is slow: {1.0 / dt:.3f}Hz {loop_time:.5f} {step_time:.5f}  {res[4]['obs_time']:.5f} {'=='*15}")
            if self._async_execution:
                time.sleep(0.001)

    @abc.abstractmethod
    def policy_reset(self):
        # raise NotImplementedError
        pass
    
    @abc.abstractmethod
    def policy_prediction(self, obs) -> th.Tensor | np.ndarray:
        """
            predict policy action from obs
            action should be numpy action
        """
        # raise NotImplementedError
        pass

    @abc.abstractmethod
    def start_inference(self):
        """Start the main inference loop.
        
        Continuously processes observations, runs inference, and executes actions
        until interrupted by keyboard input.
        """
        raise NotImplementedError
    
    def start_common_inference(self):
        time.sleep(0.001)
        # rollout episodes
        # query_frequency = int(50 / self._infer_frequency)
        infer_dt = 1.0 / self._infer_frequency
        # 50
        query_frequency = 50; num_quries = 50
        origin_chunk_anchor_mode = self._chunk_anchor_mode
        for episode_id in range(self._num_episodes):
            if self._quit: break
            
            if self._data_record:
                if episode_id:
                    self._gym_robot.save_recording()
                    log.info(f'save infer data traj!!!')
                self._gym_robot.start_recording()
                log.info(f'start record infer data traj!!!')
            if self._save_chunk_dir and episode_id:
                self._action_chunk_writer.save_episode()
                log.info(f'save {self._episode_id}th model infer action chunk!!!')
                time.sleep(1.5)
            
            if self._action_aggregation:
                self._action_aggregation.reset()
            self._gym_robot.reset(options={"arm_to_default": True})
            self._last_gripper_open = [True, True]
            self.policy_reset(); self._status_ok = True; obs = None
            
            if self._infer_stats_saving_dir:
                if len(self._infer_stats) != 0:
                    log.info(f'infer stats ee states: {self._infer_stats[episode_id-1]}')
                    # for ee_id, ee_values in self._infer_stats[episode_id-1].items():
                    #     self._infer_stats[episode_id-1][ee_id] = np.vstack(ee_values)
                    self._waiting_infer_stats = True
                    log.info(f'Please input infer succ stats (p for ending input), eg. t(succ) f (failed) x(skip): ')
                    while self._waiting_infer_stats:
                        time.sleep(0.001)
                    log.info(f'Got user feedback: {self._user_feedback}')
                    len_user_feedback = 0
                    for ee_id, ee_values in self._infer_stats[episode_id-1].items():
                        attempt_nums = len(ee_values)
                        cur_spatial_info = []
                        for cur_value_id, cur_ee_value in enumerate(ee_values):
                            if self._user_feedback[len_user_feedback+cur_value_id] == 'x':
                                # skip this unfinished attempts
                                continue
                            
                            success = True if self._user_feedback[len_user_feedback+cur_value_id] == 't' else False
                            cur_spatial_info.append([cur_ee_value.tolist(), success])
                        self._infer_stats[episode_id-1][ee_id] = cur_spatial_info
                        len_user_feedback += attempt_nums
                    log.info(f'modified infer statas: {self._infer_stats[episode_id-1]}')
                    self._user_feedback = []
                # @TODO: zyx testing
                self._infer_stats[episode_id] = {}
                ee_links = self._gym_robot._robot_motion.get_model_end_effector_link_list()
                for ee_id in range(len(ee_links)):
                    self._infer_stats[episode_id][ee_id] = []
                     
            self._episode_start = False; counter = 0
            time.sleep(1.5)
            # while not self._episode_start:
            #     if counter < 5:
            #         log.info(f'Please press s for start!!!!!!!!')
            #         counter += 1
            #     time.sleep(0.001)
            self._gym_robot.reset()
            self._gym_robot.set_init_pose()
            self._intervention = False; self._intervention_init_anchor = {}
            
            if self._save_chunk_dir:
                if self._use_relative_pose and episode_id == 0:
                    # save init pose for the absolute relative pose action space
                    init_anchor = self._gym_robot._init_pose.copy()
                    init_anchor = serialize_data(init_anchor)
                    init_pose_file = os.path.join(self._task_dir, "init_pose.json")
                    with open(init_pose_file, 'w', encoding='utf-8') as f:
                        json.dump(init_anchor, f, indent=4)
                self._action_chunk_writer.create_episode()
                self._episode_id = self._action_chunk_writer.episode_id
                time.sleep(0.5)
                log.info(f'start record model infer action chunk for {self._episode_id}!!!')
            log.info(f'Starting the {episode_id} th episodes')
            
            # inference timestamp
            pred_action_chunk = None; chunk_anchor = None; chunk_started = time.perf_counter()
            async_execute_thread = None; self._execution_index = None        
            for t in range(self._max_timestamps):                    
                if not self._status_ok or self._quit: 
                    break 
                
                if t % num_quries == 0:
                    obs = self.convert_from_gym_obs()
                    chunk_dt = time.perf_counter() - chunk_started
                    chunk_started = time.perf_counter()
                    pred_action_chunk = self.policy_prediction(obs) # [:query_frequency]
                    chunk_shape = pred_action_chunk.shape[0]
                    log.info(f'chunk shape: {pred_action_chunk.shape}, time: {1.0 / chunk_dt}Hz')
                    if self._save_chunk_dir:
                        self._action_chunk_writer.add_item(actions=pred_action_chunk)
                    # update chunk anchor
                    if self._chunk_anchor_mode:
                        # @TODO: sleep could be deleted
                        time.sleep(2.5)
                        chunk_anchor = self._gym_robot.get_ee_state()
                    if self._use_relative_pose:
                        # self._gym_robot.get_camera_infos()
                        obs_timestamp = self._gym_robot.get_obs_timestamp()
                        action_timestamps = (np.arange(len(pred_action_chunk), dtype=np.float64)
                            ) * infer_dt + obs_timestamp
                        # action timestamp check
                        # using 0.005 as the action execution latency
                        exec_latency = 3.5
                        cur_time = time.perf_counter()
                        is_new = action_timestamps > (cur_time + exec_latency)
                        # log.info(f'action ts: {action_timestamps} {cur_time + exec_latency}')
                        if np.sum(is_new) == 0:
                            log.warn(f'action chunk is old, execute the last action from chunk only')
                            self._execution_index = np.array([chunk_shape - 1])
                        else:
                            self._execution_index = np.arange(chunk_shape)[is_new]
                        log.info(f'exeution action index: {self._execution_index}')
                        self._execution_index += t
                            
                        # smooth the entire action chunk before ensembling
                        # pose_intep = PoseTrajectoryInterpolator()
                
                if self._action_aggregation is None:
                    self._action_aggregation = ActionAggregator(query_frequency=query_frequency,
                        chunk_size=chunk_shape, max_timestamps=self._max_timestamps,
                        action_size=pred_action_chunk.shape[1], k=self._weight_gain)
                elif t % num_quries == 0:
                    # update action chunk
                    self._action_aggregation.add_action_chunk(t, pred_action_chunk)

                def execute_one_action(cur_t):
                    start_time = time.perf_counter()
                    
                    # human intervention
                    if self._teleoperation_cfg:
                        raw_status, pose_tgt, tool_tgt = self._teleop_device.parse_data_2_robot_target("absolute_delta") 
                        self._intervention = True if int(raw_status) > 0 else False
                        if int(raw_status) == 0: self._intervention_init_anchor = {}
                    # calculate aggregated action
                    if not self._intervention:
                        aggregated_action = self._action_aggregation.aggregation_action(cur_t, self._weight_mode)
                        raw_action = pred_action_chunk[cur_t%query_frequency]
                        self._chunk_anchor_mode = origin_chunk_anchor_mode
                    else:
                        log.info(f'Human use the teleoperation to intervens the policy output')
                        if len(self._intervention_init_anchor) == 0:
                            ee_states = self._gym_robot.get_ee_state()
                        else: ee_states = None
                        aggregated_action = np.array([])
                        for key, tool_value in tool_tgt.items():
                            if tool_value[-1] and len(self._intervention_init_anchor) != len(list(tool_tgt.keys())):
                                assert ee_states is not None
                                self._intervention_init_anchor[key] = ee_states[key]["pose"]
                                self._tele_init_pose_rot[key] = np.hstack(([0,0,0], ee_states[key]["pose"][3:]))
                                self._tele_init_pose_rot_inv[key] = negate_pose(self._tele_init_pose_rot[key])
                                log.info(f'Updated intervention init anchor for {key}')
                                
                            # change the aggreation action array
                            if not self._teleop_device.require_axis_alignment():
                                pose_tgt[key] = transform_pose(
                                    transform_pose(self._tele_init_pose_rot_inv[key], pose_tgt[key]), self._tele_init_pose_rot[key])
                            cur_action = transform_pose(self._intervention_init_anchor[key], pose_tgt[key])
                            aggregated_action = np.hstack((aggregated_action, cur_action))
                        raw_action = aggregated_action; self._chunk_anchor_mode = None
                    # log.info(f'aggregated action: {aggregated_action}')
                    # smoothed action
                    
                    need_execution = True if not self._use_relative_pose or self._intervention else self._execution_index[0] == cur_t
                    if need_execution:
                        if self._use_relative_pose: self._execution_index = self._execution_index[1:]
                        convert_start = time.perf_counter()
                        gym_action = self.convert_to_gym_action_single_step(
                            aggregated_action, raw_action, chunk_anchor)
                        convert_time = time.perf_counter() - convert_start
                        step_start = time.perf_counter()
                        res = self._gym_robot.step(gym_action, True)
                        if res[0] is not None:
                            self.convert_from_gym_obs(res[0])
                        if self._intervention: self._chunk_anchor_mode = origin_chunk_anchor_mode
                        step_time = time.perf_counter() - step_start
                        
                    dt = time.perf_counter() - start_time
                    if dt < 1.0 / 60.0:
                        sleep_time = (1.0 / 50) -  dt
                        # time.sleep(0.4*sleep_time)
                        time.sleep(0.001)
                    else: 
                        time.sleep(0.001)
                        if need_execution:
                            time_stats = f" {'=='*8} {(1.0/step_time):.5f}HZ {(1.0/convert_time):.5f}HZ"
                        else: time_stats = ' '
                        # {(1.0/step_time):.5f}HZ {(1.0/convert_time):.5f}HZ 
                        log.warn(f"{'=='*8} Execution is slow: {1.0 / dt:.3f}Hz" + time_stats)
                    # time.sleep(0.2)
                   
                def execute_chunk_action(t_start):
                    for i in range(query_frequency):
                        if not self._status_ok or self._quit: 
                            return
                        
                        cur_t = t_start + i
                        execute_one_action(cur_t)
                
                if not self._async_execution:
                    execute_one_action(t)
                    t += 1
                else:
                    if async_execute_thread is not None and async_execute_thread.is_alive():
                        async_execute_thread.join()
                    
                    async_execute_thread = threading.Thread(target=execute_chunk_action, args=(t,))
                    async_execute_thread.start()
                    t = t + query_frequency
                    
                                
    @abc.abstractmethod
    def close(self):
        pass
    
    def image_display(self, gym_obs):
        if self._enable_display and gym_obs.get("colors"):
            display_images(gym_obs["colors"], self._display_window_name)    
    
    
    def _keyboard_on_press(self, key: str) -> None:
        # quit
        if key == 'q':
            log.info("Quit command received, shutting down...")
            print(f"{'='*15}Closing the inference thread!!!{'='*15}")
            self._gym_robot.close()
            stop_listening()
            self._quit = True
            self._status_ok = False
            if self._enable_plotting:
                self._plotter.clear_data()
                self._plotter.stop_animation()
            time.sleep(1.5)
            self.close()
            if self._save_chunk_dir is not None and hasattr(self, "_action_chunk_writer"):
                self._action_chunk_writer.close()
            cv2.destroyAllWindows()
        # reset
        elif key == 'r':
            log.info("Reset command received")
            self._gym_robot.reset()
        elif key == 'd':
            self._status_ok = False
            log.info(f"Set done to True for current episode!!!")
        elif key == 's':
            self._episode_start = True
        elif key == 'c' and self._infer_stats_saving_dir is not None:
            if len(self._infer_stats) != 0:
                json_saving(self._infer_stats_saving_dir, self._infer_stats)
                log.info(f'Successfully saving infer stats to {self._infer_stats_saving_dir}')
        elif self._waiting_infer_stats and key in ['t', 'f', 'x']:
            self._user_feedback.append(key)
            log.info(f'Cur user feedback for last episode inference is {self._user_feedback}')
        elif self._waiting_infer_stats and key == 'p':
            self._waiting_infer_stats = False
            