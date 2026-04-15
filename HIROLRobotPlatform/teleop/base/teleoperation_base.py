import numpy as np
import abc
from hardware.base.utils import convert_homo_2_7D_pose, negate_pose
import glog as log

class TeleoperationDeviceBase(abc.ABC, metaclass=abc.ABCMeta):
    def __init__(self, config):
        self._config = config
        self._is_initialized = False
        self._is_initialized = self.initialize()
    
    def require_axis_alignment(self) -> bool:
        return self._config.get('axis_alignment', False)
    
    @abc.abstractmethod
    def initialize(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def print_data(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def read_data(self):
        """
            This function is a inside function, could 
            not be called from external
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def parse_data_2_robot_target(self, mode: str) -> tuple[bool, dict, dict]:
        """Parse the data read from the teleoperation device to a robot target command.
            @params: mode: str, the mode of the robot command, ['absolute', 'relative']
            @return: 
                whether successfully get the data from the device: bool
                The dict with the key: ['single', 'left', 'right'] indicates which part of 
                robot's target; values: the 7D end effector pose target
                The second dict with same key; and the value indicates the devices other output
                which could be used for gripper/hand control
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def close(self):
        raise NotImplementedError
    
    
    def _init_tracker_robot_axis_alignment(self):
        self._require_axis_alignment = self._config.get(f'axis_alignment', False)
        # Optional axis mapping between cube frame and robot EE frame
        control_frame = self._config.get("control_frame", None)
        if not self._require_axis_alignment or control_frame is None:
            self._tracker_to_robot_pose = None
            self._robot_to_cube_pose = None
            return
         
        tracker_forward = control_frame.get("tracker_forward", "+z")
        tracker_up = control_frame.get("tracker_up", "+y")
        robot_forward = control_frame.get("robot_forward", "+z")
        robot_up = control_frame.get("robot_up", "+y")
        axis_map = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
        }
        axis_specs = {
            "tracker_forward": tracker_forward,
            "tracker_up": tracker_up,
            "robot_forward": robot_forward,
            "robot_up": robot_up,
        }
        axis_vecs = {}
        for name, axis in axis_specs.items():
            axis = axis.strip().lower()
            sign = 1.0
            if axis.startswith("-"):
                sign = -1.0
                axis = axis[1:]
            elif axis.startswith("+"):
                axis = axis[1:]
            if axis not in axis_map:
                raise ValueError(f"Invalid axis spec: {axis_specs[name]}")
            axis_vecs[name] = sign * axis_map[axis]

        tracker_fwd = axis_vecs["tracker_forward"]
        tracker_upv = axis_vecs["tracker_up"]
        if abs(np.dot(tracker_fwd, tracker_upv)) > 1e-6:
            raise ValueError(f"Cube forward/up must be orthogonal: {tracker_forward}, {tracker_up}")
        tracker_fwd = tracker_fwd / np.linalg.norm(tracker_fwd)
        tracker_upv = tracker_upv / np.linalg.norm(tracker_upv)
        tracker_right = np.cross(tracker_upv, tracker_fwd)
        if np.linalg.norm(tracker_right) < 1e-8:
            raise ValueError(f"Cube forward/up produce degenerate basis: {tracker_forward}, {tracker_up}")
        tracker_right = tracker_right / np.linalg.norm(tracker_right)
        tracker_upv = np.cross(tracker_fwd, tracker_right)
        tracker_basis = np.stack((tracker_right, tracker_upv, tracker_fwd), axis=1)

        robot_fwd = axis_vecs["robot_forward"]
        robot_upv = axis_vecs["robot_up"]
        if abs(np.dot(robot_fwd, robot_upv)) > 1e-6:
            raise ValueError(f"Robot forward/up must be orthogonal: {robot_forward}, {robot_up}")
        robot_fwd = robot_fwd / np.linalg.norm(robot_fwd)
        robot_upv = robot_upv / np.linalg.norm(robot_upv)
        robot_right = np.cross(robot_upv, robot_fwd)
        if np.linalg.norm(robot_right) < 1e-8:
            raise ValueError(f"Robot forward/up produce degenerate basis: {robot_forward}, {robot_up}")
        robot_right = robot_right / np.linalg.norm(robot_right)
        robot_upv = np.cross(robot_fwd, robot_right)
        robot_basis = np.stack((robot_right, robot_upv, robot_fwd), axis=1)

        tracker_to_robot = robot_basis @ tracker_basis.T
        tracker_to_robot_homo = np.eye(4)
        tracker_to_robot_homo[:3, :3] = tracker_to_robot
        self._tracker_to_robot_pose = convert_homo_2_7D_pose(tracker_to_robot_homo)
        self._robot_to_cube_pose = negate_pose(self._tracker_to_robot_pose)
        if not np.allclose(tracker_to_robot, np.eye(3), atol=1e-6):
            log.info(
                "CubePoseTracker axis mapping enabled: "
                f"cube {tracker_forward}/{tracker_up} -> robot {robot_forward}/{robot_up}"
            )

    