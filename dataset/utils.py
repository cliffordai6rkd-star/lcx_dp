import enum

class ActionType(enum.Enum):
    JOINT_POSITION = 0
    JOINT_POSITION_DELTA = 1
    END_EFFECTOR_POSE = 2
    END_EFFECTOR_POSE_DELTA = 3
    JOINT_TORQUE = 4
    
dict_str2action_type = {
    "joint_position": ActionType.JOINT_POSITION,
    "joint_position_delta": ActionType.JOINT_POSITION_DELTA,
    "end_effector_pose": ActionType.END_EFFECTOR_POSE,
    "end_effector_pose_delta": ActionType.END_EFFECTOR_POSE_DELTA,
    "joint_torque": ActionType.JOINT_TORQUE,
}
