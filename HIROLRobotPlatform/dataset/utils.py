import enum

class ObservationType(enum.Enum):
    JOINT_POSITION_ONLY = "joint_position"
    END_EFFECTOR_POSE = "ee_pose"
    DELTA_END_EFFECTOR_POSE = "delta_ee_pose"
    JOINT_POSITION_END_EFFECTOR = "joint_position_ee_pose"
    IMG_ONLY = "img_only"
    MASK = "mask"
    FT_ONLY = "ft_only"

class ActionType(enum.Enum):
    JOINT_POSITION = 0
    JOINT_POSITION_DELTA = 1
    END_EFFECTOR_POSE = 2
    END_EFFECTOR_POSE_DELTA = 3
    JOINT_TORQUE = 4
    COMMAND_JOINT_POSITION = 5
    COMMAND_END_EFFECTOR_POSE = 6

Action_Type_Mapping_Dict = {
    "joint_position": ActionType.JOINT_POSITION,
    "joint_position_delta": ActionType.JOINT_POSITION_DELTA,
    "end_effector_pose": ActionType.END_EFFECTOR_POSE,
    "end_effector_pose_delta": ActionType.END_EFFECTOR_POSE_DELTA,
    "command_joint_position": ActionType.COMMAND_JOINT_POSITION,
    "command_end_effector_pose": ActionType.COMMAND_END_EFFECTOR_POSE
}
