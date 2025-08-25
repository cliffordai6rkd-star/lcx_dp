from factory.tasks.robot_teleoperation import TeleoperationFactory
from factory.components.motion_factory import MotionFactory
from factory.components.robot_factory import RobotFactory
from factory.utils import parse_args
from hardware.base.utils import dynamic_load_yaml
import os

if __name__ == '__main__':
    arguments = {"config": {"short_cut": "-c",
                            "symbol": "--config",
                            "type": str, 
                            "default": "teleop/config/franka_3d_mouse.yaml",
                            "help": "Path to the config file"}}
    args = parse_args("Teleoperation for manipulation", arguments)
    
    # Load configuration from the YAML file
    cur_path = os.path.dirname(os.path.abspath(__file__))
    config = dynamic_load_yaml(args.config)

    # create robot teleoperation system
    motion_config = config["motion_config"]
    robot_system = RobotFactory(motion_config)
    robot_motion_system = MotionFactory(motion_config, robot_system)
    robot_teleoperation = TeleoperationFactory(config, robot_motion_system)
    print(f'Started the teleoperation system')
    robot_teleoperation.create_robot_teleoperation_system()
    
    print(f'finished teleoperation process!!!')
    