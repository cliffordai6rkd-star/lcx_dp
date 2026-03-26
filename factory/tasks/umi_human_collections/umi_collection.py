from factory.tasks.umi_human_collections.robot_agonostic_collection import RobotAgnosticCollection
from simulation.mujoco.mujoco_sim import MujocoSim
import glog as log
import os
import yaml
import copy
import time
class UmiCollection(RobotAgnosticCollection):
    def __init__(self, config):
        super().__init__(config)
        
        self._target_pose_key = {
            "single": "targetR",
            "left": "targetL",
            "right": "targetR",
            "head": "targetH",
        }

    def create_umi_system(self):
        self.create_system()
        
    def _on_tool(self, tools, ee_states):
        ee_tools = {}
        for tool_key, tool in tools.items():
            if tool_key not in ee_states:
                raise ValueError(f"tool {tool_key} not in poses {list(ee_states.keys())}")
            ee_tools[tool_key] = dict(position=float(tool[0]), time_stamp=time.perf_counter())
        return ee_tools

if __name__ == "__main__":
    from factory.utils import parse_args
    from hardware.base.utils import dynamic_load_yaml

    arguments = {
        "config": {
            "short_cut": "-c",
            "symbol": "--config",
            "type": str,
            "default": "factory/tasks/umi_collections/config/umi_collection_left_only_cfg.yaml",
            "help": "Path to the config file",
        },
    }
    args = parse_args("umi data collection", arguments)

    config = dynamic_load_yaml(args.config)
    umi_collection = UmiCollection(config)
    log.info("Created umi collection system")
    umi_collection.create_umi_system()

    umi_collection.collect_data()
    log.info("Finished umi collection process")
