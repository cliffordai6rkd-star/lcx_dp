from factory.tasks.umi_human_collections.robot_agonostic_collection import RobotAgnosticCollection


class HumanCollection(RobotAgnosticCollection):
    def __init__(self, config):
        super().__init__(config)

    def create_human_system(self):
        self.create_system()

if __name__ == "__main__":
    from factory.utils import parse_args
    from hardware.base.utils import dynamic_load_yaml

    arguments = {
        "config": {
            "short_cut": "-c",
            "symbol": "--config",
            "type": str,
            "default": "factory/tasks/umi_human_collections/config/umi_collection_left_only_cfg.yaml",
            "help": "Path to the config file",
        },
    }
    args = parse_args("human data collection", arguments)

    config = dynamic_load_yaml(args.config)
    human_collection = HumanCollection(config)
    human_collection.create_human_system()
    human_collection.collect_data()
