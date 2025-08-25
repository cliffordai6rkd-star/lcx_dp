import argparse, os, yaml

def parse_args(description, arguments_dict):
    parser = argparse.ArgumentParser(description=description)
    for key, value in arguments_dict.items():
        short_cut = value["short_cut"]
        symbol = value["symbol"]
        type = value["type"]
        default = value["default"]
        help = value["help"]
        parser.add_argument(short_cut, symbol, type=type, 
                            default=default, help=help)
        # parser.add_argument("-c", "--config", type=str, 
        #                     default="teleop/config/franka_3d_mouse.yaml", 
        #                     help="Path to the config file")
    return parser.parse_args()

def get_cfg(file):
    cur_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(cur_path, "..", file)
    with open(cfg_file, 'r') as stream:
        config = yaml.safe_load(stream)
    return config

