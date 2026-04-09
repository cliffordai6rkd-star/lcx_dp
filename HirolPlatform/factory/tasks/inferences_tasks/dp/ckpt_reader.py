import torch
import dill
import hydra
import glog as log

ckpt_path = "../dp_output/2026.04.06/latest.ckpt" 
with open(ckpt_path, "rb") as f:
    payload = torch.load(f, pickle_module=dill, map_location="cpu")
cfg = payload["cfg"]
log.info(f"target: {cfg._target_}")

# lerobot_v3格式 打印ckpt的shape_meta含义
try:
    log.info(f"shape_meta.obs.state_ee ={cfg.shape_meta.obs.state_ee}")
    log.info(f"shape_meta.action ={cfg.shape_meta.action}")

except Exception:
    pass

cls = hydra.utils.get_class(cfg._target_)
workspace = cls(cfg)
workspace.load_payload(payload, exclude_keys=None, include_keys=None)

policy = workspace.ema_model if cfg.training.use_ema else workspace.model
log.info(f"normalizer keys = {list(policy.normalizer.params_dict.keys())}")

for key in ("state_ee", "state"):
    if key in policy.normalizer.params_dict:
        log.info(f"\nnormalizer key = {key}")
        stats = policy.normalizer[key].get_input_stats()
        for name, value in stats.items():
            print(name, value.shape, value[:10])


#  shape_meta.obs.state_ee ={'shape': [15], 'type': 'low_dim', 
#      'components': {'ee_position': [0, 3], 'ee_quaternion': [3, 7], 'joint_positions': [7, 14], 'gripper_width': [14, 15]}}
#  shape_meta.action ={'shape': [8], 'components': {'ee_position': [0, 3], 'ee_quaternion': [3, 7], 'gripper_width': [14, 15]}}
#  number of parameters: 3.855975e+07
#  normalizer keys = ['action', 'state_ee', 'ee_cam_color', 'third_person_cam_color', 'side_cam_color']