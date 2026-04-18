import json
from pathlib import Path
from typing import Any, Dict, List

import glog as log
import numpy as np

from dataset.lerobot.reader import RerunEpisodeReader
from dataset.utils import Action_Type_Mapping_Dict, ObservationType
from factory.tasks.inferences_tasks.dp.dp_inference import DPPolicyRuntime


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _to_serializable(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: _to_serializable(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_to_serializable(item) for item in data]
    if isinstance(data, tuple):
        return [_to_serializable(item) for item in data]
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, np.generic):
        return data.item()
    return data


def _save_json(json_file: Path, data: Dict[str, Any]) -> None:
    json_file.parent.mkdir(parents=True, exist_ok=True)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(data), f, ensure_ascii=False, indent=4)


class DataDPInferencer:
    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._dp_runtime = DPPolicyRuntime(config)
        self._obs_type = ObservationType(
            config.get("observation_type", ObservationType.JOINT_POSITION_ONLY.value)
        )

        self._dataset_task_data_dir = _resolve_path(config["dataset_task_data_dir"])
        self._dataset_episode_id = config.get("dataset_episode_id", 0)
        self._dataset_skip_steps = config.get("dataset_skip_steps", 1)
        self._dataset_cam_keys = config.get("dataset_cam_keys")
        self._dataset_state_keys = config.get("dataset_state_keys")
        self._dataset_data_type = config.get("dataset_data_type", "real_robot")
        self._output_json_path = self._resolve_output_path(config)

        self._dataset_reader = RerunEpisodeReader(
            task_dir=str(self._dataset_task_data_dir),
            action_type=Action_Type_Mapping_Dict[config["action_type"]],
            action_prediction_step=1,
            action_ori_type=config.get("action_orientation_type", "euler"),
            observation_type=self._obs_type,
            state_keys=self._dataset_state_keys,
            camera_keys=self._dataset_cam_keys,
            data_type=self._dataset_data_type,
        )

    def _resolve_output_path(self, config: Dict[str, Any]) -> Path:
        output_path = config.get("output_json_path")
        if output_path:
            return _resolve_path(output_path)

        filename = f"data_inference_episode_{self._dataset_episode_id:04d}_actions.json"
        return self._dataset_task_data_dir / filename

    def _load_episode(self) -> List[Dict[str, Any]]:
        episode_data = self._dataset_reader.return_episode_observations(
            self._dataset_episode_id,
            skip_steps_nums=self._dataset_skip_steps,
        )
        if not episode_data:
            raise ValueError("dataset episode is empty")
        return episode_data

    def _frame_to_gym_obs(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "colors": frame_data.get("colors", {}),
            "state": frame_data.get("observations", {}),
        }

    def run(self) -> Path:
        episode_data = self._load_episode()
        self._dp_runtime.reset()

        predictions = []
        for frame_offset, frame_data in enumerate(episode_data):
            gym_obs = self._frame_to_gym_obs(frame_data)
            obs_dict = self._dp_runtime.convert_from_gym_obs(gym_obs)
            if obs_dict is None:
                continue

            action_chunk = self._dp_runtime.policy_prediction(obs_dict)
            predictions.append(
                {
                    "frame_offset": frame_offset,
                    "dataset_idx": frame_data.get("idx", frame_offset),
                    "action_chunk": action_chunk,
                    "first_action": action_chunk[0],
                }
            )

        output = {
            "checkpoint_path": self._config["checkpoint_path"],
            "dataset_task_data_dir": str(self._dataset_task_data_dir),
            "dataset_episode_id": self._dataset_episode_id,
            "dataset_skip_steps": self._dataset_skip_steps,
            "n_obs_steps": self._dp_runtime.n_obs_steps,
            "n_action_steps": self._dp_runtime.n_action_steps,
            "num_frames": len(episode_data),
            "num_predictions": len(predictions),
            "predictions": predictions,
        }
        _save_json(self._output_json_path, output)
        log.info(f"Saved dataset inference actions to {self._output_json_path}")
        return self._output_json_path

    def close(self) -> None:
        if hasattr(self, "_dp_runtime"):
            self._dp_runtime.close()


def main():
    from factory.utils import parse_args
    from hardware.base.utils import dynamic_load_yaml

    arguments = {
        "config": {
            "short_cut": "-c",
            "symbol": "--config",
            "type": str,
            "default": "factory/tasks/inferences_tasks/dp/config/fr3_dp_dataset_inference_cfg.yaml",
            "help": "Path to the config file",
        }
    }
    args = parse_args("dp data inference", arguments)

    config = dynamic_load_yaml(args.config)
    log.info(f"data inference config: {config}")
    inferencer = DataDPInferencer(config)
    try:
        inferencer.run()
    finally:
        inferencer.close()


if __name__ == "__main__":
    main()
