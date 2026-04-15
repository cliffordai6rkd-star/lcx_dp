# HIROLRobotPlatform Architecture and Debug Map

Use this file when navigation or failure localization matters more than generic debugging advice.

## System Shape

The repository is a unified robot-learning platform with this typical flow:

1. Compose robot, tool, sensors, controller, model, and simulation from YAML.
2. Build a robot system with `RobotFactory`.
3. Build motion/control threads with `MotionFactory`.
4. Drive the system from teleoperation, replay, or learning inference tasks.
5. Record data into `dataset/` formats and reuse them for replay or model validation.

The codebase is organized as layered subsystems, not as isolated one-off demos. Prefer reading and fixing one layer below the symptom.

## Main Subsystems

### `hardware/`

- Own robot/tool/sensor implementations.
- Base contracts live in:
  - `hardware/base/arm.py`
  - `hardware/base/tool_base.py`
  - `hardware/base/camera.py`
  - `hardware/base/ft.py`
  - `hardware/base/tactile_base.py`
  - `hardware/base/utils.py`
- Representative robot implementations:
  - `hardware/fr3/`
  - `hardware/unitreeG1/`
  - `hardware/agibot_g1/`
  - `hardware/monte01/`

### `controller/`

- Own controller logic from EE targets to joint commands.
- Main abstractions and implementations:
  - `controller/controller_base.py`
  - `controller/impedance_controller.py`
  - `controller/cartesian_impedance_controller.py`
  - `controller/whole_body_ik.py`
  - `controller/g1_controller.py`
  - `controller/duo_controller.py`

### `motion/`

- Own robot models, kinematics, IK solvers, and trajectory-facing computation.
- Key files:
  - `motion/model_base.py`
  - `motion/pin_model.py`
  - `motion/duo_model.py`
  - `motion/kinematics.py`
  - `motion/ik.py`
  - `motion/adapters/`

### `factory/components/`

- Own composition and orchestration.
- Most important files:
  - `factory/components/robot_factory.py`
  - `factory/components/motion_factory.py`
  - `factory/components/gym_interface.py`
  - `factory/components/motion_configs/`

### `teleop/`

- Own teleoperation device integration and interactive data collection.
- Main entrypoint:
  - `teleop/teleoperation.py`
- Core orchestration:
  - `factory/tasks/robot_teleoperation.py`
- Device families:
  - `teleop/space_mouse/`
  - `teleop/XR/quest3/`
  - `teleop/pika_tracker/`
  - `teleop/aruco_cube_tracker/`

### `factory/tasks/`

- Own higher-level tasks that use the common robot/motion stack.
- Important paths:
  - `factory/tasks/data_replay.py`
  - `factory/tasks/inferences_tasks/act/`
  - `factory/tasks/inferences_tasks/pi0/`
  - `factory/tasks/inferences_tasks/bc_policy/`
  - `factory/tasks/inferences_tasks/serl/`

### `dataset/`

- Own data IO, conversion, and visualization helpers.
- Important paths:
  - `dataset/lerobot/`
  - `dataset/diffusion_policy/`
  - `dataset/utils.py`

### `simulation/`

- Own simulation backends and scene configuration.
- Important paths:
  - `simulation/base/sim_base.py`
  - `simulation/mujoco/mujoco_sim.py`
  - `simulation/config/`
  - `simulation/scene_config/`

## Entrypoints by Workflow

### Teleoperation and Data Collection

- Start at `teleop/teleoperation.py`.
- Follow:
  - task selection and config merge,
  - `RobotFactory`,
  - `MotionFactory`,
  - `TeleoperationFactory.create_robot_teleoperation_system()`.

Use this path when debugging:

- device input parsing,
- reset logic,
- live target visualization,
- episode recording,
- task selection errors.

### Replay

- Start at `factory/tasks/data_replay.py`.
- Follow:
  - config validation,
  - `GymApi`,
  - `RerunEpisodeReader`,
  - action conversion and execution loop.

Use this path when debugging:

- action schema mismatch,
- replay timing problems,
- saved episode incompatibility,
- replay-to-hardware regressions.

### Learning Inference

- Start from the concrete task under `factory/tasks/inferences_tasks/`.
- ACT examples live under `factory/tasks/inferences_tasks/act/`.
- PI0 examples live under `factory/tasks/inferences_tasks/pi0/`.
- Most task execution still routes through `factory/components/gym_interface.py`.

Use this path when debugging:

- checkpoint/task-type mismatch,
- observation or camera-name mismatch,
- policy output shape mismatch,
- task-specific postprocessing behavior.

## Config Topology

The project uses YAML composition heavily. `hardware.base.utils.dynamic_load_yaml()` registers `!include`, so a bug may be in the selected config file, a nested include, or the consuming code.

Typical composition chain:

1. A teleop, replay, or inference config selects `motion_config`.
2. The selected motion config pulls in robot/tool/sensor/controller/model/simulation configs.
3. Factories look up implementation classes by string keys such as `robot`, `gripper`, `simulation`, `controller_type`, and `model_type`.

Representative files:

- `teleop/config/franka_3d_mouse.yaml`
- `factory/components/motion_configs/fr3_with_franka_hand_ik.yaml`
- `factory/tasks/inferences_tasks/act/config/fr3_act_inference_cfg.yaml`
- `factory/tasks/inferences_tasks/pi0/config/fr3_pi0_cfg.yaml`

Common config failures:

- Missing top-level key after `!include`.
- Included file shape not matching the code expectation.
- String key not present in factory dictionaries.
- Reset command dimension not matching robot/model/controller assumptions.
- Hardware and simulation flags describing incompatible execution modes.

## Symptom-to-Path Map

### Import error or startup crash

- Check entrypoint imports first.
- Check optional device imports with fallbacks, especially XR-related code in `factory/tasks/robot_teleoperation.py`.
- Check submodule/dependency-backed directories under `dependencies/`.

### Robot/system creates but motion is wrong

- Check `factory/components/motion_factory.py`.
- Check model/controller config pairing.
- Check EE link names and robot model config under `motion/config/`.
- Check controller-specific config under `controller/config/`.

### Simulation works, hardware fails

- Check `use_hardware`, time sync, network, and device-specific config.
- Check hardware implementation under `hardware/`.
- Check whether async control or smoothing changes only affect hardware paths.

### Teleop target looks wrong or drifts

- Check `factory/tasks/robot_teleoperation.py`.
- Check `inteface_output_mode`.
- Check pose frame conversion helpers in `hardware/base/utils.py`.
- Check initial pose and reset logic for XR/tracker devices.

### Replay/inference action execution looks wrong

- Check `factory/components/gym_interface.py`.
- Check `action_type`, `action_orientation_type`, relative-pose settings, and tool DOF.
- Check task-specific overrides in `factory/tasks/inferences_tasks/*/config/tasks/`.

### Learning policy behaves strangely only on one task

- Check checkpoint naming and task detection logic.
- Check:
  - `factory/tasks/inferences_tasks/utils/task_types.py`
  - `factory/tasks/inferences_tasks/utils/config_loader.py`
  - `factory/tasks/inferences_tasks/utils/gripper_controller.py`
  - `factory/tasks/inferences_tasks/utils/gripper_strategies/`

## Useful Discovery Commands

Use fast repository search instead of manual browsing.

```sh
rg -n "dynamic_load_yaml|!include" .
rg -n "controller_type|model_type|teleop_interface|use_hardware|use_simulation" teleop factory controller motion hardware
rg -n "_robot_classes|_gripper_classes|_camera_classes|_controller_classes|_model_classes" factory controller motion hardware
rg -n "ActionType|ObservationType|checkpoint_path|max_step_nums|gripper_postprocess" factory/tasks dataset
rg --files test motion/test_scripts factory/ik_benchmark
```

## Existing Tests and Validation Targets

These files are useful anchors, but not all are hermetic or hardware-free.

- General/task-system:
  - `test/test_end_to_end.py`
  - `test/test_config_priority_system.py`
  - `test/test_task_type_system.py`
- Motion/controller:
  - `test/test_motion_factory.py`
  - `test/test_robot_motion.py`
  - `test/test_impedance_controller.py`
  - `test/test_cartesian_impedance_controller.py`
  - `motion/test_scripts/test_ik_comparison.py`
  - `motion/test_scripts/test_ik_pyroki_speed.py`
- Hardware-specific:
  - `test/test_fr3_hardware.py`
  - `test/test_unitree_g1.py`
  - `test/test_xr_duo_fr3.py`
- Benchmarking:
  - `factory/ik_benchmark/run_benchmark.py`

## Extension Rules

### Add a new robot or tool

- Add or reuse a base-class implementation under `hardware/`.
- Register it in `factory/components/robot_factory.py`.
- Add matching YAML under the appropriate config folder.
- Add or extend a motion config under `factory/components/motion_configs/`.

### Add a new controller or model

- Implement the interface in `controller/` or `motion/`.
- Register it in `factory/components/motion_factory.py`.
- Add controller/model config files.
- Validate target dimension, EE links, and reset behavior.

### Add a new task or inference path

- Keep shared execution in `factory/components/gym_interface.py` when possible.
- Put task-specific logic under `factory/tasks/inferences_tasks/`.
- Reuse task-type/config-loader patterns instead of forking the whole pipeline.

## Skill Maintenance Rule

When work on this repository reveals a better subsystem map, a recurring failure mode, or a new stable extension pattern, update this reference file and trim obsolete guidance. The skill should evolve with the project instead of staying generic.
