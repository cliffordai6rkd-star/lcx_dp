---
name: hirol-robot-learning
description: Architecture-aware debugging and extension guidance for the HIROLRobotPlatform robot-learning project. Use when working in this repository to diagnose bugs, strange runtime behavior, config mismatches, teleoperation/data-collection failures, controller or simulation issues, dataset/replay/inference problems, or when adding support for new robots, tools, sensors, tasks, or learning pipelines while preserving the existing OOP and multi-robot design.
---

# Hirol Robot Learning

## Overview

Use the repository architecture to localize failures quickly across hardware, simulation, motion, controller, teleoperation, dataset, replay, and inference layers. Prefer fixing the shared abstraction, factory wiring, or config source of truth instead of adding task-specific patches.

Read `references/architecture-debug-map.md` when file-level navigation, subsystem ownership, or symptom-to-path mapping is needed.

## Follow This Workflow

1. Classify the failure before editing.
   - `startup/config`: import errors, missing keys, `!include` issues, wrong config branch.
   - `device/hardware`: connection, time sync, camera, FT, gripper, robot driver, async control.
   - `motion/controller`: IK failure, target mismatch, unstable motion, reset failure, safety rollback.
   - `task/teleop`: interface parsing, relative vs absolute target bugs, task selection, recording flow.
   - `dataset/replay/inference`: observation mismatch, action shape mismatch, checkpoint/task config mismatch.
2. Trace from the real entrypoint, not from a guessed module.
   - Teleoperation starts at `teleop/teleoperation.py`.
   - Robot teleop orchestration lives in `factory/tasks/robot_teleoperation.py`.
   - Replay starts at `factory/tasks/data_replay.py`.
   - Unified task execution API lives in `factory/components/gym_interface.py`.
   - Robot assembly lives in `factory/components/robot_factory.py`.
   - Motion assembly lives in `factory/components/motion_factory.py`.
3. Validate the config chain early.
   - Treat `hardware.base.utils.dynamic_load_yaml()` and `!include` expansion as a common failure source.
   - Check whether `motion_config` consistently wires robot, controller, model, sensors, tools, and simulation.
   - Check `use_hardware`, `use_simulation`, `teleop_interface`, `controller_type`, `model_type`, `target_site_name`, reset commands, and action representation together.
4. Reproduce at the lowest layer that still shows the bug.
   - Prefer model/controller/unit tests or a factory-level test before full teleop or inference runs.
   - If hardware is involved, isolate whether the same issue appears in simulation-only mode.
5. Patch the root cause in the shared layer.
   - Fix base classes, factory registration, config composition, or representation conversion before touching task-specific code.
6. Validate the exact path that failed.
   - Re-run the smallest relevant test or entrypoint.
   - If no test exists, add one when the bug is deterministic and repo-local.

## Respect These Project Invariants

- Treat `hardware/base`, `teleop/base`, `motion/model_base.py`, `controller/controller_base.py`, and `simulation/base/sim_base.py` as interface boundaries.
- Add new robot, tool, head, sensor, model, controller, or teleop device through subclassing plus factory registration. Avoid hard-coding special cases into tasks when the behavior belongs in a shared abstraction.
- Keep multi-robot support explicit. Reuse the existing `single` / `left` / `right` conventions and preserve DOF slicing, EE ordering, and tool ordering.
- Preserve common representations.
  - Joint states flow through `hardware.base.utils.RobotJointState`.
  - End-effector poses are generally `[x, y, z, qx, qy, qz, qw]`.
  - Inference/replay action semantics flow through `dataset.utils.ActionType` and related adapters.
- Prefer extending `factory/components/motion_configs/` and per-module config folders instead of duplicating large inline configs.
- Preserve the intended OOP direction. If a proposed fix makes one robot/task work by bypassing factories or base classes, treat it as suspect.

## Focus Areas for Common Bugs

### Config and Wiring

- Check missing nested keys after `!include`.
- Check mismatched config key names versus factory lookup dictionaries.
- Check inconsistent hardware/simulation flags across teleop, replay, and inference configs.

### Motion and Control

- Check EE link count versus target dimension.
- Check controller type against model capabilities and reset space.
- Check whether the bug appears before hardware execution, inside controller computation, or only after command dispatch.

### Teleoperation

- Check `inteface_output_mode` handling: `relative`, `absolute`, or `absolute_delta`.
- Check reset-edge logic and neutral pose initialization when XR or trackers are involved.
- Check simulation target mode versus real teleop device mode.

### Replay and Learning Inference

- Check task-specific config override behavior under `factory/tasks/inferences_tasks/*/config`.
- Check camera names, observation format, action type, orientation type, and state dimensions together.
- Check whether the failure is really a dataset/schema mismatch rather than a policy bug.

## Extend the Platform Safely

- Start from the nearest existing robot/task/config that matches the new feature.
- Reuse existing base classes and factory dictionaries before creating new orchestration code.
- Put robot-specific logic in `hardware/`, motion-model logic in `motion/`, control logic in `controller/`, and task/inference logic in `factory/tasks/`.
- Keep simulation and hardware paths comparable whenever possible so debug can fall back to simulation-first validation.

## Maintain This Skill

Update this skill whenever work on the repository reveals:

- a new stable subsystem boundary,
- a recurring failure mode,
- a new required config convention,
- a better debug entrypoint or test path,
- or a new extension pattern for robots, tools, sensors, controllers, or tasks.

Keep `SKILL.md` short. Move detailed repository knowledge into `references/architecture-debug-map.md` and refresh `agents/openai.yaml` if the skill scope changes.
