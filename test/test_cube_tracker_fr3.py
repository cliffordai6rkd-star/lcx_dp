from teleop.aruco_cube_tracker.cube_tracker import CubePoseTracker
from controller.controller_base import IKController
from simulation.mujoco.mujoco_sim import MujocoSim
from hardware.base.utils import negate_pose, transform_pose, convert_homo_2_7D_pose, transform_quat
from motion.pin_model import RobotModel
import glog as log
from cfg_handling import get_cfg
import cv2
import numpy as np
import argparse
import time


def get_sim_base_world_transform(sim):
    """Fetch world<->base transforms from mujoco sim (single or multi-base)."""
    world2base_pose = [np.array([0, 0, 0, 0, 0, 0, 1])]
    base2world_pose = [negate_pose(world2base_pose[0])]
    if len(sim.base_body_name) != 0:
        world2base_pose = []
        base2world_pose = []
        for cur_base_body in sim.base_body_name:
            cur_world2base = sim.get_body_pose(cur_base_body)
            world2base_pose.append(cur_world2base)
            base2world_pose.append(negate_pose(cur_world2base))
    return world2base_pose, base2world_pose


def main():
    parser = argparse.ArgumentParser(description="CubeTracker teleop test in Mujoco sim")
    parser.add_argument(
        "--cfg",
        type=str,
        default="teleop/aruco_cube_tracker/config/cube_tracker_right_only_fr3_cfg.yaml",
        help="Path to cube tracker YAML config",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="absolute_delta",
        choices=["absolute", "absolute_delta"],
        help="Teleop output mode",
    )
    parser.add_argument(
        "--control",
        action="store_true",
        help="Enable IK control; otherwise only place mocap for pose checking",
    )
    args = parser.parse_args()

    # Cube tracker interface
    tracker_cfg = get_cfg(args.cfg)["cube_tracker"]
    tracker = CubePoseTracker(tracker_cfg)

    # Robot model and IK controller (FR3)
    model_config = "motion/config/robot_model_fr3_pika_ati_cfg.yaml"
    model_cfg = get_cfg(model_config)["fr3_pika_ati"]
    model = RobotModel(model_cfg)

    controller_config = "controller/config/ik_fr3_cfg.yaml"
    controller_cfg = get_cfg(controller_config)["ik"]
    ik = IKController(controller_cfg, model)

    # Mujoco simulation
    mujoco_cfg = "simulation/config/mujoco_fr3_pika_ati_posi.yaml"
    mujoco = MujocoSim(get_cfg(mujoco_cfg)["mujoco"])
    world2base, base2world = get_sim_base_world_transform(mujoco)
    target_site_names = ["targetL", "target"]  # use right-hand site by default

    last_time = time.time()
    ee_frame_name = model.get_model_end_links()[0]
    log.info(f'ee frame name: {ee_frame_name}')
    total_dof = model.get_model_dof()

    interface_mode = args.mode
    enabled = False
    init_ee_pose = []
    init_ee_rot = []
    log.info(
        f"CubeTracker test started | mode={interface_mode}, control={args.control}"
    )

    while True:
        success, arm_target, tool_target = tracker.parse_data_2_robot_target(interface_mode)
        success = int(success) > 0
        if not success and enabled:
            enabled = False
            
        if success:
            # Select key among returned targets
            key = (
                "single"
                if "single" in arm_target
                else ("right" if "right" in arm_target else "left")
            )
            # log.info(f'value: {arm_target}, {key}')
            value = arm_target[key]

            # Absolute-delta: lock relative to current EE pose when user presses 'i'
            if interface_mode == "absolute_delta" and enabled:
                cur_init_pose = init_ee_pose
                tracker_robot_trans = np.hstack(([0,0,0], init_ee_rot))
                # 重要改动： ！！！！ @TODO: zyx （集成进teleop里面）
                robot_tracker_trans = negate_pose(tracker_robot_trans)
                value = transform_pose(transform_pose(robot_tracker_trans, value), tracker_robot_trans)
                value = transform_pose(cur_init_pose, value, True)
                # log.info(f"init pose: {cur_init_pose}, diff: {value}")

            # Visualize target pose via mocap site in sim
            target_mocap = target_site_names[1]  # right-hand site name
            target_mocap = target_mocap.split("_")[0]
            if interface_mode == "absolute_delta":
                world2target = transform_pose(world2base[0], value)
            else:
                world2target = value
            mujoco.set_target_mocap_pose(target_mocap, world2target)

            # --- Debug overlay on RGB frame from the tracker camera ---
            try:
                cam_pack = tracker._camera.capture_all_data()
                img = cam_pack.get("image", None)
                depth_raw = cam_pack.get("depth_map", None)
                depth_m = None
                if depth_raw is not None:
                    depth_m = depth_raw.astype(np.float32) * getattr(
                        tracker._camera, "g_depth_scale", 0.001
                    )

                # choose underlying cube tracker according to active side
                if key == "single":
                    side = "right" if getattr(tracker, "_output_right", False) else "left"
                else:
                    side = key

                if img is not None and side in tracker._cube_trackers:
                    pose_cam = tracker._cube_trackers[side].get_pose(img, depth=depth_m)
                    if pose_cam is not None:
                        overlay = tracker._cube_trackers[side].overlay_cube_pose(
                            img, {side: pose_cam}
                        )
                        cv2.imshow("CubeTracker Overlay", overlay)
                    else:
                        cv2.imshow("CubeTracker Overlay", img)
                elif img is not None:
                    cv2.imshow("CubeTracker Overlay", img)
            except Exception:
                # Keep test robust even if visualization fails in headless
                pass

            # Decide whether to compute IK and command joints
            do_control = args.control and (
                (enabled and interface_mode == "absolute_delta")
                or interface_mode == "absolute"
            )

            # Enable absolute_delta on rising edge of keypress ('i' in interface)
            if interface_mode == "absolute_delta":
                pressed = tool_target[key][-1]
                if pressed and not enabled:
                    joint_position = mujoco.get_joint_states()._positions
                    cur_ee_pose_homo = model.get_frame_pose(
                        ee_frame_name, joint_position, True, "single"
                    )
                    init_ee_pose = convert_homo_2_7D_pose(cur_ee_pose_homo)
                    log.info(f'updated init robot ee pose: {negate_pose(init_ee_pose)[3:]}')
                    init_ee_rot = init_ee_pose[3:]
                    enabled = True
                    do_control = False  # skip control for this cycle

            if do_control:
                joint_states = mujoco.get_joint_states()
                targets = [{ee_frame_name: value}]
                success_ik, joint_target, mode = ik.compute_controller(
                    targets, joint_states
                )
                if success_ik:
                    mujoco.set_joint_command([mode] * total_dof, joint_target)

        # 50 Hz loop
        # pump GUI events regardless of success
        _ = cv2.waitKey(1) & 0xFF

        dt = time.time() - last_time
        if dt < (1.0 / 50):
            time.sleep((1.0 / 50) - dt)
        last_time = time.time()


if __name__ == "__main__":
    main()
