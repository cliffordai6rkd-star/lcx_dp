"""
    Contributions to the unitree robotics xr_teleoperation projects
    with modification by ZYX
"""

import os
import json
import cv2
import time
# try:
import rerun as rr
import rerun.blueprint as rrb
from dataset.lerobot.reader import RerunEpisodeReader, ActionType
from datetime import datetime
os.environ["RUST_LOG"] = "error"

class RerunLogger:
    def __init__(self, task_dir, prefix = "", IdxRangeBoundary = 30, memory_limit = None, example_item_data = None):
        self._task_dir = task_dir
        self.prefix = prefix
        self.IdxRangeBoundary = IdxRangeBoundary
        
        rr.init(datetime.now().strftime("Runtime_%Y%m%d_%H%M%S"))
        if memory_limit:
            rr.spawn(memory_limit = memory_limit, hide_welcome_screen = True)
        else:
            rr.spawn(hide_welcome_screen = True)

        # Set up blueprint for live visualization
        if self.IdxRangeBoundary and example_item_data:
            self.setup_blueprint(example_item_data)

    def setup_blueprint(self, example_item_data):
        joint_states_views = []
        action_views = []
        image_views = []

        # Dynamically create joint states views based on example data
        joint_states = example_item_data.get('joint_states', {})
        for joint_key in joint_states.keys():
            joint_view = rrb.TimeSeriesView(
                origin = f"{self.prefix}{joint_key}/joint_states",
                time_ranges=[
                    rrb.VisibleTimeRange(
                        "idx",
                        start = rrb.TimeRangeBoundary.cursor_relative(seq = -self.IdxRangeBoundary),
                        end = rrb.TimeRangeBoundary.cursor_relative(),
                    )
                ],
                plot_legend = rrb.PlotLegend(visible = True),
            )
            joint_states_views.append(joint_view)
            logger_mp.info(f'Created joint states view for: {joint_key}')

        # Dynamically create action views based on example data
        actions = example_item_data.get('actions', {})
        for action_key in actions.keys():
            if actions[action_key] is not None:
                action_view = rrb.TimeSeriesView(
                    origin = f"{self.prefix}{action_key}/actions",
                    time_ranges=[
                        rrb.VisibleTimeRange(
                            "idx",
                            start = rrb.TimeRangeBoundary.cursor_relative(seq = -self.IdxRangeBoundary),
                            end = rrb.TimeRangeBoundary.cursor_relative(),
                        )
                    ],
                    plot_legend = rrb.PlotLegend(visible = True),
                )
                action_views.append(action_view)
                logger_mp.info(f'Created action view for: {action_key}')

        # Dynamically create image views based on example data
        colors = example_item_data.get('colors', {})
        for color_key in colors.keys():
            if colors[color_key] is not None:
                image_view = rrb.Spatial2DView(
                    origin = f"{self.prefix}colors/{color_key}",
                    time_ranges=[
                        rrb.VisibleTimeRange(
                            "idx",
                            start = rrb.TimeRangeBoundary.cursor_relative(seq = -self.IdxRangeBoundary),
                            end = rrb.TimeRangeBoundary.cursor_relative(),
                        )
                    ],
                )
                image_views.append(image_view)
                logger_mp.info(f'Created image view for: {color_key}')

        # Organize views: joint states, actions, and images in three columns
        joint_states_column = rrb.Vertical(contents=joint_states_views)
        actions_column = rrb.Vertical(contents=action_views)
        images_column = rrb.Vertical(contents=image_views)
        
        grid = rrb.Horizontal(contents=[joint_states_column, actions_column, images_column],
                             column_shares=[1, 1, 1]
        )
        
        # Add control panels
        blueprint_content = [
            grid,
            rr.blueprint.SelectionPanel(state=rrb.PanelState.Collapsed),
            rr.blueprint.TimePanel(state=rrb.PanelState.Collapsed)
        ]
        rr.send_blueprint(rrb.Vertical(contents=blueprint_content))

    def log_item_data(self, item_data: dict):
        rr.set_time("idx", sequence=item_data.get('idx', 0))

        # Log states
        states = item_data.get('joint_states', {}) or {}
        for key, joint_state in states.items():
            positions = joint_state["position"]
            # logger_mp.info(f'{key} joint_state: {positions}')
            for idx, val in enumerate(positions):
                rr.log(f"{self.prefix}{key}/joint_states/qpos/{idx}", rr.Scalars([val]))

        # Log actions
        actions = item_data.get('actions', {}) or {}
        logger_mp.info(f'Actions data: {list(actions.keys())}')
        for action_key, action_val in actions.items():
            if action_val is not None:
                logger_mp.info(f'Logging action {action_key} with shape: {action_val.shape if hasattr(action_val, "shape") else type(action_val)}')
                for idx, val in enumerate(action_val):
                    rr.log(f"{self.prefix}{action_key}/actions/qpos/{idx}", rr.Scalars([val]))
            else:
                logger_mp.warning(f'Could not find {action_key} for action_key')

        # Log colors (images) - dynamically support all available images
        colors = item_data.get('colors', {}) or {}
        logger_mp.info(f'Colors data: {list(colors.keys())}')
        for color_key, color_val in colors.items():
            if color_val is not None:
                logger_mp.info(f'Logging image {color_key} with shape: {color_val.shape if hasattr(color_val, "shape") else type(color_val)}')
                rr.log(f"{self.prefix}colors/{color_key}", rr.Image(color_val))
            else:
                logger_mp.warning(f'Could not find {color_key} for color_key')

        # # Log depths (images)
        # depths = item_data.get('depths', {}) or {}
        # for depth_key, depth_val in depths.items():
        #     if depth_val is not None:
        #         # rr.log(f"{self.prefix}depths/{depth_key}", rr.Image(depth_val))
        #         pass # Handle depth if needed

        # # Log tactile if needed
        # tactiles = item_data.get('tactiles', {}) or {}
        # for hand, tactile_vals in tactiles.items():
        #     if tactile_vals is not None:
        #         pass # Handle tactile if needed

    def log_episode_data(self, episode_data: list):
        for item_data in episode_data:
            self.log_item_data(item_data)

if __name__ == "__main__":
    import logging_mp
    logger_mp = logging_mp.get_logger(__name__, level=logging_mp.INFO)

    # # TEST DATA OF data_dir
    # /home/yuxuan/Code/hirol/HIROLRobotPlatform/dataset/data/liquid_transfer
    # /home/yuxuan/Code/hirol/HIROLRobotPlatform/dataset/data/peg_in_hole
    # /home/yuxuan/Code/hirol/HIROLRobotPlatform/dataset/data/block_stacking
    # /home/yuxuan/Code/hirol/HIROLRobotPlatform/dataset/data/solid_transfer
    data_dir = "/home/yuxuan/Code/hirol/HIROLRobotPlatform/dataset/data/peg_in_hole"
    episode_data_number = 122
    fps = 100
    skip_step_nums = 3
    episode_dir = f"episode_{str(episode_data_number).zfill(4)}"
    if os.path.exists(os.path.join(data_dir, episode_dir)):
        logger_mp.info(f'Found the {episode_dir} in {data_dir}')
        episode_reader2 = RerunEpisodeReader(task_dir = data_dir)
        user_input = input("Please enter the start signal (enter 'on' to start the subsequent program):\n")
        episode_data = episode_reader2.return_episode_data(episode_data_number, skip_step_nums)
        logger_mp.info(f'Successfully load the episode data')
        if user_input.lower() == 'on':
            # logger_mp.info("Starting offline visualization with fixed idx size...")
            # Use first episode data as example for blueprint setup
            example_data = episode_data[0] if episode_data else None
            online_logger = RerunLogger(task_dir=data_dir, prefix="offline/", IdxRangeBoundary = 60, memory_limit="2GB", example_item_data=example_data)
            for item_data in episode_data:
                logger_mp.info(f'item data: {item_data}')
                online_logger.log_item_data(item_data)
                time.sleep(1/fps) # 30hz
            logger_mp.info("Offline visualization completed.")
    else:
        logger_mp.warning(f'Could not find {data_dir}/{episode_dir}')
        