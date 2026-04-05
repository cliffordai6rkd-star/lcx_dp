"""
Animation plotter for real-time visualization of joint states and actions
"""

import os
import time
import threading
import collections
import queue
import json
from typing import Union

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glog as log


class AnimationPlotter:
    """Multi-canvas real-time animation plotter for joint states and actions."""
    
    def __init__(
        self, 
        joint_state_names: list[str],
        action_names: list[str],
        max_points: int = 50,
        figsize: tuple[int, int] = (10, 6),
        cols: int = 3,
        is_debug = False,
        update_frequency = 300,  # Higher frequency for more responsive plotting,
        enable_display = True
    ) -> None:
        """Initialize multi-canvas animation plotter.
        
        Args:
            max_points: Maximum display points (sliding window buffer)
            figsize: Overall figure size
            cols: Number of subplot columns
        """
        assert len(joint_state_names) == len(action_names), f"len of joint name{len(joint_state_names)} != len action name {len(action_names)}"
        
        self._enable_display = enable_display
        self.joint_state_names = joint_state_names
        self.action_names = action_names
        self.signal_count = len(joint_state_names)
        self.max_points = max_points
        self.figsize = figsize
        self.is_debug = is_debug
        self._has_gui = False  # Will be set in _init_plots()
        self._update_frequency = update_frequency
        
        # Smart column calculation: don't use more columns than signals
        self.cols = min(cols, self.signal_count)
        self.rows = (self.signal_count + self.cols - 1) // self.cols
        
        # Simplified thread safety
        self._lock = threading.Lock()
        self._active = False  # Single state flag
        self._update_timer = None
        self._plot_queue: queue.Queue = queue.Queue(maxsize=100)
        
        # Data storage with fixed-size buffers
        self.joint_data: dict[str, collections.deque] = {}
        self.action_data: dict[str, collections.deque] = {}
        self.timestamps: collections.deque = collections.deque(maxlen=max_points)
        
        for i in range(self.signal_count):
            self.joint_data[joint_state_names[i]] = collections.deque(maxlen=max_points)
            self.action_data[action_names[i]] = collections.deque(maxlen=max_points)
        
        # Matplotlib objects - lazy initialization
        self.fig = None
        self.axes = None
        self.joint_lines = None
        self.action_lines = None
        self._animation_ready = False
        
        log.info(f"AnimationPlotter initialized with {self.signal_count} signals, "
                f"max_points={max_points}, layout={self.rows}x{cols}")
    
    def _lazy_init_plots(self) -> None:
        """Initialize plots only when first data arrives."""
        if not self._enable_display:
            # Skip actual plot initialization if display is disabled
            self._animation_ready = True
            log.info("Animation system running in no-display mode")
            return
            
        try:
            self._init_plots()
            plt.ion()
            if self.fig:
                self.fig.show()
            self._animation_ready = True
            log.info("Animation plots lazy-initialized with display")
        except Exception as e:
            log.error(f"Lazy initialization failed: {e}")
            # Fall back to no-display mode
            self._enable_display = False
            self._animation_ready = True
            log.warning("Falling back to no-display mode")
    
    def _setup_matplotlib_backend(self) -> bool:
        """Simplified backend setup."""
        current_backend = matplotlib.get_backend()
        
        # If not using Agg, assume it works
        if current_backend != 'Agg':
            log.info(f"{'=='}*15 Using existing backend: {current_backend} {'=='}*15")
            return True
            
        # Simple DISPLAY check
        if os.environ.get('DISPLAY'):
            # Try TkAgg first (most common)
            matplotlib.use('TkAgg')
            log.info("{'=='}*15 Using GUI backend: TkAgg {'=='}*15")
            return True
        
        # Headless mode
        matplotlib.use('Agg')
        log.warning("{'=='}*15 Using headless mode (Agg backend) {'=='}*15")
        return False

    def _init_plots(self) -> None:
        """Initialize matplotlib subplots with simplified backend selection."""
        # Setup backend
        self._has_gui = self._setup_matplotlib_backend()
        log.info(f'Current matplotlib backend has gui: {self._has_gui}')
        
        if self._has_gui:
            plt.ion()  # Enable interactive mode for GUI
        
        self.fig, self.axes = plt.subplots(
            self.rows, self.cols, 
            figsize=self.figsize,
            facecolor='white'
        )
        
        # Handle different subplot configurations
        if self.signal_count == 1 and self.rows == 1 and self.cols == 1:
            # Single subplot case
            self.axes = [self.axes]
        else:
            # Multiple subplots - always flatten to get a consistent list
            self.axes = self.axes.flatten()
        
        self.joint_lines = []
        self.action_lines = []
        
        for i in range(self.signal_count):
            ax = self.axes[i]
            ax.set_title(f"plot_{i}", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Time Steps", fontsize=10)
            ax.set_ylabel("Value", fontsize=10)
            
            # 确保线条样式正确：joint_state实线，action虚线
            joint_line, = ax.plot([], [], 'b-', linewidth=2, label=self.joint_state_names[i], alpha=0.9)
            action_line, = ax.plot([], [], 'r--', linewidth=2, label=self.action_names[i], alpha=0.9)

            self.joint_lines.append(joint_line)
            self.action_lines.append(action_line)
            
            ax.legend(loc='upper right', fontsize=9)
            
            # 设置初始轴限制，避免空白
            ax.set_xlim(0, 10)
            ax.set_ylim(-2, 2)
        
        # Hide unused subplots
        for i in range(self.signal_count, len(self.axes)):
            self.axes[i].set_visible(False)
        
        plt.tight_layout()
        self._cur_time_stamp = 0
        log.info("Matplotlib subplots initialized")
    
    def update_signal(
        self, 
        joint_states: Union[list, np.ndarray], 
        actions: Union[list, np.ndarray]
    ) -> None:
        """Thread-safe update of signal data via queue mechanism.
        
        This method can be safely called from any thread. The actual plotting
        is handled by the main thread updater.
        
        Args:
            joint_states: Joint state data, length must match signal_names
            actions: Model output action data, length must match signal_names
        """
        # Convert to numpy arrays for consistent handling
        if self.is_debug:
            log.info(f"joint_states: {joint_states}, actions: {actions}")
        
        if not isinstance(joint_states, np.ndarray):
            joint_states = np.array(joint_states)
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        
        assert len(joint_states) == self.signal_count, f"joint states expected {self.signal_count}, but get {len(joint_states)}"
        assert len(actions) == self.signal_count, f"actions expected {self.signal_count}, but get {len(actions)}"
        
        self._queue_plot_update(joint_states, actions)
    
    def _queue_plot_update(self, joint_states: np.ndarray, actions: np.ndarray) -> None:
        """Queue plot data for thread-safe processing.
        
        Args:
            joint_states: Joint state data to plot
            actions: Action data to plot
        """
        try:
            current_time = time.time()
            plot_data = {
                'timestamp': current_time,
                'joint_states': joint_states.copy(),
                'actions': actions.copy()
            }
            
            # Non-blocking put with LRU policy
            try:
                self._plot_queue.put_nowait(plot_data)
            except queue.Full:
                # Remove oldest item and add new one (LRU policy)
                try:
                    self._plot_queue.get_nowait()
                    self._plot_queue.put_nowait(plot_data)
                except queue.Empty:
                    pass  # Queue was emptied by another thread
                    
        except Exception as e:
            log.warning(f"Failed to queue plot data: {e}")
    
    def _process_plot_queue(self) -> None:
        """Simplified queue processing with lazy initialization."""
        if not self._active:
            return
            
        # Lazy initialization on first queue processing
        if not self._animation_ready and not self._plot_queue.empty():
            self._lazy_init_plots()
            
        if self.fig is None:
            return
        
        # Process all available queue items
        updates = 0
        while not self._plot_queue.empty():
            plot_data = self._plot_queue.get_nowait()
            self._update_data_buffers(plot_data)
            updates += 1
        
        if updates > 0:
            if self.is_debug:
                log.info(f"📊 Processing {updates} updates")
            self._update_plots_directly()
        else:
            if self.is_debug:
                log.info(f"📊 No updates to process")
    
    def _update_data_buffers(self, plot_data: dict) -> None:
        """Update internal data buffers with queued data.
        
        Args:
            plot_data: Dictionary containing timestamp, joint_states, actions
        """
        joint_states = plot_data['joint_states']
        actions = plot_data['actions']
        
        # Dimension validation
        if len(joint_states) != self.signal_count or len(actions) != self.signal_count:
            log.warning(f"Data dimension mismatch: joints={len(joint_states)}, actions={len(actions)}, expected={self.signal_count}")
            log.warning(f"Joint states: {joint_states}")
            log.warning(f"Actions: {actions}")
            log.warning(f"Signal names: {self.joint_state_names}")
            return
            
        with self._lock:
            # Add timestamp (using sequential index)
            len_data = len(self.timestamps)
            need_to_pop = False
            if len_data >= self.max_points:
                self.timestamps.popleft()
                need_to_pop = True
            self.timestamps.append(self._cur_time_stamp)
            self._cur_time_stamp += 1
            
            # Add data points for each signal
            for i in range(self.signal_count):
                if need_to_pop:
                    self.joint_data[self.joint_state_names[i]].popleft()
                    self.action_data[self.action_names[i]].popleft()
                self.joint_data[self.joint_state_names[i]].append(float(joint_states[i]))
                self.action_data[self.action_names[i]].append(float(actions[i]))
            
            # Debug: Log data update occasionally
            if self._cur_time_stamp % 10 == 0 and self.is_debug:
                log.info(f"📈 Data update {self._cur_time_stamp}: joints={[f'{float(joint_states[j]):.3f}' for j in range(min(2, len(joint_states)))]},"
                         f" actions={[f'{float(actions[j]):.3f}' for j in range(min(2, len(actions)))]}")
    
    def _update_plots_directly(self) -> None:
        """Update matplotlib plots directly (main thread only) with data length validation."""
        with self._lock:
            # Check if we have data to plot
            if len(self.timestamps) == 0:
                log.warn(f"🔴 No timestamp data to plot")
                return
            
            if self.is_debug:
                log.info(f"🔵 Updating plots: timestamps={len(self.timestamps)}, signal_count={self.signal_count}, joint_lines={len(self.joint_lines) if self.joint_lines else 0}")
            
            # Get current data
            x_data = list(self.timestamps)
            
            # Update each signal subplot
            for i in range(self.signal_count):
                joint_key = self.joint_state_names[i]
                action_key = self.action_names[i]
                if (i >= len(self.joint_lines) or 
                    len(self.joint_data[joint_key]) <= 0 or 
                    len(self.action_data[action_key]) <= 0):
                    log.warn(f"⚠️ Skipping subplot {i}: joint_lines_len={len(self.joint_lines)}, joint_data_len={len(self.joint_data[joint_key])}, action_data_len={len(self.action_data[action_key])}")
                    continue
                
                if self.is_debug:
                    log.info(f"✅ Updating subplot {i}: joint_data={len(self.joint_data[joint_key])}, action_data={len(self.action_data[action_key])}")
                    
                # Get y data for this signal
                joint_y = list(self.joint_data[joint_key])
                action_y = list(self.action_data[action_key])
                
                # Data length validation to prevent numpy broadcast errors
                x_len = len(x_data)
                joint_len = len(joint_y)
                action_len = len(action_y)
                
                # Ensure all data arrays have matching length
                if x_len != joint_len or x_len != action_len:
                    log.warning(f"📏 Data length mismatch for signal {i}: x={x_len}, joint={joint_len}, action={action_len}")
                    # Use minimum length to avoid broadcast errors
                    min_len = min(x_len, joint_len, action_len)
                    if min_len <= 0:
                        log.warning(f"📏 Skipping signal {i} due to zero-length data")
                        continue
                    x_data_safe = x_data[:min_len]
                    joint_y_safe = joint_y[:min_len]
                    action_y_safe = action_y[:min_len]
                    log.info(f"📏 Truncated data to length {min_len} for signal {i}")
                else:
                    x_data_safe = x_data
                    joint_y_safe = joint_y
                    action_y_safe = action_y
                
                try:
                    # Update line data with thread-safe approach
                    self.joint_lines[i].set_data(x_data_safe, joint_y_safe)
                    self.action_lines[i].set_data(x_data_safe, action_y_safe)
                except (RuntimeError, ValueError) as e:
                    if "main thread" in str(e):
                        # Skip this update if not in main thread - will retry next cycle
                        if self.is_debug:
                            log.debug(f"⏭️ Skipping signal {i} update (threading issue)")
                        continue
                    else:
                        log.error(f"❌ Failed to set data for signal {i}: {e}")
                        continue
                
                # Update axis limits for visibility
                if x_data:
                    x_min, x_max = min(x_data), max(x_data)
                    if x_max > x_min:
                        self.axes[i].set_xlim(x_min - 1, x_max + 1)
                
                if joint_y or action_y:
                    all_y = joint_y + action_y
                    if all_y:
                        y_min, y_max = min(all_y), max(all_y)
                        if y_max > y_min:
                            y_margin = max(0.2, (y_max - y_min) * 0.15)
                            self.axes[i].set_ylim(y_min - y_margin, y_max + y_margin)
                        else:
                            self.axes[i].set_ylim(y_min - 1, y_min + 1)
            
            # Non-blocking redraw with controlled event processing
            try:
                # Use draw_idle() for non-blocking updates - queues drawing for next event loop
                self.fig.canvas.draw_idle()
                # Minimal event processing to ensure plots are displayed
                self._trigger_plot_update()
            except (RuntimeError, ValueError) as e:
                if "main thread" in str(e):
                    if self.is_debug:
                        log.debug(f"⏭️ Skipping plot draw (threading issue)")
                else:
                    log.warning(f"🎨 Plot drawing failed: {e}")
                
    def _trigger_plot_update(self) -> None:
        """Trigger minimal matplotlib event processing for plot updates."""
        if self.fig and plt.get_fignums():
            # Minimal pause to ensure GUI updates - much faster than plt.pause()
            self.fig.canvas.flush_events()
                    
    def start_animation(self) -> None:
        """Start lazy-initialized plotting system."""
        if self._active:
            log.warning("Animation already running")
            return
        
        # Don't initialize plots immediately - do it lazily when first data arrives
        self._animation_ready = False
        log.info("Animation system ready for lazy initialization")
    
    def start_main_thread_updater(self) -> None:
        """Start the main thread plot updater timer.
        
        Must be called from the main thread.
        
        Raises:
            RuntimeError: If not called from main thread
        """
        import threading as thread_module
        assert thread_module.current_thread() == thread_module.main_thread(), "Must be called from main thread"
        
        if self._active:
            log.warning("Updater already running")
            return
            
        self._active = True
        
        def _timer_callback():
            if self._active:
                if self.is_debug:
                    log.info(f"⏰ Timer callback triggered")
                self._process_plot_queue()
            else:
                if self.is_debug:
                    log.warn(f"⏰ Timer callback but not active")
        
        # Always use threading timer for reliability
        import threading as thread_module
        def fallback_timer():
            while self._active:
                start = time.perf_counter()
                _timer_callback()
                time.sleep(1.0/self._update_frequency)
                used = time.perf_counter() - start
                if self.is_debug:
                    log.info(f'fall back timer thread time used {used * 1000}ms for one loop')
        timer_thread = thread_module.Thread(target=fallback_timer, daemon=True)
        timer_thread.start()
        log.info(f"🟡 Threading timer started: frequency={self._update_frequency}Hz")
    
    def stop_animation(self) -> None:
        """Stop animation and cleanup."""
        self._active = False
        
        # Stop timer
        if self._update_timer is not None:
            self._update_timer.stop()
            self._update_timer = None
            
        # Clear queue
        while not self._plot_queue.empty():
            self._plot_queue.get_nowait()
                
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
        log.info("Animation stopped")
    
    def clear_data(self) -> None:
        """Clear all stored data and queue."""
        with self._lock:
            self.timestamps.clear()
            for key in self.joint_data:
                self.joint_data[key].clear()
            for key in self.action_data:
                self.action_data[key].clear()
            self._cur_time_stamp = 0
            
        # Clear plot queue
        while not self._plot_queue.empty():
            try:
                self._plot_queue.get_nowait()
            except queue.Empty:
                break
                
        log.info("Animation data and queue cleared")
    
    def save_data(self, filepath: str) -> None:
        """Save current trajectory data to JSON file.
        
        Args:
            filepath: Output file path
        """
        with self._lock:
            data = {
                'joint_state_names': self.joint_state_names,
                'action_names': self.action_names,
                'timestamps': list(self.timestamps),
                'joint_data': {i: list(self.joint_data[self.joint_state_names[i]]) for i in range(self.signal_count)},
                'action_data': {i: list(self.action_data[self.action_names[i]]) for i in range(self.signal_count)},
                'max_points': self.max_points
            }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            log.info(f"Trajectory data saved to {filepath}")
        except Exception as e:
            log.error(f"Failed to save data to {filepath}: {e}")
            raise
        