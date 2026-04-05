# Smoother Module Usage Guide

## Overview

The smoother module is now integrated directly into **RobotFactory**, providing second-order critical damped smoothing for joint commands. This ensures smooth, jerk-free robot movements without oscillations.

## Quick Start

### 1. Basic Configuration in RobotFactory

```yaml
# config/your_robot_config.yaml

# Enable smoother in RobotFactory
use_smoother: true

# Smoother configuration
smoother_config:
  type: "critical_damped"      # or "adaptive_critical_damped"
  omega_n: 25.0                 # Natural frequency (rad/s)
  control_frequency: 800        # Internal smoother frequency (Hz)
  zeta: 1.0                    # Damping ratio (1.0 = critical damping)

# Robot and other configurations
use_hardware: true
use_simulation: false
robot: "fr3"
# ... other robot configs
```

### 2. Adaptive Smoother Configuration

For tasks requiring variable stiffness based on tracking error:

```yaml
use_smoother: true

smoother_config:
  type: "adaptive_critical_damped"
  
  # Base parameters
  omega_n: 25.0
  control_frequency: 800
  zeta: 1.0
  
  # Adaptive parameters
  omega_n_min: 15.0      # Soft mode for small errors
  omega_n_max: 40.0      # Fast mode for large errors
  error_thresholds:
    low: 0.01            # Below this = soft mode (rad)
    high: 0.1            # Above this = fast mode (rad)
  transition: "sigmoid"  # Smooth transition between modes
  omega_change_limit: 5.0  # Max omega change rate (rad/s)
```

## Integration with RobotFactory

### How It Works

1. **Automatic Activation**: Smoother automatically activates when:
   - `use_smoother: true` in config
   - Control mode is `"position"` or `"velocity"`
   - Smoother successfully initializes

2. **Transparent Operation**: When enabled, smoother intercepts joint commands in `set_joint_commands()` and provides smoothed output

3. **Mode-Specific**: Smoother ONLY activates for position/velocity control, NOT for torque control

### Usage Example

```python
from factory.components.robot_factory import RobotFactory

# Load config with smoother enabled
config = load_yaml_config("config/robot_with_smoother.yaml")

# Create robot system
robot_factory = RobotFactory(config)
robot_factory.create_robot_system()

# Use normally - smoother works transparently
target_joints = np.array([0.1, 0.2, 0.3, -0.5, 0.0, 0.2, 0.0])

# Position control - smoother automatically smooths the command
robot_factory.set_joint_commands(target_joints, ['position'], execute_hardware=True)

# Torque control - smoother does NOT activate
torques = np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
robot_factory.set_joint_commands(torques, ['torque'], execute_hardware=True)
```

## Advanced Features

### 1. Move to Start Position

RobotFactory's `move_to_start()` now supports smooth movement:

```python
# Smooth movement to specified position (uses smoother)
target_position = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0])
robot_factory.move_to_start(target_position)

# Immediate reset to default position (bypasses smoother)
robot_factory.move_to_start(None)
```

### 2. Pause/Resume During Special Operations

```python
# Pause smoother for manual control or special operations
robot_factory.pause_smoother()

# ... perform special operations ...

# Resume smoother with position sync
robot_factory.resume_smoother(sync_to_current=True)
```

### 3. Dynamic Parameter Adjustment

```python
# Adjust response speed during operation
robot_factory.set_smoother_omega_n(35.0)  # Faster response
robot_factory.set_smoother_omega_n(15.0)  # Softer response

# Get smoother state for debugging
state = robot_factory.get_smoother_state()
print(f"Position: {state['position']}")
print(f"Velocity: {state['velocity']}")
```

## Configuration Examples

### 1. Teleoperation with Smoother

```yaml
# teleop/config/franka_3d_mouse_smoother.yaml

use_hardware: true
use_simulation: false
robot: "fr3"

# Enable smoother for smooth teleoperation
use_smoother: true
smoother_config:
  type: "adaptive_critical_damped"
  omega_n: 25.0
  omega_n_min: 18.0      # Soft for precision
  omega_n_max: 35.0      # Fast for large movements
  error_thresholds:
    low: 0.008           # ~0.46° 
    high: 0.08           # ~4.6°
  control_frequency: 800

# Disable trajectory planner when using smoother
use_trajectory_planner: false

# Other teleoperation settings
teleop_interface: "space_mouse"
control_frequency: 100
```

### 2. RL Training with Smoother

```yaml
# rl_training_config.yaml

use_smoother: true
smoother_config:
  type: "critical_damped"
  omega_n: 30.0          # Faster for RL responsiveness
  control_frequency: 800

# RL typically uses position control
controller_type: "ik"
control_frequency: 50    # RL policy frequency
```

### 3. Precision Assembly Tasks

```yaml
# precision_assembly_config.yaml

use_smoother: true
smoother_config:
  type: "adaptive_critical_damped"
  omega_n: 20.0
  omega_n_min: 12.0      # Very soft for contact
  omega_n_max: 25.0      # Moderate for approach
  error_thresholds:
    low: 0.005           # ~0.3°
    high: 0.05           # ~2.9°
  transition: "sigmoid"
```

## Parameter Tuning Guide

### omega_n (Natural Frequency)

| Value (rad/s) | Settling Time | Characteristics | Use Case |
|--------------|---------------|-----------------|----------|
| 10-15 | 0.31-0.46s | Very soft, slow | Delicate contact tasks |
| 15-20 | 0.23-0.31s | Soft, compliant | Human interaction |
| 20-30 | 0.15-0.23s | Balanced | General manipulation |
| 30-40 | 0.12-0.15s | Fast, responsive | Quick positioning |
| 40-50 | 0.09-0.12s | Very fast, stiff | Rapid movements |

**Formula**: Settling time (95%) ≈ 4.6 / omega_n

### Choosing Smoother Type

- **critical_damped**: Use for consistent, predictable behavior
- **adaptive_critical_damped**: Use when task requires variable compliance

## Performance Characteristics

- **Latency**: <1ms command processing
- **CPU Usage**: ~1-2% for 800Hz internal loop
- **Memory**: <10MB for 7-DOF robot
- **Zero Overshoot**: Critical damping ensures no oscillation
- **Thread-Safe**: Independent smoother thread with proper synchronization

## Common Issues & Solutions

### Issue: Motion too slow
```yaml
# Increase omega_n
smoother_config:
  omega_n: 35.0  # Faster response
```

### Issue: Motion too aggressive for contact
```yaml
# Decrease omega_n
smoother_config:
  omega_n: 15.0  # Softer response
```

### Issue: Smoother not activating
- Check `use_smoother: true` in config
- Verify control mode is "position" or "velocity"
- Check logs for initialization errors

### Issue: Jitter during teleoperation
```yaml
# Use adaptive mode with appropriate thresholds
smoother_config:
  type: "adaptive_critical_damped"
  error_thresholds:
    low: 0.01   # Adjust based on sensor noise
    high: 0.05  # Adjust based on typical movements
```

## API Reference

### RobotFactory Methods

```python
# Core smoother control
robot_factory.pause_smoother()                    # Pause smoothing
robot_factory.resume_smoother(sync_to_current)    # Resume with optional sync
robot_factory.set_smoother_omega_n(omega_n)       # Adjust response speed
robot_factory.get_smoother_state()                # Get current state

# Automatic smoother integration
robot_factory.set_joint_commands(cmd, mode, exec_hw)  # Auto-smoothed if applicable
robot_factory.move_to_start(joint_commands)           # Smooth if target provided
```

### Configuration Structure

```yaml
use_smoother: bool                    # Enable/disable smoother
smoother_config:
  type: str                          # "critical_damped" or "adaptive_critical_damped"
  omega_n: float                     # Natural frequency (10-50 rad/s)
  control_frequency: float           # Internal loop rate (500-1000 Hz)
  zeta: float                       # Damping ratio (default 1.0)
  
  # Adaptive mode only
  omega_n_min: float                # Minimum omega_n
  omega_n_max: float                # Maximum omega_n
  error_thresholds:
    low: float                      # Low error threshold (rad)
    high: float                     # High error threshold (rad)
  transition: str                   # "linear" or "sigmoid"
  omega_change_limit: float        # Max omega change rate
```

## Testing

```bash
# Test smoother integration
cd HIROLRobotPlatform
python test/test_smoother_integration.py

# Test with hardware (be careful!)
python test/test_smoother_integration.py --hardware

# Test specific functionality
python test/test_smoother_integration.py --test 2  # Test move_to_start
```

## Migration from MotionFactory

If you were previously using trajectory planner in MotionFactory:

1. **Disable trajectory planner**: Set `use_trajectory_planner: false`
2. **Enable smoother**: Set `use_smoother: true`
3. **Configure smoother**: Add `smoother_config` section
4. **No code changes needed**: RobotFactory handles everything internally

## Best Practices

1. **Start Conservative**: Begin with omega_n=20-25 and adjust based on performance
2. **Test First**: Always test in simulation before hardware
3. **Monitor Performance**: Check CPU usage and control loop timing
4. **Use Adaptive for Variable Tasks**: When task requires both precision and speed
5. **Sync After Reset**: Always sync smoother after emergency stops or resets

## Comparison with Trajectory Planner

| Feature | Trajectory Planner | Smoother |
|---------|-------------------|----------|
| Planning | Pre-computed trajectory | Real-time smoothing |
| Latency | Higher (planning time) | Minimal (<1ms) |
| Flexibility | Fixed trajectory | Adapts to command changes |
| CPU Usage | Spikes during planning | Constant low usage |
| Use Case | Known waypoints | Continuous control |

## Support

For issues or questions:
- Check robot logs for smoother initialization messages
- Verify configuration with `test/test_smoother_integration.py`
- Review this documentation for parameter tuning