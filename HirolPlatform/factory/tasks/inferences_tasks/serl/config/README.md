# SERL Configuration Files

This directory contains configuration files for using HIROLRobotPlatform with HIL-SERL.

## File Structure

- `serl_fr3_config.yaml` - Main FR3 configuration for SERL
- `serl_impedance_params.yaml` - Impedance parameters for different compliance modes
- Other robot configs can be added here

## Configuration Inheritance

The SERL configurations use YAML inheritance to keep files clean and maintainable:

```yaml
# Base configuration
motion_config: !include factory/components/motion_configs/fr3_with_hand_impedance.yaml

# Override specific parameters
motion_config:
  use_smoother: false  # SERL handles smoothing
  use_trajectory_planner: false  # Direct control
```

## Key Design Principles

1. **Inherit Base Configs**: Use existing HIROLRobotPlatform configs as base
2. **Minimal Overrides**: Only override what's necessary for SERL
3. **Compliance Modes**: Support multiple stiffness/damping settings
4. **Safety First**: Include workspace limits and torque saturation

## Creating Config for New Robot

To create a SERL config for a new robot (e.g., Unitree G1):

1. Create `serl_unitree_g1_config.yaml`:
```yaml
# Base configuration for Unitree G1
motion_config: !include factory/components/motion_configs/unitree_g1_impedance.yaml

# SERL overrides
motion_config:
  use_smoother: false
  use_trajectory_planner: false
  
  # Robot-specific controller parameters
  controller_config:
    # ... specific parameters
    
# SERL parameters
serl_params:
  # ... same structure as FR3
```

2. Adjust compliance parameters based on robot characteristics
3. Update workspace limits for robot's reach
4. Test with conservative parameters first

## Compliance Mode Selection

Different tasks require different compliance:

- **Normal**: General manipulation tasks
- **Precision**: Fine positioning, insertion tasks  
- **Reset**: Moving to home position
- **Soft**: Delicate object handling
- **Stiff**: High-precision assembly
- **Teaching**: Human demonstration/guidance

## Usage in SerlRobotInterface

```python
# Load configuration
config_path = "path/to/serl_fr3_config.yaml"
robot = SerlRobotInterface(config_path)

# Update compliance dynamically
robot.update_params(ComplianceParams(
    translational_stiffness=2000,
    rotational_stiffness=150,
    # ...
))
```

## Tips

1. Start with lower stiffness values and increase gradually
2. Test force limits before deployment
3. Use precision mode only when needed (higher wear)
4. Monitor force/torque readings during operation
5. Adjust gripper_sleep based on gripper hardware