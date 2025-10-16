"""Mock implementation of panda_py for simulation-only mode"""
import numpy as np
import time


class MockRobotState:
    """Mock robot state"""
    def __init__(self):
        self.q = np.zeros(7)  # joint positions
        self.dq = np.zeros(7)  # joint velocities
        self.tau_J = np.zeros(7)  # joint torques
        self.O_T_EE = np.eye(4).flatten()  # end-effector pose
        self.max_width = 0.08  # gripper max width
        self.width = 0.08  # current gripper width
        self.is_grasped = False


class MockController:
    """Mock base controller"""
    def __init__(self):
        self._control_value = None

    def set_control(self, value):
        self._control_value = value

    def set_damping(self, damping):
        pass

    def set_stiffness(self, stiffness):
        pass

    def set_filter(self, coefficient):
        pass


class MockJointPosition(MockController):
    """Mock joint position controller"""
    pass


class MockIntegratedVelocity(MockController):
    """Mock integrated velocity controller"""
    pass


class MockPureTorque(MockController):
    """Mock pure torque controller"""
    pass


class MockControllers:
    """Mock controllers module"""
    JointPosition = MockJointPosition
    IntegratedVelocity = MockIntegratedVelocity
    PureTorque = MockPureTorque


class MockPanda:
    """Mock Panda robot"""
    def __init__(self, ip):
        self._ip = ip
        self._state = MockRobotState()
        self._controller = None
        self._teaching_mode_enabled = False

    def get_state(self):
        """Get current robot state"""
        return self._state

    def start_controller(self, controller):
        """Start a controller"""
        self._controller = controller

    def stop_controller(self):
        """Stop the current controller"""
        self._controller = None

    def get_pose(self):
        """Get end-effector pose"""
        return self._state.O_T_EE

    def teaching_mode(self, enable):
        """Enable/disable teaching mode"""
        self._teaching_mode_enabled = enable

    def raise_error(self):
        """Check for errors - mock always returns no error"""
        pass

    def recover(self):
        """Recover from error state"""
        pass

    def move_to_start(self):
        """Move to start position"""
        self._state.q = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])

    def get_robot(self):
        """Get robot object for low-level control"""
        return MockRobot()


class MockRobot:
    """Mock low-level robot object"""
    def set_collision_behavior(self, torque_min, torque_max, force_min, force_max):
        """Set collision behavior thresholds"""
        pass


class MockGripper:
    """Mock Franka gripper"""
    def __init__(self, ip):
        self._ip = ip
        self._state = MockRobotState()

    def homing(self):
        """Home the gripper"""
        self._state.width = self._state.max_width
        return True

    def read_once(self):
        """Read gripper state once"""
        return self._state

    def move(self, width, speed):
        """Move gripper to specified width"""
        self._state.width = width
        time.sleep(0.01)  # Simulate movement time

    def grasp(self, width, speed, force, epsilon_inner, epsilon_outer):
        """Grasp with specified parameters"""
        self._state.width = width
        self._state.is_grasped = True
        time.sleep(0.01)  # Simulate grasp time

    def stop(self):
        """Stop gripper"""
        pass


class MockLibfranka:
    """Mock libfranka module"""
    Gripper = MockGripper


# Module-level exports
controllers = MockControllers()
Panda = MockPanda
libfranka = MockLibfranka()
