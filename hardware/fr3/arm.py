from panda_py import controllers, libfranka, Panda
from hardware.base.arm import ArmBase
import glog as log

class Arm(ArmBase):
    def __init__(self, ip:str, mode='velocity'):
        super().__init__()
        self.instance = Panda(ip)
        self.model = self.instance.get_model()
        if 'velocity' == mode:
            self.controller = controllers.IntegratedVelocity()
        elif 'impedance' == mode:
            self.controller = controllers.CartesianImpedance()
        elif 'position' == mode:
            self.controller = controllers.JointPosition()
        elif 'torque' == mode:
            self.controller = controllers.AppliedTorque()
        elif 'force' == mode:
            self.controller = controllers.AppliedForce()
        elif 'pid_force' == mode:
            self.controller = controllers.Force()
        else:
            raise NotImplementedError

    def get_model(self):
        return self.model
    
    # def get_spatial_mass_matrix(self):
    #     return self.model.mass(robot_state=self.instance.get_state())
    
    def get_ee_orientation(self):
        """
        Current end-effector orientation in robot base frame.
        """
        orientation = self.instance.get_orientation()
        log.info(f"End Effector Orientation: {orientation}")
        return orientation

    def get_ee_position(self):
        """
        Current end-effector position in robot base frame.
        """
        position = self.instance.get_position()
        log.info(f"End Effector Position: {position}")
        return position
    
    def get_ee_pose(self):
        """
        Current end-effector pose (position and orientation) in robot base frame.
        """
        pose = self.instance.get_pose()
        log.info(f"End Effector Pose: {pose}")
        return pose

    def get_controller_time(self):
        return self.controller.get_time()
    
    def set_impedance_control(self, position, orientation):
        """
        Set the Cartesian impedance control with position and orientation.
        """
        if not isinstance(self.controller, controllers.CartesianImpedance):
            raise TypeError("Controller is not in impedance mode.")
        self.controller.set_control(position, orientation)
    
    def set_impedance_damping_ratio(self, damping):
        """
        Set the damping ratio for the Cartesian impedance controller.
        """
        if not isinstance(self.controller, controllers.CartesianImpedance):
            raise TypeError("Controller is not in impedance mode.")
        self.controller.set_damping_ratio(damping)
    
    def set_impedance_filter(self, filter_coeff: float):
        """
        Set the filter coefficient for the Cartesian impedance controller.
        """
        if not isinstance(self.controller, controllers.CartesianImpedance):
            raise TypeError("Controller is not in impedance mode.")
        self.controller.set_filter(filter_coeff)
    
    def set_impedance(self, impedance):
        """
        Set the stiffness (impedance) for the Cartesian impedance controller.
        """
        if not isinstance(self.controller, controllers.CartesianImpedance):
            raise TypeError("Controller is not in impedance mode.")
        self.controller.set_impedance(impedance)

    def set_impedance_nullspace_stiffness(self, nullspace_stiffness: float):
        """
        Set the nullspace stiffness for the Cartesian impedance controller.
        """
        if not isinstance(self.controller, controllers.CartesianImpedance):
            raise TypeError("Controller is not in impedance mode.")
        self.controller.set_nullspace_stiffness(nullspace_stiffness)
    
    def get_velocity_qd(self):
        """
        Get the current joint velocities (qd) from the IntegratedVelocity controller.
        """
        if not isinstance(self.controller, controllers.IntegratedVelocity):
            raise TypeError("Controller is not in velocity mode.")
        return self.controller.get_qd()

    def set_velocity_control(self, velocity):
        """
        Set the joint velocity control for the IntegratedVelocity controller.
        """
        if not isinstance(self.controller, controllers.IntegratedVelocity):
            raise TypeError("Controller is not in velocity mode.")
        self.controller.set_control(velocity)

    def set_velocity_damping(self, damping):
        """
        Set the damping for the IntegratedVelocity controller.
        """
        if not isinstance(self.controller, controllers.IntegratedVelocity):
            raise TypeError("Controller is not in velocity mode.")
        self.controller.set_damping(damping)

    def set_velocity_stiffness(self, stiffness):
        """
        Set the stiffness for the IntegratedVelocity controller.
        """
        if not isinstance(self.controller, controllers.IntegratedVelocity):
            raise TypeError("Controller is not in velocity mode.")
        self.controller.set_stiffness(stiffness)

    def set_position_control(self, position, velocity):
        """
        Set the joint position control for the JointPosition controller.
        """
        if not isinstance(self.controller, controllers.JointPosition):
            raise TypeError("Controller is not in position mode.")
        self.controller.set_control(position, velocity)

    def set_position_damping(self, damping):
        """
        Set the damping for the JointPosition controller.
        """
        if not isinstance(self.controller, controllers.JointPosition):
            raise TypeError("Controller is not in position mode.")
        self.controller.set_damping(damping)

    def set_position_stiffness(self, stiffness):
        """
        Set the stiffness for the JointPosition controller.
        """
        if not isinstance(self.controller, controllers.JointPosition):
            raise TypeError("Controller is not in position mode.")
        self.controller.set_stiffness(stiffness)

    def set_position_filter(self, filter_coeff: float):
        """
        Set the filter coefficient for the JointPosition controller.
        """
        if not isinstance(self.controller, controllers.JointPosition):
            raise TypeError("Controller is not in position mode.")
        self.controller.set_filter(filter_coeff)
    

    def set_torque_control(self, torque):
        """
        Set the joint torque control for the AppliedTorque controller.
        """
        if not isinstance(self.controller, controllers.AppliedTorque):
            raise TypeError("Controller is not in torque mode.")
        self.controller.set_control(torque)

    def set_torque_damping(self, damping):
        """
        Set the damping for the AppliedTorque controller.
        """
        if not isinstance(self.controller, controllers.AppliedTorque):
            raise TypeError("Controller is not in torque mode.")
        self.controller.set_damping(damping)

    def set_torque_filter(self, filter_coeff: float):
        """
        Set the filter coefficient for the AppliedTorque controller.
        """
        if not isinstance(self.controller, controllers.AppliedTorque):
            raise TypeError("Controller is not in torque mode.")
        self.controller.set_filter(filter_coeff)

    
    def set_force_control(self, force):
        """
        Set the Cartesian force control for the AppliedForce controller.
        """
        if not isinstance(self.controller, controllers.AppliedForce):
            raise TypeError("Controller is not in force mode.")
        self.controller.set_control(force)

    def set_force_damping(self, damping):
        """
        Set the damping for the AppliedForce controller.
        """
        if not isinstance(self.controller, controllers.AppliedForce):
            raise TypeError("Controller is not in force mode.")
        self.controller.set_damping(damping)

    def set_force_filter(self, filter_coeff: float):
        """
        Set the filter coefficient for the AppliedForce controller.
        """
        if not isinstance(self.controller, controllers.AppliedForce):
            raise TypeError("Controller is not in force mode.")
        self.controller.set_filter(filter_coeff)

    def set_pid_force_control(self, force):
        """
        Set the Cartesian force control for the Force controller.
        """
        if not isinstance(self.controller, controllers.Force):
            raise TypeError("Controller is not in pid_force mode.")
        self.controller.set_control(force)

    def set_pid_force_filter(self, filter_coeff: float):
        """
        Set the filter coefficient for the Force controller.
        """
        if not isinstance(self.controller, controllers.Force):
            raise TypeError("Controller is not in pid_force mode.")
        self.controller.set_filter(filter_coeff)

    def set_pid_force_integral_gain(self, k_i: float):
        """
        Set the integral gain for the Force controller.
        """
        if not isinstance(self.controller, controllers.Force):
            raise TypeError("Controller is not in pid_force mode.")
        self.controller.set_integral_gain(k_i)

    def set_pid_force_proportional_gain(self, k_p: float):
        """
        Set the proportional gain for the Force controller.
        """
        if not isinstance(self.controller, controllers.Force):
            raise TypeError("Controller is not in pid_force mode.")
        self.controller.set_proportional_gain(k_p)

    def get_state(self):
        log.info(self.instance.get_state())
    
    def set_control_mode(self, mode:str):
        self.mode = mode
    
    def move_to_start(self):
        self.instance.move_to_start()

    def move_to_pose(self, pose):
        self.instance.move_to_pose(pose)

    def move_to_joint_position(self, q):
        self.instance.move_to_joint_position(q)

