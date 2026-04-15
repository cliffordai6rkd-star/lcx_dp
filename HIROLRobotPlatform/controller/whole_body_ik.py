from __future__ import annotations

from motion.pin_model import RobotModel
from controller.controller_base import ControllerBase
import casadi
import pinocchio.casadi as cpin
import pinocchio as pin
from hardware.base.utils import RobotJointState, convert_7D_2_homo, convert_quat_to_rot_matrix
from controller.utils.weighted_moving_filter import WeightedMovingFilter
import glog as log
import numpy as np
import time

DEBUG = False

class WholeBodyIk(ControllerBase):
    def __init__(self, config, robot_model: RobotModel):
        super().__init__(config, robot_model)
        self.tracking_frames = config["tracking_frames"]
        self.target_threshold = config.get("target_threshold", 1e-3)
        self.rot_target_threshold = config.get("rot_target_threshold", 1e-3)
        self.joint_state_threshold = config.get("joint_state_threshold", 2e-4)
        self._nullspace_tracking = config.get('nullspace_tracking', None)
        self.show_debug = config.get("show_debug", False)
        self.translation_weight = config["trans_weight"]
        self.rot_weight = config["rot_weight"]
        self.nullspace_weight = config.get("nullspace_weight", 1.0)
        self.smooth_weight = config["smooth_weight"]
        self.regularization_weight = config["regularization_weight"]
        self.solver_max_iter = config.get("max_iter", 50)
        self.solver_tol = config.get("solver_tol", 1e-6)
        self.acceptable_tol = config.get("acceptable_tol", 1e-5)
        self.filter = WeightedMovingFilter([1.0/8] * 8, self._robot_model.nq)
        
        # init of casadi opti problem
        self._init_casadi_problem()
        
        # cache for reducing solution time
        self.previous_robot_position = None
        self.previous_solution = None
        self.previous_targets = None
        
    @staticmethod
    def _rotation_distance(R_a: np.ndarray, R_b: np.ndarray) -> float:
        cos_theta = 0.5 * (np.trace(R_a @ R_b.T) - 1.0)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return float(np.arccos(cos_theta))
    
    def _is_target_updated(self, target_values: list[np.ndarray]) -> bool:
        if self.previous_targets is None or len(self.previous_targets) != len(target_values):
            return True
        
        for i, cur_target in enumerate(target_values):
            prev_target = self.previous_targets[i]
            track_cfg = self.tracking_frames[i]
            if track_cfg["require_trans"] and track_cfg["require_rot"]:
                trans_diff = np.linalg.norm(cur_target[:3, 3] - prev_target[:3, 3])
                if trans_diff > self.target_threshold:
                    return True
                rot_diff = self._rotation_distance(cur_target[:3, :3], prev_target[:3, :3])
                if rot_diff > self.rot_target_threshold:
                    return True
            elif track_cfg["require_trans"]:
                if np.linalg.norm(cur_target - prev_target) > self.target_threshold:
                    return True
            elif track_cfg["require_rot"]:
                rot_diff = self._rotation_distance(cur_target, prev_target)
                if rot_diff > self.rot_target_threshold:
                    return True
        return False
    
    def _update_target_cache(self, target_values: list[np.ndarray]) -> None:
        self.previous_targets = [np.array(value, copy=True) for value in target_values]
        
    def compute_controller(self, target, robot_state: RobotJointState | None = None):
        """
            Get the inverse kinematics solution for multiple target
            @params:
                target: Dict[np.ndarray], multiple target, key: frame name, 
                    value: 7D pose, [x,y,z,qx,qy,qz,qw] 
            @return: return the actuated joint positions (dim: nv)
        """
        if robot_state is None:
            raise ValueError("robot_state should not be None for WholeBodyIk")
        
        # init of the problem
        init_q = np.asarray(robot_state._positions).flatten()
        init_guess = self.previous_solution if self.previous_solution is not None else init_q
        self.opti.set_initial(self.var_q, init_guess)
        # pin.framesForwardKinematics(self.pin_model, self.pin_data, init_q)
            
        # set parameter values
        self.opti.set_value(self.parameters[0], init_guess)

        target_values = []
        for i in range(len(target)):
            target_dict = target[i]
            frame_name = list(target_dict.keys())[0]
            value = target_dict[frame_name]
            if self.tracking_frames[i]["name"] != frame_name:
                raise ValueError("target frame name is not matching: "
                                    f"expected: {self.tracking_frames[i]['name']}, actual: {frame_name}")
            
            if self.tracking_frames[i]["require_trans"] and self.tracking_frames[i]["require_rot"]:
                target_value = convert_7D_2_homo(value)
            # @TODO: support only trans or only rot
            elif self.tracking_frames[i]["require_trans"]:
                target_value = value[:3]
            elif self.tracking_frames[i]["require_rot"]:
                target_value = convert_quat_to_rot_matrix(value[:4])
            target_values.append(target_value)
        
        # update target to the optimization problem
        for i, cur_target in enumerate(target_values):
            self.opti.set_value(self.parameters[i+1], cur_target)
        
        target_updated = self._is_target_updated(target_values)
        robot_state_updated = True
        if self.previous_robot_position is not None:
            robot_state_updated = np.linalg.norm(init_q - self.previous_robot_position) > self.joint_state_threshold
        need_solve = self.previous_solution is None or target_updated or robot_state_updated
        self.previous_robot_position = np.array(init_q, copy=True)
        self._update_target_cache(target_values)

        # solve optimization
        try:
            if need_solve:
                start = time.perf_counter()
                self.opti.solve()
                solve_time = (time.perf_counter() - start) * 1000.0
                # log.info(f'solve time: {solve_time*1000:.1f}ms')
                q_target = np.asarray(self.opti.value(self.var_q)).flatten()
                self.previous_solution = q_target
                if self.show_debug:
                    log.info(f"WBIK solve time: {solve_time:.3f}ms")
            else:
                q_target = self.previous_solution
            
            self.filter.add_data(q_target)
            q_target = np.asarray(self.filter.filtered_data).flatten()
            self.previous_solution = q_target
            # tau_eff = self._robot_model.id(q_target, robot_state._velocities, np.zeros_like(q_target))
            # q_target = np.hstack((q_target, tau_eff))
            
            if DEBUG:
                log.error(f"trans_cost: {self.opti.debug.value(self.translation_cost)}")
                log.error(f"rot_cost:   {self.opti.debug.value(self.rotation_cost)}")
                log.error(f"smooth:     {self.opti.debug.value(self.smooth_cost)}")
                log.error(f"reg:        {self.opti.debug.value(self.regularization_cost)}")
            
            return True, q_target, ["position"]*len(target)
        except RuntimeError as e:
            # Keep controller output continuous on solver hiccups.
            fallback_q = self.previous_solution if self.previous_solution is not None else init_q
            fallback_q = np.asarray(fallback_q).flatten()
            if self.show_debug:
                log.error(f"IK solve failed, using fallback joint target: {e}")
            return True, fallback_q, ["position"]*len(target)
    
    def _init_casadi_problem(self):
        self.pin_model, self.pin_data = self._robot_model.get_pin_model_N_data()
        self.cmodel = cpin.Model(self.pin_model)
        self.cdata = self.cmodel.createData()
        
        # define symbolic variables
        self.variables = [casadi.SX.sym("q", self._robot_model.nq, 1)]
        for tracking_frame in self.tracking_frames:
            # parse the frame target dimension
            self.variables.append(casadi.SX.sym(tracking_frame["name"],
                                  *tracking_frame["dim"]))        
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.variables[0])
        
        # cost functions
        trans_errors = []; trans_variables = [self.variables[0]]
        rotation_errros = []; rotation_variables = [self.variables[0]]
        for i in range(len(self.tracking_frames)):
            if self.tracking_frames[i]["require_trans"]:
                trans_errors.append(self._trans_err(self.tracking_frames[i]["name"],
                                                self.variables[1+i], 
                                                self.tracking_frames[i]["trans_scalar"]))
                trans_variables.append(self.variables[1+i])
            if self.tracking_frames[i]["require_rot"]:
                rotation_errros.append(self._rot_err(self.tracking_frames[i]["name"],
                                                self.variables[1+i],
                                                self.tracking_frames[i]["rot_scalar"]))
                rotation_variables.append(self.variables[1+i])

        self.translation_func = casadi.Function(
            "translation_error",
            trans_variables,
            [casadi.vertcat(
                    *trans_errors
                )],
        )
        self.rot_func = casadi.Function(
            "rotation_error",
            rotation_variables,
            [casadi.vertcat(
                *rotation_errros
                )],
        )
        
        if self._nullspace_tracking is not None:
            targets = []; joint_ids = []; scalars = []
            for nullspace_info in self._nullspace_tracking:
                joint_name = nullspace_info["joint_name"]
                joint_id = self.pin_model.getJointId(joint_name)
                # @TODO: hack HERE, must find a way to parse the joint id
                joint_ids.append(joint_id-1)
                targets.append(nullspace_info["position_target"])
                scalars.append(nullspace_info["scalar"])
                log.info(f'nullspace joint_id: {joint_id} for {joint_name}')
            self.null_space_func = casadi.Function(
                "nullspace_error",
                [self.variables[0]],
                [self._nullspace_err(targets, self.variables[0], joint_ids, scalars)],
            )
        
        # casadi opti
        self.opti = casadi.Opti()
        # variables
        self.var_q = self.opti.variable(self._robot_model.nq)
        # first parameters is the last q for smoothing
        self.parameters = [self.opti.parameter(self._robot_model.nq)]
        trans_params = []
        rot_params = []
        for i in range(len(self.tracking_frames)):
            curr_parameter = self.opti.parameter(*self.tracking_frames[i]["dim"])
            self.parameters.append(curr_parameter)
            if self.tracking_frames[i]["require_trans"]:
                trans_params.append(curr_parameter)
            if self.tracking_frames[i]["require_rot"]:
                rot_params.append(curr_parameter)

        # costs
        self.translation_cost = casadi.sumsqr(self.translation_func(self.var_q, *trans_params))
        self.rotation_cost = casadi.sumsqr(self.rot_func(self.var_q, *rot_params))
        self.smooth_cost = casadi.sumsqr(self.var_q - self.parameters[0])
        self.regularization_cost = casadi.sumsqr(self.var_q)
        
        # optimization constraints and objectives
        self.opti.subject_to(self.opti.bounded(
            self.pin_model.lowerPositionLimit,
            self.var_q,
            self.pin_model.upperPositionLimit)
        )
        # @TODO: add the constraint if there is mimic joints
        # self.opti.
        optimization_objectives = self.translation_weight * self.translation_cost \
                                    + self.rot_weight * self.rotation_cost \
                                    + self.regularization_weight * self.regularization_cost \
                                    + self.smooth_weight * self.smooth_cost 
        if self._nullspace_tracking is not None:
            self.nullspace_cost = self.null_space_func(self.var_q)
            optimization_objectives += casadi.sumsqr(self.nullspace_cost)
        self.opti.minimize(optimization_objectives)
        opts = {
            'ipopt':{
                'print_level':0,
                'max_iter':self.solver_max_iter,
                'tol':self.solver_tol,
                'acceptable_tol':self.acceptable_tol,
                'warm_start_init_point':'yes',
                # 以下参数可以删除
                # 'mu_strategy':'adaptive', # 自适应障碍参数
                # 'hessian_approximation':'limited-memory', # 使用L-BFGS近似
                # 'linear_solver':'mumps', # 更稳定的线性求解器， ma57 mumps
                # 'nlp_scaling_method':'gradient-based' # 基于梯度的缩放
            },
            'print_time':self.show_debug,# print or not
            'calc_lam_p':False
            # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
        }
        self.opti.solver("ipopt", opts)
        
    def _trans_err(self, frame_name, varaibale, scalar = 1.0):
        frame_id = self._robot_model.frames_name2id[frame_name]
        return scalar * (self.cdata.oMf[frame_id].translation - varaibale[:3, 3])
    
    def _rot_err(self, frame_name, variable, scalar = 1.0):
        frame_id = self._robot_model.frames_name2id[frame_name]
        R = self.cdata.oMf[frame_id].rotation
        Rd = variable[:3, :3]
        # Stable SO(3) error to avoid NaN gradients from log3 around edge cases.
        skew_err = Rd @ R.T - R @ Rd.T
        return scalar * 0.5 * casadi.vertcat(skew_err[2, 1], skew_err[0, 2], skew_err[1, 0])
        
    def _nullspace_err(self, targets:list, variable, joint_ids:list, sclars:list):
        error = 0.0
        for i, target in enumerate(targets):
            error += sclars[i] * (target - variable[joint_ids[i]])
        return error
    