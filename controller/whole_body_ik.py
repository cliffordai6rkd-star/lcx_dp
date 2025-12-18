from __future__ import annotations

from motion.pin_model import RobotModel
from controller.controller_base import ControllerBase
import casadi
import pinocchio.casadi as cpin
import pinocchio as pin
from hardware.base.utils import RobotJointState, convert_7D_2_homo, convert_quat_to_rot_matrix, convert_homo_2_7D_pose
from controller.utils.weighted_moving_filter import WeightedMovingFilter
import glog as log
import numpy as np
import time

DEBUG = False

class WholeBodyIk(ControllerBase):
    def __init__(self, config, robot_model: RobotModel):
        super().__init__(config, robot_model)
        self.tracking_frames = config["tracking_frames"]
        self.target_threshold = config.get("target_threshold", 8e-3)
        self._nullspace_tracking = config.get('nullspace_tracking', None)
        self.show_debug = config.get("show_debug", False)
        self.translation_weight = config["trans_weight"]
        self.rot_weight = config["rot_weight"]
        self.nullspace_weight = config.get("nullspace_weight", 1.0)
        self.smooth_weight = config["smooth_weight"]
        self.regularization_weight = config["regularization_weight"]
        self.filter = WeightedMovingFilter([1.0/8] * 8, self._robot_model.nq)
        
        # init of casadi opti problem
        self._init_casadi_problem()
        
        # cache for reducing solution time
        self.previous_robot_position = None
        self.previous_solution = None
        
    def compute_controller(self, target, robot_state: RobotJointState | None = None):
        """
            Get the inverse kinematics solution for multiple target
            @params:
                target: Dict[np.ndarray], multiple target, key: frame name, 
                    value: 7D pose, [x,y,z,qx,qy,qz,qw] 
            @return: return the actuated joint positions (dim: nv)
        """
        # init of the problem
        init_q = robot_state._positions
        self.opti.set_initial(self.var_q, init_q)
        
        # if self.previous_robot_position is None or not np.allclose(init_q, self.previous_robot_position):
        # log.info(f'Updated the kinematics!!!!')
        pin.framesForwardKinematics(self.pin_model, self.pin_data, init_q)
        pin.computeForwardKinematicsDerivatives(self.pin_model, self.pin_data, init_q, 
                                                robot_state._velocities, 
                                                robot_state._accelerations)
        self.previous_robot_position = init_q  
            
        # set parameter values
        self.opti.set_value(self.parameters[0], init_q)

        target_updated = False
        target_values = []
        for i in range(len(target)):
            target_dict = target[i]
            frame_name = list(target_dict.keys())[0]
            value = target_dict[frame_name]
            if self.tracking_frames[i]["name"] != frame_name:
                raise ValueError("target frame name is not matching: "
                                    f"expected: {self.tracking_frames[i]['name']}, actual: {frame_name}")
            
            tracking_frame_name = self.tracking_frames[i]["name"]
            cur_state = self._robot_model.get_frame_pose(tracking_frame_name, init_q, False, "single")
            cur_state = convert_homo_2_7D_pose(cur_state)
            if self.tracking_frames[i]["require_trans"] and self.tracking_frames[i]["require_rot"]:
                raw_value = value 
                target_value = convert_7D_2_homo(raw_value)
            # @TODO: support only trans or only rot
            elif self.tracking_frames[i]["require_trans"]:
                raw_value = value[:3]; cur_state = cur_state[:3]
                target_value = raw_value
            elif self.tracking_frames[i]["require_rot"]:
                raw_value = value[:4]; cur_state = cur_state[3:]
                target_value = convert_quat_to_rot_matrix(raw_value)
            target_values.append(target_value)
            
            if self.previous_solution is not None:
                # {self.parameters[i + 1].shape}
                diff = np.linalg.norm(cur_state - raw_value)
                if diff > self.target_threshold:
                    target_updated = True
                # log.info(f'Target updated {target_updated} for {i}th target with diff {diff}')
        
        # update target to the optimization problem
        if target_updated or self.previous_solution is None:
            for i, cur_target in enumerate(target_values):
                self.opti.set_value(self.parameters[i+1], cur_target)
            
        # solve optimization
        try:
            if target_updated or self.previous_solution is None:
                start = time.perf_counter()
                soln = self.opti.solve()
                solve_time = time.perf_counter() - start
                # log.info(f'solve time: {solve_time*1000:.1f}ms')
                q_target = self.opti.value(self.var_q)
                self.previous_solution = q_target
            else: q_target = self.previous_solution
            
            self.filter.add_data(q_target)
            q_target = self.filter.filtered_data
            # tau_eff = self._robot_model.id(q_target, robot_state._velocities, np.zeros_like(q_target))
            # q_target = np.hstack((q_target, tau_eff))
            
            if DEBUG:
                log.error(f"trans_cost: {self.opti.debug.value(self.translation_cost)}")
                log.error(f"rot_cost:   {self.opti.debug.value(self.rotation_cost)}")
                log.error(f"smooth:     {self.opti.debug.value(self.smooth_cost)}")
                log.error(f"reg:        {self.opti.debug.value(self.regularization_cost)}")
            
            return True, q_target, ["position"]*len(target)
        except RuntimeError as e:
            # print(f"ERROR in convergence{e}")
            log.error(f"IK solve failed: {e}")

            # ---- 关键：只用 debug.value / debug.show_infeasibilities ----
            try:
                q_dbg = self.opti.debug.value(self.var_q)
                f_dbg = self.opti.debug.value(self.opti.f)
                g_dbg = self.opti.debug.value(self.opti.g)
                log.error(f"q_dbg: {q_dbg.T}")
                log.error(f"f_dbg: {f_dbg}")
                log.error(f"g has non-finite: {~np.isfinite(g_dbg).all()}")
                self.opti.debug.show_infeasibilities()
                log.error(f"trans_cost: {self.opti.debug.value(self.translation_cost)}")
                log.error(f"rot_cost:   {self.opti.debug.value(self.rotation_cost)}")
                log.error(f"smooth:     {self.opti.debug.value(self.smooth_cost)}")
                log.error(f"reg:        {self.opti.debug.value(self.regularization_cost)}")
                dq_trans = casadi.jacobian(self.translation_cost, self.var_q)
                dq_rot   = casadi.jacobian(self.rotation_cost, self.var_q)
                dq_smooth= casadi.jacobian(self.smooth_cost, self.var_q)
                dq_reg   = casadi.jacobian(self.regularization_cost, self.var_q)

                gt = self.opti.debug.value(dq_trans)
                gr = self.opti.debug.value(dq_rot)
                gs = self.opti.debug.value(dq_smooth)
                gg = self.opti.debug.value(dq_reg)

                log.error(f"grad trans finite? {np.isfinite(gt).all()}")
                log.error(f"grad rot   finite? {np.isfinite(gr).all()}")
                log.error(f"grad smooth finite? {np.isfinite(gs).all()}")
                log.error(f"grad reg   finite? {np.isfinite(gg).all()}")
            except Exception as ee:
                log.error(f"debug dump failed: {ee}")
            return False, None, "position"
    
    
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
        # first parameters is the last q
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
                'max_iter':50, # 增加最大迭代次数
                'tol':1e-5,     # 更严格的收敛条件
                # 以下参数可以删除
                # 'acceptable_tol':1e-5, # 可接受的容差
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
        R_err = variable[:3, :3] @ self.cdata.oMf[frame_id].rotation.T
        err = scalar * cpin.log3(R_err)
        # R  = self.cdata.oMf[frame_id].rotation      # 当前 R
        # Rd = variable[:3, :3]                       # 目标 Rd
        # err = scalar * casadi.reshape(Rd - R, 9, 1)
        return err
        
    def _nullspace_err(self, targets:list, variable, joint_ids:list, sclars:list):
        error = 0.0
        for i, target in enumerate(targets):
            error += sclars[i] * (target - variable[joint_ids[i]])
        return error