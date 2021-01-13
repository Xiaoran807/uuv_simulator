#!/usr/bin/env python
# Copyright (c) 2016-2019 The UUV Simulator Authors.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import print_function
import rospy
import numpy as np
from uuv_control_interfaces import DPControllerBase, DPPIDControllerBase
from uuv_control_msgs.srv import *
from uuv_control_interfaces.vehicle import cross_product_operator
from std_msgs.msg import Int32



class ROV_MB_SMController(DPControllerBase):
    _LABEL = 'Model-based Sliding Mode Controller'

    def __init__(self):
        DPControllerBase.__init__(self, True)
        self._logger.info('Initializing: ' + self._LABEL)

        # Lambda - Slope of the Sliding Surface
        self._lambda = np.zeros(6)
        # Rho Constant - Vector of positive terms for assuring sliding surface reaching condition
        self._rho_constant = np.zeros(6)
        # k - PD gain (P term = k * lambda , D term = k)
        self._k = np.zeros(6)
        # c - slope of arctan (the greater, the more similar with the sign function)
        self._c = np.zeros(6)
        # Adapt slope - Adaptation gain for the estimation of uncertainties
        # and disturbances upper boundaries
        # adapt_slope = [proportional to surface distance, prop. to square
        # of pose errors, prop. to square of velocity errors]
        self._adapt_slope = np.zeros(3)
        # Rho_0 - rho_adapt treshold for drift prevention
        self._rho_0 = np.zeros(6)
        # Drift prevent - Drift prevention slope
        self._drift_prevent = 0

        if rospy.has_param('~lambda'):
            coefs = rospy.get_param('~lambda')
            if len(coefs) == 6:
                self._lambda = np.array(coefs)
            else:
                raise rospy.ROSException('lambda coefficients: 6 coefficients '
                                         'needed')
        print('lambda=', self._lambda)

        if rospy.has_param('~rho_constant'):
            coefs = rospy.get_param('~rho_constant')
            if len(coefs) == 6:
                self._rho_constant = np.array(coefs)
            else:
                raise rospy.ROSException('rho_constant coefficients: 6 coefficients '
                                         'needed')
        print('rho_constant=', self._rho_constant)

        if rospy.has_param('~k'):
            coefs = rospy.get_param('~k')
            if len(coefs) == 6:
                self._k = np.array(coefs)
            else:
                raise rospy.ROSException('k coefficients: 6 coefficients '
                                         'needed')
        print('k=', self._k)

        if rospy.has_param('~c'):
            coefs = rospy.get_param('~c')
            if len(coefs) == 6:
                self._c = np.array(coefs)
            else:
                raise rospy.ROSException('c coefficients: 6 coefficients '
                                         'needed')
        print('c=', self._c)

        if rospy.has_param('~adapt_slope'):
            coefs = rospy.get_param('~adapt_slope')
            if len(coefs) == 3:
                self._adapt_slope = np.array(coefs)
            else:
                raise rospy.ROSException('adapt_slope coefficients: 6 coefficients '
                                         'needed')
        print('adapt_slope=', self._adapt_slope)

        if rospy.has_param('~rho_0'):
            coefs = rospy.get_param('~rho_0')
            if len(coefs) == 6:
                self._rho_0 = np.array(coefs)
            else:
                raise rospy.ROSException('rho_0 coefficients: 6 coefficients '
                                         'needed')
        print('rho_0=', self._rho_0)

        if rospy.has_param('~drift_prevent'):
            scalar = rospy.get_param('~drift_prevent')
            if not isinstance(scalar, list):
                self._drift_prevent = scalar
            else:
                raise rospy.ROSException('drift_prevent needs to be a scalar value')

        print('drift_prevent=', self._drift_prevent)

        # Enable(1) / disable(0) integral term in the sliding surface
        if rospy.has_param('~enable_integral_term'):
            self._sliding_int = rospy.get_param('~enable_integral_term')
        else:
            self._sliding_int = 0

        # Enable(1) / disable(0) adaptive uncertainty upper boundaries for
        # robust control
        if rospy.has_param('~adaptive_bounds'):
            self._adaptive_bounds = rospy.get_param('~adaptive_bounds')
        else:
            self._adaptive_bounds = 1

        # Enable(1) / disable(0) constant uncertainty upper boundaries for
        # robust control
        if rospy.has_param('~constant_bound'):
            self._constant_bound = rospy.get_param('~constant_bound')
        else:
            self._constant_bound = 1

        # Enable(1) / disable(0) equivalent control term
        if rospy.has_param('~ctrl_eq'):
            self._ctrl_eq = rospy.get_param('~ctrl_eq')
        else:
            self._ctrl_eq = 1

        # Enable(1) / disable(0) linear control term
        if rospy.has_param('~ctrl_lin'):
            self._ctrl_lin = rospy.get_param('~ctrl_lin')
        else:
            self._ctrl_lin = 1

        # Enable(1) / disable(0) robust control term
        if rospy.has_param('~ctrl_robust'):
            self._ctrl_robust = rospy.get_param('~ctrl_robust')
        else:
            self._ctrl_robust = 1
        # Integrator component
        self._int = np.zeros(6)
        # Error for the vehicle pose
        self._error_pose = np.zeros(6)
        # Sliding Surface
        self._s_b = np.zeros(6)
        # Time derivative of the rotation matrix
        self._rotBtoI_dot = np.zeros(shape=(3, 3), dtype=float)
        # Linear acceleration estimation
        self._accel_linear_estimate_b = np.zeros(3)
        # Angular acceleration estimation
        self._accel_angular_estimate_b = np.zeros(3)
        # Acceleration estimation
        self._accel_estimate_b = np.zeros(6)
        # adaptive term of uncertainties upper bound estimation
        self._rho_adapt = np.zeros(6)
        # Upper bound for model uncertainties and disturbances
        self._rho_total = np.zeros(6)
        # Equivalent control
        self._f_eq = np.zeros(6)
        # Linear term of controller
        self._f_lin = np.zeros(6)
        # Robust control
        self._f_robust = np.zeros(6)
        # Total control
        self._tau = np.zeros(6)
	self._slidingSurface=np.zeros(6)
	self._vel=np.zeros(3)
	self._vehi=np.zeros(1)

        self._services['set_mb_sm_controller_params'] = rospy.Service(
            'set_mb_sm_controller_params',
            SetMBSMControllerParams,
            self.set_mb_sm_controller_params_callback)

        self._services['get_mb_sm_controller_params'] = rospy.Service(
            'get_mb_sm_controller_params',
            GetMBSMControllerParams,
            self.get_mb_sm_controller_params_callback)
        self._is_init = True
        self._logger.info(self._LABEL + ' ready')

        # Feedback acceleration gain
        self._Hm = np.eye(6)
        if rospy.has_param('Hm'):
            hm = rospy.get_param('Hm')
            if len(hm) == 6:
                self._Hm = self._vehicle_model.Mtotal +  np.diag(hm)
            else:
                raise rospy.ROSException('Invalid feedback acceleration gain coefficients')

        self._accel_ff = np.zeros(6)
        self.e_v_b_prev=np.zeros(6)
        self.vel_veh_b_prev=np.zeros(6)
        self.error_pose_Euler_prev=np.zeros(6)
        self.errorPoseEuler=np.zeros(6)
        self.acc_ref_prev=np.zeros(6)
        self.e_p_b_prev=np.zeros(6)
        self._int_e=np.zeros(6)
        self._error_pose_prev=np.zeros(6)
        self.S_prev=np.zeros(3)
        self.rho_x_prev=0
        self.rho_y_prev=0
        self.rho_z_prev=0
        self.F_tau=np.zeros(6)
        self.s_x_prev=0
        self.s_y_prev=0
        self.s_z_prev=0
        self.F_x_prev=0
        self.F_y_prev=0
        self.F_z_prev=0
        self.h_hat_x=0
        self.h_hat_y=0
        self.h_hat_z=0
        self.acc=np.zeros(6)
        self.error_int_prev=np.zeros(6)
        self.acc_prev1=np.zeros(6)
        self.acc_prev2=np.zeros(6)
        self.F_x_prev=0
        self.F_y_prev=0
        self.F_z_prev=0
        self.F_x_prev1=0
        self.F_y_prev1=0
        self.F_z_prev1=0
        self.F_x_prev2=0
        self.F_y_prev2=0
        self.F_z_prev2=0
        self.vel_vehicle_prev=np.zeros(3)
        self.acc_imu_prev1=np.zeros(3)
        self.acc_imu_prev2=np.zeros(3)
        self.acc_imu_prev3=np.zeros(3)
        self.acc_imu_prev4=np.zeros(3)
        self.acc_imu_prev5=np.zeros(3)
        self.acc_imu_prev6=np.zeros(3)
        self.acc_imu_prev7=np.zeros(3)
        self.acc_imu_prev8=np.zeros(3)
        self.vehicle_model_vel_prev=np.zeros(3)
        self.h_hat_p=0
        self.h_hat_q=0
        self.h_hat_r=0
        self.rho_p_prev=0
        self.rho_q_prev=0
        self.rho_r_prev=0
        self.F_p_prev=0
        self.F_q_prev=0
        self.F_r_prev=0
        self.acc_angular_prev=np.zeros(3)
        self.acc_angular_prev1=np.zeros(3)
        self.acc_angular_prev2=np.zeros(3)
        self.acc_angular_prev3=np.zeros(3)
        self.acc_angular_prev4=np.zeros(3)
        self.acc_angular_prev5=np.zeros(3)
        self.acc_angular_prev6=np.zeros(3)
        self.acc_angular_prev7=np.zeros(3)
        self.acc_angular_prev8=np.zeros(3)
        self._int_e_prev=np.zeros(6)
        self.acc_cal_fromVel_prev=np.zeros(3)

    def _reset_controller(self):
        super(ROV_MB_SMController, self)._reset_controller()
        self._sliding_int = 0
        self._adaptive_bounds = 0
        self._constant_bound = 0
        self._ctrl_eq = 0
        self._ctrl_lin = 0
        self._ctrl_robust = 0
        self._prev_t = 0
        self._int = np.zeros(6)
        self._error_pose = np.zeros(6)
        self._s_b = np.zeros(6)
        self._rotBtoI_dot = np.zeros(shape=(3, 3), dtype=float)
        self._accel_linear_estimate_b = np.zeros(3)
        self._accel_angular_estimate_b = np.zeros(3)
        self._accel_estimate_b = np.zeros(6)
        self._rho_adapt = np.zeros(6)
        self._rho_total = np.zeros(6)
        self._f_eq = np.zeros(6)
        self._f_lin = np.zeros(6)
        self._f_robust = np.zeros(6)
        self._tau = np.zeros(6)
	self._slidingSurface=np.zeros(6)
	self._vel=np.zeros(3)
	self._vehi=np.zeros(1)
        self._accel_ff = np.zeros(6)	
        self.vel_veh_b_prev=np.zeros(6)
        self._int_e=np.zeros(6)
        self._error_pose_prev=np.zeros(6)
        self.F_tau=np.zeros(6)
        self.h_hat_x=0
        self.h_hat_y=0
        self.h_hat_z=0
        self.acc=np.zeros(6)

    def set_mb_sm_controller_params_callback(self, request):
        return SetMBSMControllerParamsResponse(True)

    def get_mb_sm_controller_params_callback(self, request):
        return GetMBSMControllerParamsResponse(
            self._lambda.tolist(),
            self._rho_constant.tolist(),
            self._k.tolist(),
            self._c.tolist(),
            self._adapt_slope.tolist(),
            self._rho_0.tolist(),
            self._drift_prevent)


        # default smc
    #def update_controller(self):
    #    if not self._is_init:
    #        return False
    #    t = rospy.Time.now().to_sec()
    #    dt = t - self._prev_t
    #    if self._prev_t < 0.0:
    #        dt = 0.05
    #    self._int += 0.5 * (self.error_pose_euler - self._error_pose) * self._dt
    #    self._error_pose = self.error_pose_euler
    #    e_p_linear_b = self._errors['pos']
    #    e_v_linear_b = self._errors['vel'][0:3]
    #    e_p_angular_b = self.error_orientation_rpy
    #    e_v_angular_b = self._errors['vel'][3:6]
    #    e_p_b = np.hstack((e_p_linear_b, e_p_angular_b))
    #    e_v_b = np.hstack((e_v_linear_b, e_v_angular_b))
    #    self._s_b = -e_v_b - np.multiply(self._lambda, e_p_b) \
    #                - self._sliding_int * np.multiply(np.square(self._lambda)/4, self._int)
    #    self._rotBtoI_dot = np.dot(cross_product_operator(self._vehicle_model._vel[3:6]), self._vehicle_model.rotBtoI)
    #    self._accel_linear_estimate_b = np.dot(
    #        self._vehicle_model.rotItoB, (self._reference['acc'][0:3] - \
    #                                      np.dot(self._rotBtoI_dot, self._vehicle_model._vel[0:3]))) + \
    #                                      np.multiply(self._lambda[0:3], e_v_linear_b) + \
    #                                      self._sliding_int * np.multiply(np.square(self._lambda[0:3]) / 4, e_p_linear_b)
    #    self._accel_angular_estimate_b = np.dot(self._vehicle_model.rotItoB, (self._reference['acc'][3:6] -
    #                                            np.dot(self._rotBtoI_dot, self._vehicle_model._vel[3:6]))) + \
    #                                            np.multiply(self._lambda[3:6], e_v_angular_b) + \
    #                                            self._sliding_int * np.multiply(np.square(self._lambda[3:6]) / 4,
    #                                                                            e_p_angular_b)
    #    self._accel_estimate_b = np.hstack((self._accel_linear_estimate_b, self._accel_angular_estimate_b))
    #    acc = self._vehicle_model.to_SNAME(self._accel_estimate_b)
    #    self._f_eq = self._vehicle_model.compute_force(acc, use_sname=False)
    #    self._f_lin = - np.multiply(self._k, self._s_b)
    #    self._rho_total = self._adaptive_bounds * self._rho_adapt + self._constant_bound * self._rho_constant
    #    self._rho_adapt = self._rho_adapt + \
    #                      (self._adapt_slope[0] * np.abs(self._s_b) +
    #                      (self._adapt_slope[1] * np.abs(self._s_b) * np.abs(e_p_b) * np.abs(e_p_b)) +
    #                      (self._adapt_slope[2] * np.abs(self._s_b) * np.abs(e_v_b) * np.abs(e_v_b)) +
    #                       self._drift_prevent * (self._rho_0 - self._rho_adapt)) * dt
    #    self._f_robust = - np.multiply(self._rho_total, (2 / np.pi) * np.arctan(np.multiply(self._c, self._s_b)))
    #    self.F_tau = self._ctrl_eq * self._f_eq + self._ctrl_lin * self._f_lin + self._ctrl_robust * self._f_robust
   # 	self._slidingSurface=self._s_b
   #     self._tau[0]=self.F_tau[0]
  #      self._tau[1]=self.F_tau[1]
  #      self._tau[2]=self.F_tau[2]
  #      self._tau[3]=self.F_tau[3]
  #      self._tau[4]=self.F_tau[4]
  #      self._tau[5]=self.F_tau[5]


        # Default mb smc with delay in ref
    #def update_controller(self):
    #    if not self._is_init:
     #       return False
     #   t = rospy.Time.now().to_sec()
     #   dt = t - self._prev_t
     #   if self._prev_t < 0.0:
     #       dt = 0.05
     #   error_pose_ref_d=self.error_pose_euler_ref_d
     #   self._int += 0.5 * (error_pose_ref_d - self._error_pose_prev) * self._dt
     #   e_p_linear_b = self._errors_ref_d['pos']
     #   e_v_linear_b = self._errors_ref_d['vel'][0:3]
     #   e_p_angular_b = self.error_orientation_rpy_ref_d
     #   e_v_angular_b = self._errors_ref_d['vel'][3:6]
     #   e_p_b = np.hstack((e_p_linear_b, e_p_angular_b))
     #   e_v_b = np.hstack((e_v_linear_b, e_v_angular_b))
     #   self._s_b = -e_v_b - np.multiply(self._lambda, e_p_b) \
     #               - self._sliding_int * np.multiply(np.square(self._lambda)/4, self._int)
     #   self._rotBtoI_dot = np.dot(cross_product_operator(self._vehicle_model._vel[3:6]), self._vehicle_model.rotBtoI)
     #   self._accel_linear_estimate_b = np.dot(
     #       self._vehicle_model.rotItoB, (self.acc_ref_prev7[0:3] - \
     #                                     np.dot(self._rotBtoI_dot, self._vehicle_model._vel[0:3]))) + \
     #                                     np.multiply(self._lambda[0:3], e_v_linear_b) + \
     #                                     self._sliding_int * np.multiply(np.square(self._lambda[0:3]) / 4, e_p_linear_b)
     #   self._accel_angular_estimate_b = np.dot(self._vehicle_model.rotItoB, (self.acc_ref_prev7[3:6] -
     #                                           np.dot(self._rotBtoI_dot, self._vehicle_model._vel[3:6]))) + \
     #                                           np.multiply(self._lambda[3:6], e_v_angular_b) + \
     #                                           self._sliding_int * np.multiply(np.square(self._lambda[3:6]) / 4,
     #                                                                           e_p_angular_b)
     #   self._accel_estimate_b = np.hstack((self._accel_linear_estimate_b, self._accel_angular_estimate_b))
     #   acc = self._vehicle_model.to_SNAME(self._accel_estimate_b)
     #   self._f_eq = self._vehicle_model.compute_force(acc, use_sname=False)
     #   self._f_lin = - np.multiply(self._k, self._s_b)
     #   self._rho_total = self._adaptive_bounds * self._rho_adapt + self._constant_bound * self._rho_constant
     #   self._rho_adapt = self._rho_adapt + \
     #                     (self._adapt_slope[0] * np.abs(self._s_b) +
     #                     (self._adapt_slope[1] * np.abs(self._s_b) * np.abs(e_p_b) * np.abs(e_p_b)) +
     #                     (self._adapt_slope[2] * np.abs(self._s_b) * np.abs(e_v_b) * np.abs(e_v_b)) +
     #                      self._drift_prevent * (self._rho_0 - self._rho_adapt)) * dt
     #   self._f_robust = - np.multiply(self._rho_total, (2 / np.pi) * np.arctan(np.multiply(self._c, self._s_b)))
     #   self.F_tau = self._ctrl_eq * self._f_eq + self._ctrl_lin * self._f_lin + self._ctrl_robust * self._f_robust
     #   self._slidingSurface=self._s_b
     #   self._error_pose_prev = error_pose_ref_d
     #   self._tau[0]=self.F_tau[0]
     #   self._tau[1]=self.F_tau[1]
     #   self._tau[2]=self.F_tau[2]
     #   self._tau[3]=self.F_tau[3]
     #   self._tau[4]=self.F_tau[4]
     #   self._tau[5]=self.F_tau[5]


        # default smc with delay in position measurement
    #def update_controller(self):
    #    if not self._is_init:
    #        return False
    #    t = rospy.Time.now().to_sec()
    #    dt = t - self._prev_t
    #    if self._prev_t < 0.0:
    #        dt = 0.05
    #    self._int += 0.5 * (self.error_pose_euler_veh_d - self._error_pose) * self._dt
    #    self._error_pose = self.error_pose_euler_veh_d
    #    e_p_linear_b = self._errors_veh_d['pos']
    #    e_v_linear_b = self._errors_veh_d['vel'][0:3]
    #    e_p_angular_b = self.error_orientation_rpy_veh_d
    #    e_v_angular_b = self._errors_veh_d['vel'][3:6]
    #    e_p_b = np.hstack((e_p_linear_b, e_p_angular_b))
    #    e_v_b = np.hstack((e_v_linear_b, e_v_angular_b))
    #    self._s_b = -e_v_b - np.multiply(self._lambda, e_p_b) \
    #                - self._sliding_int * np.multiply(np.square(self._lambda)/4, self._int)
    #    self._rotBtoI_dot = np.dot(cross_product_operator(self.vel_veh_prev4[3:6]), self._vehicle_model.rotBtoI)
    #    self._accel_linear_estimate_b = np.dot(
    #        self._vehicle_model.rotItoB, (self._reference['acc'][0:3] - \
    #                                      np.dot(self._rotBtoI_dot, self.vel_veh_prev4[0:3]))) + \
    #                                      np.multiply(self._lambda[0:3], e_v_linear_b) + \
    #                                      self._sliding_int * np.multiply(np.square(self._lambda[0:3]) / 4, e_p_linear_b)
    #    self._accel_angular_estimate_b = np.dot(self._vehicle_model.rotItoB, (self._reference['acc'][3:6] -
    #                                            np.dot(self._rotBtoI_dot, self.vel_veh_prev4[3:6]))) + \
    #                                            np.multiply(self._lambda[3:6], e_v_angular_b) + \
    #                                            self._sliding_int * np.multiply(np.square(self._lambda[3:6]) / 4,
    #                                                                            e_p_angular_b)
    #    self._accel_estimate_b = np.hstack((self._accel_linear_estimate_b, self._accel_angular_estimate_b))
    #    acc = self._vehicle_model.to_SNAME(self._accel_estimate_b)
    #    self._f_eq = self._vehicle_model.compute_force(acc, use_sname=False)
    #    self._f_lin = - np.multiply(self._k, self._s_b)
    #    self._rho_total = self._adaptive_bounds * self._rho_adapt + self._constant_bound * self._rho_constant
    #    self._rho_adapt = self._rho_adapt + \
    #                      (self._adapt_slope[0] * np.abs(self._s_b) +
    #                      (self._adapt_slope[1] * np.abs(self._s_b) * np.abs(e_p_b) * np.abs(e_p_b)) +
    #                      (self._adapt_slope[2] * np.abs(self._s_b) * np.abs(e_v_b) * np.abs(e_v_b)) +
    #                       self._drift_prevent * (self._rho_0 - self._rho_adapt)) * dt
    #    self._f_robust = - np.multiply(self._rho_total, (2 / np.pi) * np.arctan(np.multiply(self._c, self._s_b)))
    #    self.F_tau = self._ctrl_eq * self._f_eq + self._ctrl_lin * self._f_lin + self._ctrl_robust * self._f_robust
    #	self._slidingSurface=self._s_b
    #    self._tau[0]=self.F_tau[0]
    #    self._tau[1]=self.F_tau[1]
    #    self._tau[2]=self.F_tau[2]
    #    self._tau[3]=self.F_tau[3]
    #    self._tau[4]=self.F_tau[4]
    #    self._tau[5]=self.F_tau[5]






        #  Proposed control without delay
    #def update_controller(self):
    #    if not self._is_init:
    #        return False
    #    t = rospy.Time.now().to_sec()
    #    dt = t - self._prev_t
    #    if self._prev_t < 0.0:
    #        dt = 0.05
    #    self._int += 0.5 * (self.error_pose_euler - self._error_pose) * self._dt
    #    self._error_pose = self.error_pose_euler
    #    e_p_linear_b = self._errors['pos']
    #    e_v_linear_b = self._errors['vel'][0:3]
    #    e_p_angular_b = self.error_orientation_rpy
    #    e_v_angular_b = self._errors['vel'][3:6]
    #    e_p_b = np.hstack((e_p_linear_b, e_p_angular_b))
    #    e_v_b = np.hstack((e_v_linear_b, e_v_angular_b))
    #    self._s_b = -e_v_b - np.multiply(self._lambda, e_p_b) \
    #                - self._sliding_int * np.multiply(np.square(self._lambda)/4, self._int)
    #    self._rotBtoI_dot = np.dot(cross_product_operator(self._vehicle_model._vel[3:6]), self._vehicle_model.rotBtoI)
    #    self._accel_linear_estimate_b = np.dot(
    #        self._vehicle_model.rotItoB, (self._reference['acc'][0:3] - \
    #                                      np.dot(self._rotBtoI_dot, self._vehicle_model._vel[0:3]))) + \
     #                                     np.multiply(self._lambda[0:3], e_v_linear_b) + \
     #                                     self._sliding_int * np.multiply(np.square(self._lambda[0:3]) / 4, e_p_linear_b)
     #   self._accel_angular_estimate_b = np.dot(self._vehicle_model.rotItoB, (self._reference['acc'][3:6] -
     #                                           np.dot(self._rotBtoI_dot, self._vehicle_model._vel[3:6]))) + \
     #                                           np.multiply(self._lambda[3:6], e_v_angular_b) + \
     #                                           self._sliding_int * np.multiply(np.square(self._lambda[3:6]) / 4,
     #                                                                           e_p_angular_b)
     #   self._accel_estimate_b = np.hstack((self._accel_linear_estimate_b, self._accel_angular_estimate_b))
     #   acc = self._vehicle_model.to_SNAME(self._accel_estimate_b)
     #   self._f_eq = self._vehicle_model.compute_force(acc, use_sname=False)
     #   self._f_lin = - np.multiply(self._k, self._s_b)
     #   self._rho_total = self._adaptive_bounds * self._rho_adapt + self._constant_bound * self._rho_constant
     #   self._rho_adapt = self._rho_adapt + \
     #                     (self._adapt_slope[0] * np.abs(self._s_b) +
     #                     (self._adapt_slope[1] * np.abs(self._s_b) * np.abs(e_p_b) * np.abs(e_p_b)) +
     #                     (self._adapt_slope[2] * np.abs(self._s_b) * np.abs(e_v_b) * np.abs(e_v_b)) +
     #                      self._drift_prevent * (self._rho_0 - self._rho_adapt)) * dt
     #   self._f_robust = - np.multiply(self._rho_total, (2 / np.pi) * np.arctan(np.multiply(self._c, self._s_b)))
     #   self.F_tau = self._ctrl_eq * self._f_eq + self._ctrl_lin * self._f_lin + self._ctrl_robust * self._f_robust
     #   kp=0.12
     #   ki=0.1
     #   kd=0.1
     #   mu=0.1
     #   m_bar_x=1042#100, 2642
     #   m_bar_y=1000#300, 3084
     #   m_bar_z=1500#200, 5522
     #   delta=1.0001
     #   beta=.00001#.00001
     #   Ldelta_c_x=8000
     #   Ldelta_c_y=8000
     #   Ldelta_c_z=8000
     #   LDelta_d_x=100
     #   LDelta_d_y=100
     #   LDelta_d_z=100
     #   H_para=.4
     #   acc_imu=self.imuAccLinear
     #   acc_cal_fromVel=(self._vehicle_model._vel[0:3]-self.vel_vehicle_prev)/dt
     #   e_p_linear_b = self._errors['pos']
     #   e_v_linear_b = self._errors['vel'][0:3]
     #   e_p_angular_b = self.error_orientation_rpy
     #   e_v_angular_b = self._errors['vel'][3:6]
     #   e_p_b = np.hstack((e_p_linear_b, e_p_angular_b))
     #   e_v_b = np.hstack((e_v_linear_b, e_v_angular_b))
     #   acc_ref=self._vehicle_model.to_SNAME(self._reference['acc'])
     #   acc_ref_diff=acc_ref-self.acc_ref_prev
     #   e_p_b_diff=e_p_b-self.e_p_b_prev
     #   e_v_b_diff=e_v_b-self.e_v_b_prev
     #   error_pose=self.error_pose_euler
     #   self._int_e += 0.5 * (error_pose - self._error_pose_prev) * self._dt
     #   s_x=kp*e_p_b[0]+ki*self._int_e[0]+kd*e_v_b[0]        
     #   s_y=kp*e_p_b[1]+ki*self._int_e[1]+kd*e_v_b[1]        
     #   s_z=kp*e_p_b[2]+ki*self._int_e[2]+kd*e_v_b[2]        
     #   s_diff_x=s_x-self.s_x_prev
     #   s_diff_y=s_y-self.s_y_prev
     #   s_diff_z=s_z-self.s_z_prev
     #   S=np.hstack((s_x,s_y,s_z,0,0,0))
      #  #LDelta_d_x=np.abs(acc_ref_diff[0]+kp/kd*e_v_b_diff[0]+ki/kd*e_p_b_diff[0]+1/kd/mu*s_diff_x)
      #  #LDelta_d_y=np.abs(acc_ref_diff[1]+kp/kd*e_v_b_diff[1]+ki/kd*e_p_b_diff[1]+1/kd/mu*s_diff_y)
      #  #LDelta_d_z=np.abs(acc_ref_diff[2]+kp/kd*e_v_b_diff[2]+ki/kd*e_p_b_diff[2]+1/kd/mu*s_diff_z)
      #  rho_x=delta*(self.rho_x_prev+kd*LDelta_d_x+kd*Ldelta_c_x)/(m_bar_x-delta)
      #  rho_y=delta*(self.rho_y_prev+kd*LDelta_d_y+kd*Ldelta_c_y)/(m_bar_y-delta)
      #  rho_z=delta*(self.rho_z_prev+kd*LDelta_d_z+kd*Ldelta_c_z)/(m_bar_z-delta)
      #  v_x=acc_ref[0]+kp/kd*e_v_b[0]+ki/kd*e_p_b[0]+1/mu/kd*s_x+rho_x/kd*np.tanh(s_x/beta)
      #  v_y=acc_ref[1]+kp/kd*e_v_b[1]+ki/kd*e_p_b[1]+1/mu/kd*s_y+rho_y/kd*np.tanh(s_y/beta)
      #  v_z=acc_ref[2]+kp/kd*e_v_b[2]+ki/kd*e_p_b[2]+1/mu/kd*s_z+rho_x/kd*np.tanh(s_z/beta)
      #  self.h_hat_x=.6*(self.F_x_prev-m_bar_x*self.acc_imu_prev1[0])
      #  self.h_hat_y=.6*(self.F_y_prev-m_bar_y*self.acc_imu_prev1[1])
      #  self.h_hat_z=.2*(self.F_z_prev-m_bar_z*self.acc_imu_prev1[2])
   #     H_hat=np.hstack((self.h_hat_x,self.h_hat_y,self.h_hat_z))
   #     F_x=m_bar_x*v_x+self.h_hat_x
   #     F_y=m_bar_y*v_y+self.h_hat_y
   #     F_z=m_bar_z*v_z+self.h_hat_z   
   #     F=np.hstack((F_x,F_y,F_z,self.F_tau[3],self.F_tau[4],self.F_tau[5]))
   #     F_SNAME=self._vehicle_model.to_SNAME(F)
   #     self.acc = self._vehicle_model.compute_acc(
   #         F_SNAME, use_sname=False)
    #    self._error_pose_prev = error_pose
    #    self.acc_ref_prev=acc_ref
    #    self.e_p_b_prev=e_p_b
    #    self.e_v_b_prev=e_v_b
    #    self.s_x_prev=s_x
    #    self.s_y_prev=s_y
    #    self.s_z_prev=s_z
    #    self.rho_x_prev=rho_x  
    #    self.rho_y_prev=rho_y
    #    self.rho_z_prev=rho_z
    #    self.F_x_prev=F_x
    #    self.F_y_prev=F_y
    #    self.F_z_prev=F_z
    #    self.acc_imu_prev1=acc_imu
    #    self.vel_vehicle_prev=self._vehicle_model._vel[0:3]
    #    self._slidingSurface=S
    #    self._tau[0]=F[0]
    #    self._tau[1]=F[1]
    #    self._tau[2]=F[2]
    #    self._tau[3]=self.F_tau[3]
    #    self._tau[4]=self.F_tau[4]
    #    self._tau[5]=self.F_tau[5]



        #  Proposed control without delay, full 6 dof
    def update_controller(self):
        if not self._is_init:
            return False
        t = rospy.Time.now().to_sec()
        dt = t - self._prev_t
        if self._prev_t < 0.0:
            dt = 0.05
        self._int += 0.5 * (self.error_pose_euler - self._error_pose) * self._dt
        self._error_pose = self.error_pose_euler
        e_p_linear_b = self._errors['pos']
        e_v_linear_b = self._errors['vel'][0:3]
        e_p_angular_b = self.error_orientation_rpy
        e_v_angular_b = self._errors['vel'][3:6]
        e_p_b = np.hstack((e_p_linear_b, e_p_angular_b))
        e_v_b = np.hstack((e_v_linear_b, e_v_angular_b))
        self._s_b = -e_v_b - np.multiply(self._lambda, e_p_b) \
                    - self._sliding_int * np.multiply(np.square(self._lambda)/4, self._int)
        self._rotBtoI_dot = np.dot(cross_product_operator(self._vehicle_model._vel[3:6]), self._vehicle_model.rotBtoI)
        self._accel_linear_estimate_b = np.dot(
            self._vehicle_model.rotItoB, (self._reference['acc'][0:3] - \
                                          np.dot(self._rotBtoI_dot, self._vehicle_model._vel[0:3]))) + \
                                          np.multiply(self._lambda[0:3], e_v_linear_b) + \
                                          self._sliding_int * np.multiply(np.square(self._lambda[0:3]) / 4, e_p_linear_b)
        self._accel_angular_estimate_b = np.dot(self._vehicle_model.rotItoB, (self._reference['acc'][3:6] -
                                                np.dot(self._rotBtoI_dot, self._vehicle_model._vel[3:6]))) + \
                                                np.multiply(self._lambda[3:6], e_v_angular_b) + \
                                                self._sliding_int * np.multiply(np.square(self._lambda[3:6]) / 4,
                                                                                e_p_angular_b)
        self._accel_estimate_b = np.hstack((self._accel_linear_estimate_b, self._accel_angular_estimate_b))
        acc = self._vehicle_model.to_SNAME(self._accel_estimate_b)
        self._f_eq = self._vehicle_model.compute_force(acc, use_sname=False)
        self._f_lin = - np.multiply(self._k, self._s_b)
        self._rho_total = self._adaptive_bounds * self._rho_adapt + self._constant_bound * self._rho_constant
        self._rho_adapt = self._rho_adapt + \
                          (self._adapt_slope[0] * np.abs(self._s_b) +
                          (self._adapt_slope[1] * np.abs(self._s_b) * np.abs(e_p_b) * np.abs(e_p_b)) +
                          (self._adapt_slope[2] * np.abs(self._s_b) * np.abs(e_v_b) * np.abs(e_v_b)) +
                           self._drift_prevent * (self._rho_0 - self._rho_adapt)) * dt
        self._f_robust = - np.multiply(self._rho_total, (2 / np.pi) * np.arctan(np.multiply(self._c, self._s_b)))
        self.F_tau = self._ctrl_eq * self._f_eq + self._ctrl_lin * self._f_lin + self._ctrl_robust * self._f_robust
        kp_x=0.2
        ki_x=0.1
        kd_x=0.1
        mu_x=0.1
        kp_y=0.2
        ki_y=0.1
        kd_y=0.1
        mu_y=0.1
        kp_z=1
        ki_z=.5
        kd_z=0.1
        mu_z=0.1
        kp_p=0.1
        ki_p=0.1
        kd_p=0.1
        mu_p=0.1
        kp_q=0.4
        ki_q=0.1
        kd_q=0.1
        mu_q=0.1
        kp_r=0.3
        ki_r=0.1
        kd_r=0.1
        mu_r=0.1
        m_bar_x=2042#100, 2642
        m_bar_y=3000#300, 3084
        m_bar_z=500#200, 5522
        m_bar_p=800
        m_bar_q=800
        m_bar_r=800
        delta=1.0001
        beta=1#.00001
        Ldelta_c_x=500
        Ldelta_c_y=500
        Ldelta_c_z=500
        Ldelta_c_p=100
        Ldelta_c_q=100
        Ldelta_c_r=100
        LDelta_d_x=100
        LDelta_d_y=100
        LDelta_d_z=100
        H_para_x=.6
        H_para_y=.7
        H_para_z=0.5
        H_para_p=0
        H_para_q=0
        H_para_r=0

        acc_angular=self._vehicle_model.to_SNAME(np.dot(self._vehicle_model.rotItoB, np.dot(self._rotBtoI_dot, self._vehicle_model._vel[3:6])))
        acc_cal_fromVel=(self._vehicle_model._vel[0:3]-self.vel_vehicle_prev)/dt
        acc_imu=self.imuAccLinear
        e_p_linear_b = self._errors['pos']
        e_v_linear_b = self._errors['vel'][0:3]
        e_p_angular_b = self.error_orientation_rpy
        e_v_angular_b = self._errors['vel'][3:6]
        e_p_b = np.hstack((e_p_linear_b, e_p_angular_b))
        e_v_b = np.hstack((e_v_linear_b, e_v_angular_b))
        acc_ref=self._reference['acc']

        acc_ref_diff=acc_ref-self.acc_ref_prev
        e_p_b_diff=e_p_b-self.e_p_b_prev
        e_v_b_diff=e_v_b-self.e_v_b_prev
        error_pose=self.error_pose_euler
        self._int_e += 0.5 * (error_pose - self._error_pose_prev) * self._dt
        s_x=kp_x*e_p_b[0]+ki_x*self._int_e[0]+kd_x*e_v_b[0]        
        s_y=kp_y*e_p_b[1]+ki_y*self._int_e[1]+kd_y*e_v_b[1]        
        s_z=kp_z*e_p_b[2]+ki_z*self._int_e[2]+kd_z*e_v_b[2] 
        s_p=kp_p*e_p_b[3]+ki_p*self._int_e[3]+kd_p*e_v_b[3]  
        s_q=kp_q*e_p_b[4]+ki_q*self._int_e[4]+kd_q*e_v_b[4]
        s_r=kp_r*e_p_b[5]+ki_r*self._int_e[5]+kd_r*e_v_b[5]  
        s_diff_x=s_x-self.s_x_prev
        s_diff_y=s_y-self.s_y_prev
        s_diff_z=s_z-self.s_z_prev
        S=np.hstack((s_x,s_y,s_z,s_p,s_q,s_r))
        rho_x=delta*(self.rho_x_prev+kd_x*LDelta_d_x+kd_x*Ldelta_c_x)/(m_bar_x-delta)
        rho_y=delta*(self.rho_y_prev+kd_y*LDelta_d_y+kd_y*Ldelta_c_y)/(m_bar_y-delta)
        rho_z=delta*(self.rho_z_prev+kd_z*LDelta_d_z+kd_z*Ldelta_c_z)/(m_bar_z-delta)
        rho_p=delta*(self.rho_p_prev+kd_p*LDelta_d_z+kd_p*Ldelta_c_p)/(m_bar_p-delta)
        rho_q=delta*(self.rho_q_prev+kd_q*LDelta_d_z+kd_q*Ldelta_c_q)/(m_bar_q-delta)
        rho_r=delta*(self.rho_r_prev+kd_r*LDelta_d_z+kd_r*Ldelta_c_r)/(m_bar_r-delta)
        v_x=acc_ref[0]+kp_x/kd_x*e_v_b[0]+ki_x/kd_x*e_p_b[0]+1/mu_x/kd_x*s_x+rho_x/kd_x*np.tanh(s_x/beta)
        v_y=acc_ref[1]+kp_y/kd_y*e_v_b[1]+ki_y/kd_y*e_p_b[1]+1/mu_y/kd_y*s_y+rho_y/kd_y*np.tanh(s_y/beta)
        v_z=acc_ref[2]+kp_z/kd_z*e_v_b[2]+ki_z/kd_z*e_p_b[2]+1/mu_z/kd_z*s_z+rho_z/kd_z*np.tanh(s_z/beta)
        v_p=acc_ref[3]+kp_p/kd_p*e_v_b[3]+ki_p/kd_p*e_p_b[3]+1/mu_p/kd_p*s_p+rho_p/kd_p*np.tanh(s_p/beta)
        v_q=acc_ref[4]+kp_q/kd_q*e_v_b[4]+ki_q/kd_q*e_p_b[4]+1/mu_q/kd_q*s_q+rho_q/kd_q*np.tanh(s_q/beta)
        v_r=acc_ref[5]+kp_r/kd_r*e_v_b[5]+ki_r/kd_r*e_p_b[5]+1/mu_r/kd_r*s_r+rho_r/kd_r*np.tanh(s_r/beta)


        self.h_hat_x= H_para_x*(self.F_x_prev-m_bar_x*acc_imu[0])
        self.h_hat_y= H_para_y*(self.F_y_prev-m_bar_y*acc_imu[1])
        self.h_hat_z= H_para_z*(self.F_z_prev-m_bar_z*acc_imu[2])
        self.h_hat_p= H_para_p*(self.F_p_prev-m_bar_p*acc_angular[0])
        self.h_hat_q= H_para_q*(self.F_q_prev-m_bar_q*acc_angular[1])
        self.h_hat_r= H_para_r*(self.F_r_prev-m_bar_r*acc_angular[2])


        H_hat=np.hstack((self.h_hat_x,self.h_hat_y,self.h_hat_z))
        F_x=m_bar_x*v_x+self.h_hat_x
        F_y=m_bar_y*v_y+self.h_hat_y
        F_z=m_bar_z*v_z+self.h_hat_z
        F_p=m_bar_p*v_p+self.h_hat_p 
        F_q=m_bar_q*v_q+self.h_hat_q    
        F_r=m_bar_r*v_r+self.h_hat_r 
        F=np.hstack((F_x,F_y,F_z,self.F_tau[3],self.F_tau[4],self.F_tau[5]))
        F_SNAME=self._vehicle_model.to_SNAME(F)
        self.acc = self._vehicle_model.compute_acc(
            F_SNAME, use_sname=False)
        self._error_pose_prev = error_pose
        self.acc_ref_prev=acc_ref
        self.e_p_b_prev=e_p_b
        self.e_v_b_prev=e_v_b
        self.s_x_prev=s_x
        self.s_y_prev=s_y
        self.s_z_prev=s_z
        self.rho_x_prev=rho_x  
        self.rho_y_prev=rho_y
        self.rho_z_prev=rho_z
        self.rho_p_prev=rho_p
        self.rho_q_prev=rho_q
        self.rho_r_prev=rho_r
        self.F_x_prev=F_x
        self.F_y_prev=F_y
        self.F_z_prev=F_z
        self.F_p_prev=F_p
        self.F_q_prev=F_q
        self.F_r_prev=F_r
        self.acc_angular_prev=acc_angular
        self.acc_imu_prev1=acc_imu
        self.acc_cal_fromVel_prev=acc_cal_fromVel
        self.vel_vehicle_prev=self._vehicle_model._vel[0:3]
        self._slidingSurface=S
        self._tau[0]=F[0]
        self._tau[1]=F[1]
        self._tau[2]=F[2]
        self._tau[3]=F_p
        self._tau[4]=F_q
        self._tau[5]=F_r






        #  Proposed control (design without delay) with delay in ref
    #def update_controller(self):
    #    if not self._is_init:
    #        return False
    #    t = rospy.Time.now().to_sec()
    #    dt = t - self._prev_t
    #    if self._prev_t < 0.0:
    #        dt = 0.05
    #    error_pose_ref_d=self.error_pose_euler_ref_d
    #    self._int += 0.5 * (error_pose_ref_d - self._error_pose_prev) * self._dt
    #    e_p_linear_b = self._errors_ref_d['pos']
    #    e_v_linear_b = self._errors_ref_d['vel'][0:3]
    #    e_p_angular_b = self.error_orientation_rpy_ref_d
    #    e_v_angular_b = self._errors_ref_d['vel'][3:6]
    #    e_p_b = np.hstack((e_p_linear_b, e_p_angular_b))
    #    e_v_b = np.hstack((e_v_linear_b, e_v_angular_b))
    #    acc_imu=self.imuAccLinear
    #    self._s_b = -e_v_b - np.multiply(self._lambda, e_p_b) \
    #                - self._sliding_int * np.multiply(np.square(self._lambda)/4, self._int)
    #    self._rotBtoI_dot = np.dot(cross_product_operator(self._vehicle_model._vel[3:6]), self._vehicle_model.rotBtoI)
    #    self._accel_linear_estimate_b = np.dot(
    #        self._vehicle_model.rotItoB, (self.acc_ref_prev7[0:3] - \
    #                                      np.dot(self._rotBtoI_dot, self._vehicle_model._vel[0:3]))) + \
    #                                      np.multiply(self._lambda[0:3], e_v_linear_b) + \
    #                                      self._sliding_int * np.multiply(np.square(self._lambda[0:3]) / 4, e_p_linear_b)
    #    self._accel_angular_estimate_b = np.dot(self._vehicle_model.rotItoB, (self.acc_ref_prev7[3:6] -
    #                                            np.dot(self._rotBtoI_dot, self._vehicle_model._vel[3:6]))) + \
    #                                            np.multiply(self._lambda[3:6], e_v_angular_b) + \
    #                                            self._sliding_int * np.multiply(np.square(self._lambda[3:6]) / 4,
    #                                                                            e_p_angular_b)
    #    self._accel_estimate_b = np.hstack((self._accel_linear_estimate_b, self._accel_angular_estimate_b))
    #    acc = self._vehicle_model.to_SNAME(self._accel_estimate_b)
    #    self._f_eq = self._vehicle_model.compute_force(acc, use_sname=False)
    #    self._f_lin = - np.multiply(self._k, self._s_b)
    #    self._rho_total = self._adaptive_bounds * self._rho_adapt + self._constant_bound * self._rho_constant
    #    self._rho_adapt = self._rho_adapt + \
    #                      (self._adapt_slope[0] * np.abs(self._s_b) +
    #                      (self._adapt_slope[1] * np.abs(self._s_b) * np.abs(e_p_b) * np.abs(e_p_b)) +
    #                      (self._adapt_slope[2] * np.abs(self._s_b) * np.abs(e_v_b) * np.abs(e_v_b)) +
    #                       self._drift_prevent * (self._rho_0 - self._rho_adapt)) * dt
    #    self._f_robust = - np.multiply(self._rho_total, (2 / np.pi) * np.arctan(np.multiply(self._c, self._s_b)))
    #    self.F_tau = self._ctrl_eq * self._f_eq + self._ctrl_lin * self._f_lin + self._ctrl_robust * self._f_robust
    #    kp=0.12
    #    ki=0.1
    #    kd=0.1
    #    mu=0.1
    #    m_bar_x=2642#100, 2642
    #    m_bar_y=3084#300, 3084
    #    m_bar_z=5522#200, 5522
    #    delta=1.0001
    #    beta=.00001#.00001
    #    Ldelta_c_x=8000
    #    Ldelta_c_y=8000
    #    Ldelta_c_z=8000
    #    LDelta_d_x=100
    #    LDelta_d_y=100
    #    LDelta_d_z=100
    #    H_para=1
    #    e_p_linear_b_ref_d = self._errors_ref_d['pos']
    #    e_v_linear_b_ref_d = self._errors_ref_d['vel'][0:3]
    #    e_p_angular_b_ref_d = self.error_orientation_rpy_ref_d
    #    e_v_angular_b_ref_d = self._errors_ref_d['vel'][3:6]
    #    e_p_b_ref_d = np.hstack((e_p_linear_b_ref_d, e_p_angular_b_ref_d))
    #    e_v_b_ref_d = np.hstack((e_v_linear_b_ref_d, e_v_angular_b_ref_d))
    #    acc_ref=self._vehicle_model.to_SNAME(self.acc_ref_prev7)
    #    acc_ref_diff=acc_ref-self.acc_ref_prev
    #    e_p_b_diff=e_p_b-self.e_p_b_prev
    #    e_v_b_diff=e_v_b-self.e_v_b_prev
    #    error_pose_ref_d=self.error_pose_euler_ref_d
    #    self._int_e += 0.5 * (error_pose_ref_d - self._error_pose_prev) * self._dt
    #    s_x=kp*e_p_b_ref_d[0]+ki*self._int_e[0]+kd*e_v_b_ref_d[0]        
    #    s_y=kp*e_p_b_ref_d[1]+ki*self._int_e[1]+kd*e_v_b_ref_d[1]        
    #    s_z=kp*e_p_b_ref_d[2]+ki*self._int_e[2]+kd*e_v_b_ref_d[2]        
    #    s_diff_x=s_x-self.s_x_prev
    #    s_diff_y=s_y-self.s_y_prev
    #    s_diff_z=s_z-self.s_z_prev
    #    S=np.hstack((s_x,s_y,s_z,0,0,0))
    #    #LDelta_d_x=np.abs(acc_ref_diff[0]+kp/kd*e_v_b_diff[0]+ki/kd*e_p_b_diff[0]+1/kd/mu*s_diff_x)
    #    #LDelta_d_y=np.abs(acc_ref_diff[1]+kp/kd*e_v_b_diff[1]+ki/kd*e_p_b_diff[1]+1/kd/mu*s_diff_y)
    #    #LDelta_d_z=np.abs(acc_ref_diff[2]+kp/kd*e_v_b_diff[2]+ki/kd*e_p_b_diff[2]+1/kd/mu*s_diff_z)
    #    rho_x=delta*(self.rho_x_prev+kd*LDelta_d_x+kd*Ldelta_c_x)/(m_bar_x-delta)
    #    rho_y=delta*(self.rho_y_prev+kd*LDelta_d_y+kd*Ldelta_c_y)/(m_bar_y-delta)
    #    rho_z=delta*(self.rho_z_prev+kd*LDelta_d_z+kd*Ldelta_c_z)/(m_bar_z-delta)
     #   v_x=acc_ref[0]+kp/kd*e_v_b_ref_d[0]+ki/kd*e_p_b_ref_d[0]+1/mu/kd*s_x+rho_x/kd*np.tanh(s_x/beta)
     #   v_y=acc_ref[1]+kp/kd*e_v_b_ref_d[1]+ki/kd*e_p_b_ref_d[1]+1/mu/kd*s_y+rho_y/kd*np.tanh(s_y/beta)
     #   v_z=acc_ref[2]+kp/kd*e_v_b_ref_d[2]+ki/kd*e_p_b_ref_d[2]+1/mu/kd*s_z+rho_x/kd*np.tanh(s_z/beta)
     #   self.h_hat_x=.6*(self.F_x_prev-m_bar_x*self.acc_imu_prev1[0])
     #   self.h_hat_y=.6*(self.F_y_prev-m_bar_y*self.acc_imu_prev1[1])
     #   self.h_hat_z=.2*(self.F_z_prev-m_bar_z*self.acc_imu_prev1[2])
     #   H_hat=np.hstack((self.h_hat_x,self.h_hat_y,self.h_hat_z))
     #   F_x=m_bar_x*v_x+self.h_hat_x
     #   F_y=m_bar_y*v_y+self.h_hat_y
     #   F_z=m_bar_z*v_z+self.h_hat_z   
     #   F=np.hstack((F_x,F_y,F_z,self.F_tau[3],self.F_tau[4],self.F_tau[5]))
     #   self.acc = self._vehicle_model.compute_acc(
     #       self._vehicle_model.to_SNAME(F), use_sname=False)
     #   self._error_pose_prev = error_pose_ref_d
     #   self.acc_ref_prev=acc_ref
     #   self.e_p_b_prev=e_p_b
     #   self.e_v_b_prev=e_v_b
     #   self.s_x_prev=s_x
     #   self.s_y_prev=s_y
     #   self.s_z_prev=s_z
     #   self.rho_x_prev=rho_x  
     #   self.rho_y_prev=rho_y
     #   self.rho_z_prev=rho_z
     #   self.F_x_prev=F_x
     #   self.F_y_prev=F_y
     #   self.F_z_prev=F_z
     #   self.acc_imu_prev1=acc_imu
    #	self._slidingSurface=S
    #    self._tau[0]=F[0]
    #    self._tau[1]=F[1]
    #    self._tau[2]=F[2]
    #    self._tau[3]=self.F_tau[3]
    #    self._tau[4]=self.F_tau[4]
    #    self._tau[5]=self.F_tau[5]




#        #  Proposed control (design with delay) with delay in ref
#    def update_controller(self):
#        if not self._is_init:
#            return False
#        t = rospy.Time.now().to_sec()
#        dt = t - self._prev_t
#        if self._prev_t < 0.0:
#            dt = 0.05
#        error_pose_ref_d=self.error_pose_euler_ref_d
#        self._int += 0.5 * (error_pose_ref_d - self._error_pose_prev) * self._dt
#        e_p_linear_b = self._errors_ref_d['pos']
#        e_v_linear_b = self._errors_ref_d['vel'][0:3]
#        e_p_angular_b = self.error_orientation_rpy_ref_d
#        e_v_angular_b = self._errors_ref_d['vel'][3:6]
#        e_p_b = np.hstack((e_p_linear_b, e_p_angular_b))
#        e_v_b = np.hstack((e_v_linear_b, e_v_angular_b))
#
#        self._s_b = -e_v_b - np.multiply(self._lambda, e_p_b) \
#                    - self._sliding_int * np.multiply(np.square(self._lambda)/4, self._int)
#        self._rotBtoI_dot = np.dot(cross_product_operator(self._vehicle_model._vel[3:6]), self._vehicle_model.rotBtoI)
#        self._accel_linear_estimate_b = np.dot(
#            self._vehicle_model.rotItoB, (self.acc_ref_prev7[0:3] - \
#                                          np.dot(self._rotBtoI_dot, self._vehicle_model._vel[0:3]))) + \
#                                          np.multiply(self._lambda[0:3], e_v_linear_b) + \
#                                          self._sliding_int * np.multiply(np.square(self._lambda[0:3]) / 4, e_p_linear_b)
#        self._accel_angular_estimate_b = np.dot(self._vehicle_model.rotItoB, (self.acc_ref_prev7[3:6] -
#                                                np.dot(self._rotBtoI_dot, self._vehicle_model._vel[3:6]))) + \
#                                                np.multiply(self._lambda[3:6], e_v_angular_b) + \
#                                                self._sliding_int * np.multiply(np.square(self._lambda[3:6]) / 4,
#                                                                                e_p_angular_b)
#        self._accel_estimate_b = np.hstack((self._accel_linear_estimate_b, self._accel_angular_estimate_b))
#        acc = self._vehicle_model.to_SNAME(self._accel_estimate_b)
#        self._f_eq = self._vehicle_model.compute_force(acc, use_sname=False)
#        self._f_lin = - np.multiply(self._k, self._s_b)
#        self._rho_total = self._adaptive_bounds * self._rho_adapt + self._constant_bound * self._rho_constant
##        self._rho_adapt = self._rho_adapt + \
 #                         (self._adapt_slope[0] * np.abs(self._s_b) +
 #                         (self._adapt_slope[1] * np.abs(self._s_b) * np.abs(e_p_b) * np.abs(e_p_b)) +
 #                         (self._adapt_slope[2] * np.abs(self._s_b) * np.abs(e_v_b) * np.abs(e_v_b)) +
 #                          self._drift_prevent * (self._rho_0 - self._rho_adapt)) * dt
 #       self._f_robust = - np.multiply(self._rho_total, (2 / np.pi) * np.arctan(np.multiply(self._c, self._s_b)))
 #       self.F_tau = self._ctrl_eq * self._f_eq + self._ctrl_lin * self._f_lin + self._ctrl_robust * self._f_robust
 #       kp_x=0.5
 #       ki_x=0.1
 #       kd_x=0.1
 #       mu_x=0.1

  #      kp_y=0.5
  #      ki_y=0.1
  #      kd_y=0.1
  #      mu_y=0.1

  #      kp_z=.5
  #      ki_z=.1
  #      kd_z=0.1
  #      mu_z=0.1

  #      kp_p=0.1
  #      ki_p=0.1
  #      kd_p=0.1
  #      mu_p=0.1

  #      kp_q=0.4
  #      ki_q=0.1
  #      kd_q=0.1
  #      mu_q=0.1

   #     kp_r=0.3
   #     ki_r=0.1
   #     kd_r=0.1
   #     mu_r=0.1
   #     m_bar_x=2042#100, 2642
   #     m_bar_y=3000#300, 3084
   #     m_bar_z=500#200, 5522
   #     m_bar_p=800#725
   #     m_bar_q=800#1094
   #     m_bar_r=300#791
   #     delta=1.0001
   #     beta=1#.00001

   #     Ldelta_c_x=500
   #     Ldelta_c_y=500
   #     Ldelta_c_z=500
   #     Ldelta_c_p=100
   #     Ldelta_c_q=100
   #     Ldelta_c_r=100
   #     LDelta_d_x=100
   #     LDelta_d_y=100
   #     LDelta_d_z=100

   #     H_para_x=.6
   #     H_para_y=.7
   #     H_para_z=.5
   #     H_para_p=0
   #     H_para_q=0
   #     H_para_r=0
   #     delta_z=0.001

    #    acc_imu=self.imuAccLinear


    #    acc_angular=self._vehicle_model.to_SNAME(np.dot(self._vehicle_model.rotItoB, np.dot(self._rotBtoI_dot, self._vehicle_model._vel[3:6])))     
    #    e_p_linear_b_ref_d = self._errors_ref_d['pos']
    #    e_v_linear_b_ref_d = self._errors_ref_d['vel'][0:3]
    #    e_p_angular_b_ref_d = self.error_orientation_rpy_ref_d
    #    e_v_angular_b_ref_d = self._errors_ref_d['vel'][3:6]
    #    e_p_b_ref_d = np.hstack((e_p_linear_b_ref_d, e_p_angular_b_ref_d))
    #    e_v_b_ref_d = np.hstack((e_v_linear_b_ref_d, e_v_angular_b_ref_d))
    #    #acc_ref=self._vehicle_model.to_SNAME(self.acc_ref_prev7)
    #    if t>40 and t<=90:
    #        acc_ref=self.acc_ref_prev7
    #    else:
    #        acc_ref=self.acc_ref_prev1
    #    acc_ref_diff=acc_ref-self.acc_ref_prev
    #    e_p_b_diff=e_p_b-self.e_p_b_prev
    #    e_v_b_diff=e_v_b-self.e_v_b_prev
    #    error_pose_ref_d=self.error_pose_euler_ref_d
    #    self._int_e += 0.5 * (error_pose_ref_d - self._error_pose_prev) * self._dt
    #    s_x=kp_x*e_p_b_ref_d[0]+ki_x*self._int_e[0]+kd_x*e_v_b_ref_d[0]        
    #    s_y=kp_y*e_p_b_ref_d[1]+ki_y*self._int_e[1]+kd_y*e_v_b_ref_d[1]        
    #    s_z=kp_z*e_p_b_ref_d[2]+ki_z*self._int_e[2]+kd_z*e_v_b_ref_d[2] 
    #    s_p=kp_p*e_p_b[3]+ki_p*self._int[3]+kd_p*e_v_b[3]  
    #    s_q=kp_q*e_p_b[4]+ki_q*self._int[4]+kd_q*e_v_b[4]
    #    s_r=kp_r*e_p_b[5]+ki_r*self._int[5]+kd_r*e_v_b[5]       
    #    s_diff_x=s_x-self.s_x_prev
    #    s_diff_y=s_y-self.s_y_prev
    #    s_diff_z=s_z-self.s_z_prev
    #    S=np.hstack((s_x,s_y,s_z,0,0,0))
    #    rho_x=(kd_x*delta+m_bar_x*delta_z)/(kd_x*m_bar_x-kd_x*delta-m_bar_x*delta_z)*(self.rho_x_prev+kd_x*LDelta_d_x+kd_x*Ldelta_c_x)
    #    rho_y=(kd_y*delta+m_bar_y*delta_z)/(kd_y*m_bar_y-kd_y*delta-m_bar_y*delta_z)*(self.rho_y_prev+kd_y*LDelta_d_y+kd_y*Ldelta_c_y)
    #    rho_z=(kd_z*delta+m_bar_z*delta_z)/(kd_z*m_bar_z-kd_z*delta-m_bar_z*delta_z)*(self.rho_z_prev+kd_z*LDelta_d_z+kd_z*Ldelta_c_z)

    #    rho_p=(kd_p*delta+m_bar_p*delta_z)/(kd_p*m_bar_p-kd_p*delta-m_bar_p*delta_z)*(self.rho_p_prev+kd_p*LDelta_d_x+kd_p*Ldelta_c_p)
    #    rho_q=(kd_q*delta+m_bar_q*delta_z)/(kd_q*m_bar_q-kd_q*delta-m_bar_q*delta_z)*(self.rho_q_prev+kd_q*LDelta_d_y+kd_q*Ldelta_c_q)
    #    rho_r=(kd_r*delta+m_bar_r*delta_z)/(kd_r*m_bar_r-kd_r*delta-m_bar_r*delta_z)*(self.rho_r_prev+kd_r*LDelta_d_z+kd_r*Ldelta_c_r)

    #    v_x=acc_ref[0]+1/mu_x/kd_x*s_x+rho_x/kd_x*np.tanh(s_x/beta)
    #    v_y=acc_ref[1]+1/mu_y/kd_y*s_y+rho_y/kd_y*np.tanh(s_y/beta)
    #    v_z=acc_ref[2]+1/mu_z/kd_z*s_z+rho_z/kd_z*np.tanh(s_z/beta)

    #    v_p=acc_ref[3]+1/mu_p/kd_p*s_p+rho_p/kd_p*np.tanh(s_p/beta)
    #    v_q=acc_ref[4]+1/mu_q/kd_q*s_q+rho_q/kd_q*np.tanh(s_q/beta)
    #    v_r=acc_ref[5]+1/mu_r/kd_r*s_r+rho_r/kd_r*np.tanh(s_r/beta)


    #    self.h_hat_x=H_para_x*(self.F_x_prev-m_bar_x*acc_imu[0])
    #    self.h_hat_y=H_para_y*(self.F_y_prev-m_bar_y*acc_imu[1])
    #    self.h_hat_z=H_para_z*(self.F_z_prev-m_bar_z*acc_imu[2])
    #    self.h_hat_p=H_para_p*(self.F_p_prev-m_bar_p*acc_angular[0])
    #    self.h_hat_q=H_para_q*(self.F_q_prev-m_bar_q*acc_angular[1])
    #    self.h_hat_r=H_para_r*(self.F_r_prev-m_bar_r*acc_angular[2])
    #    H_hat=np.hstack((self.h_hat_x,self.h_hat_y,self.h_hat_z))
    #    F_x=m_bar_x*v_x+self.h_hat_x
    #    F_y=m_bar_y*v_y+self.h_hat_y
    #    F_z=m_bar_z*v_z+self.h_hat_z   
    #    F_p=m_bar_p*v_p+self.h_hat_p 
    #    F_q=m_bar_q*v_q+self.h_hat_q    
    #    F_r=m_bar_r*v_r+self.h_hat_r 
    #    F=np.hstack((F_x,F_y,F_z,self.F_tau[3],self.F_tau[4],self.F_tau[5]))
    #    self.acc = self._vehicle_model.compute_acc(
    #        self._vehicle_model.to_SNAME(F), use_sname=False)
    #    self._error_pose_prev = error_pose_ref_d
    #    self.acc_ref_prev=acc_ref
    #    self.e_p_b_prev=e_p_b
    #    self.e_v_b_prev=e_v_b
    #    self.s_x_prev=s_x
    #    self.s_y_prev=s_y
    #    self.s_z_prev=s_z
    #    self.rho_x_prev=rho_x  
    #    self.rho_y_prev=rho_y
    #    self.rho_z_prev=rho_z
    #    self.rho_p_prev=rho_p
    #    self.rho_q_prev=rho_q
    #    self.rho_r_prev=rho_r
    #    self.F_x_prev=F_x
    #    self.F_y_prev=F_y
    #    self.F_z_prev=F_z
    #    self.F_p_prev=F_p
    #    self.F_q_prev=F_q
    #    self.F_r_prev=F_r
   #	self._slidingSurface=S
   #     self.acc_imu_prev1=acc_imu
    #    self.acc_angular_prev=acc_angular
    #    self._tau[0]=F[0]
    #    self._tau[1]=F[1]
    #    self._tau[2]=F[2]
    #    self._tau[3]=F_p
    #    self._tau[4]=F_q
    #    self._tau[5]=F_r










        #  Proposed control (design without delay) with delay in vehicle position measurement
   # def update_controller(self):
   #     if not self._is_init:
   #         return False
   #     t = rospy.Time.now().to_sec()
   #     dt = t - self._prev_t
   #     if self._prev_t < 0.0:
   #         dt = 0.05
   #     self._int += 0.5 * (self.error_pose_euler_veh_d - self._error_pose) * self._dt
   #     self._error_pose = self.error_pose_euler_veh_d
   #     e_p_linear_b = self._errors_veh_d['pos']
   #     e_v_linear_b = self._errors_veh_d['vel'][0:3]
   #     e_p_angular_b = self.error_orientation_rpy_veh_d
   #     e_v_angular_b = self._errors_veh_d['vel'][3:6]
   #     e_p_b = np.hstack((e_p_linear_b, e_p_angular_b))
   #     e_v_b = np.hstack((e_v_linear_b, e_v_angular_b))
   #     acc_imu=self.imuAccLinear
   #     acc_imu_1=self.acc_imu_prev1
   #     acc_imu_2=self.acc_imu_prev2
   #     acc_imu_3=self.acc_imu_prev3
   #     acc_imu_4=self.acc_imu_prev4
   #     acc_imu_5=self.acc_imu_prev5
   #     acc_imu_6=self.acc_imu_prev6
   #     acc_imu_7=self.acc_imu_prev7
   #     self._s_b = -e_v_b - np.multiply(self._lambda, e_p_b) \
   #                 - self._sliding_int * np.multiply(np.square(self._lambda)/4, self._int)
   #     self._rotBtoI_dot = np.dot(cross_product_operator(self.vel_veh_prev3[3:6]), self._vehicle_model.rotBtoI)
   #     self._accel_linear_estimate_b = np.dot(
   #         self._vehicle_model.rotItoB, (self._reference['acc'][0:3] - \
   #                                       np.dot(self._rotBtoI_dot, self.vel_veh_prev3[0:3]))) + \
   #                                       np.multiply(self._lambda[0:3], e_v_linear_b) + \
   #                                       self._sliding_int * np.multiply(np.square(self._lambda[0:3]) / 4, e_p_linear_b)
   #     self._accel_angular_estimate_b = np.dot(self._vehicle_model.rotItoB, (self._reference['acc'][3:6] -
   #                                             np.dot(self._rotBtoI_dot, self.vel_veh_prev3[3:6]))) + \
   #                                             np.multiply(self._lambda[3:6], e_v_angular_b) + \
   #                                             self._sliding_int * np.multiply(np.square(self._lambda[3:6]) / 4,
   #                                                                             e_p_angular_b)
   #     self._accel_estimate_b = np.hstack((self._accel_linear_estimate_b, self._accel_angular_estimate_b))
   #     acc = self._vehicle_model.to_SNAME(self._accel_estimate_b)
   #     self._f_eq = self._vehicle_model.compute_force(acc, use_sname=False)
   #     self._f_lin = - np.multiply(self._k, self._s_b)
   #     self._rho_total = self._adaptive_bounds * self._rho_adapt + self._constant_bound * self._rho_constant
   #     self._rho_adapt = self._rho_adapt + \
   #                       (self._adapt_slope[0] * np.abs(self._s_b) +
   #                       (self._adapt_slope[1] * np.abs(self._s_b) * np.abs(e_p_b) * np.abs(e_p_b)) +
   #                       (self._adapt_slope[2] * np.abs(self._s_b) * np.abs(e_v_b) * np.abs(e_v_b)) +
   #                        self._drift_prevent * (self._rho_0 - self._rho_adapt)) * dt
   #     self._f_robust = - np.multiply(self._rho_total, (2 / np.pi) * np.arctan(np.multiply(self._c, self._s_b)))
   #     self.F_tau = self._ctrl_eq * self._f_eq + self._ctrl_lin * self._f_lin + self._ctrl_robust * self._f_robust
   #     kp=0.12
   #     ki=0.1
   #     kd=0.1
   #     mu=0.1
   #     m_bar_x=2642#100, 2642
   #     m_bar_y=3084#300, 3084
   #     m_bar_z=5522#200, 5522
   #     delta=1.0001
   #     beta=.00001#.00001
   #     Ldelta_c_x=8000
   #     Ldelta_c_y=8000
   #     Ldelta_c_z=8000
   #     LDelta_d_x=100
   #     LDelta_d_y=100
   #     LDelta_d_z=100
   #     H_para=1
   #     acc_ref=self._vehicle_model.to_SNAME(self._reference['acc'])
   #     acc_ref_diff=acc_ref-self.acc_ref_prev
   #     e_p_b_diff=e_p_b-self.e_p_b_prev
   #     e_v_b_diff=e_v_b-self.e_v_b_prev
   #     error_pose=self.error_pose_euler_veh_d
   #     self._int_e += 0.5 * (error_pose - self._error_pose_prev) * self._dt
   #     s_x=kp*e_p_b[0]+ki*self._int_e[0]+kd*e_v_b[0]        
   #     s_y=kp*e_p_b[1]+ki*self._int_e[1]+kd*e_v_b[1]        
   #     s_z=kp*e_p_b[2]+ki*self._int_e[2]+kd*e_v_b[2]        
   #     s_diff_x=s_x-self.s_x_prev
   #     s_diff_y=s_y-self.s_y_prev
   #     s_diff_z=s_z-self.s_z_prev
   #     S=np.hstack((s_x,s_y,s_z,0,0,0))
   #     #LDelta_d_x=np.abs(acc_ref_diff[0]+kp/kd*e_v_b_diff[0]+ki/kd*e_p_b_diff[0]+1/kd/mu*s_diff_x)
   #     #LDelta_d_y=np.abs(acc_ref_diff[1]+kp/kd*e_v_b_diff[1]+ki/kd*e_p_b_diff[1]+1/kd/mu*s_diff_y)
   #     #LDelta_d_z=np.abs(acc_ref_diff[2]+kp/kd*e_v_b_diff[2]+ki/kd*e_p_b_diff[2]+1/kd/mu*s_diff_z)
   #     rho_x=delta*(self.rho_x_prev+kd*LDelta_d_x+kd*Ldelta_c_x)/(m_bar_x-delta)
   #     rho_y=delta*(self.rho_y_prev+kd*LDelta_d_y+kd*Ldelta_c_y)/(m_bar_y-delta)
   #     rho_z=delta*(self.rho_z_prev+kd*LDelta_d_z+kd*Ldelta_c_z)/(m_bar_z-delta)
   #     v_x=acc_ref[0]+kp/kd*e_v_b[0]+ki/kd*e_p_b[0]+1/mu/kd*s_x+rho_x/kd*np.tanh(s_x/beta)
   #     v_y=acc_ref[1]+kp/kd*e_v_b[1]+ki/kd*e_p_b[1]+1/mu/kd*s_y+rho_y/kd*np.tanh(s_y/beta)
   #     v_z=acc_ref[2]+kp/kd*e_v_b[2]+ki/kd*e_p_b[2]+1/mu/kd*s_z+rho_x/kd*np.tanh(s_z/beta)
   #     self.h_hat_x=.4*self.F_x_prev-m_bar_x*self.acc_imu_prev4[0]
   #     self.h_hat_y=.4*self.F_y_prev-m_bar_y*self.acc_imu_prev4[1]
   #     self.h_hat_z=.2*self.F_z_prev-m_bar_z*self.acc_imu_prev4[2]
   #     H_hat=np.hstack((self.h_hat_x,self.h_hat_y,self.h_hat_z))
   #     F_x=m_bar_x*v_x+self.h_hat_x
   #     F_y=m_bar_y*v_y+self.h_hat_y
   #     F_z=m_bar_z*v_z+self.h_hat_z   
   #     F=np.hstack((F_x,F_y,F_z,self.F_tau[3],self.F_tau[4],self.F_tau[5]))
   #     self.acc = self._vehicle_model.compute_acc(
   #        self._vehicle_model.to_SNAME(F), use_sname=False)
   #     acc_veh_esti_prev1=self.acc
   #     acc_veh_esti_prev2=self.acc_prev1
   #     self.acc_prev1=acc_veh_esti_prev1
   #     self.acc_prev2=acc_veh_esti_prev2
   #     self._error_pose_prev = error_pose
   #     self.acc_ref_prev=acc_ref
   #     self.e_p_b_prev=e_p_b
   #     self.e_v_b_prev=e_v_b
   #     self.s_x_prev=s_x
   #     self.s_y_prev=s_y
   #     self.s_z_prev=s_z
   #     self.rho_x_prev=rho_x  
   #     self.rho_y_prev=rho_y
   #     self.rho_z_prev=rho_z
   #     self.F_x_prev=F_x
   #     self.F_y_prev=F_y
   #     self.F_z_prev=F_z
  # 	self._slidingSurface=S
  #      self.acc_imu_prev1=acc_imu
  #      self.acc_imu_prev2=acc_imu_1
  #      self.acc_imu_prev3=acc_imu_2
   #     self.acc_imu_prev4=acc_imu_3
   #     self.acc_imu_prev5=acc_imu_4
   #     self.acc_imu_prev6=acc_imu_5
   #     self.acc_imu_prev7=acc_imu_6
   #     self.acc_imu_prev8=acc_imu_7
   #     self._tau[0]=F[0]
   #     self._tau[1]=F[1]
   #     self._tau[2]=F[2]
   #     self._tau[3]=self.F_tau[3]
   #     self._tau[4]=self.F_tau[4]
   #     self._tau[5]=self.F_tau[5]

        #  Proposed control (design with delay) with delay in vehicle position measurement
 #   def update_controller(self):
 #       if not self._is_init:
 #           return False
 #       t = rospy.Time.now().to_sec()
 #       dt = t - self._prev_t
 #       if self._prev_t < 0.0:
 #           dt = 0.05
 #       self._int += 0.5 * (self.error_pose_euler_veh_d - self._error_pose) * self._dt
 #       self._error_pose = self.error_pose_euler_veh_d
 #       e_p_linear_b = self._errors_veh_d['pos']
 #       e_v_linear_b = self._errors_veh_d['vel'][0:3]
 #       e_p_angular_b = self.error_orientation_rpy_veh_d
 #       e_v_angular_b = self._errors_veh_d['vel'][3:6]
 #       e_p_b = np.hstack((e_p_linear_b, e_p_angular_b))
 #       e_v_b = np.hstack((e_v_linear_b, e_v_angular_b))
 #       acc_imu=self.imuAccLinear
 #       acc_imu_1=self.acc_imu_prev1
 #       acc_imu_2=self.acc_imu_prev2
 #       acc_imu_3=self.acc_imu_prev3
 #       acc_imu_4=self.acc_imu_prev4
 #       acc_imu_5=self.acc_imu_prev5
 #       acc_imu_6=self.acc_imu_prev6
 #       acc_imu_7=self.acc_imu_prev7
 #       self._s_b = -e_v_b - np.multiply(self._lambda, e_p_b) \
 #                   - self._sliding_int * np.multiply(np.square(self._lambda)/4, self._int)
 #       self._rotBtoI_dot = np.dot(cross_product_operator(self.vel_veh_prev5[3:6]), self._vehicle_model.rotBtoI)
 #       self._accel_linear_estimate_b = np.dot(
 #           self._vehicle_model.rotItoB, (self._reference['acc'][0:3] - \
 #                                         np.dot(self._rotBtoI_dot, self.vel_veh_prev5[0:3]))) + \
 #                                         np.multiply(self._lambda[0:3], e_v_linear_b) + \
 #                                         self._sliding_int * np.multiply(np.square(self._lambda[0:3]) / 4, e_p_linear_b)
 #       self._accel_angular_estimate_b = np.dot(self._vehicle_model.rotItoB, (self._reference['acc'][3:6] -
 #                                               np.dot(self._rotBtoI_dot, self.vel_veh_prev5[3:6]))) + \
 #                                               np.multiply(self._lambda[3:6], e_v_angular_b) + \
 #                                               self._sliding_int * np.multiply(np.square(self._lambda[3:6]) / 4,
 #                                                                               e_p_angular_b)
 #       self._accel_estimate_b = np.hstack((self._accel_linear_estimate_b, self._accel_angular_estimate_b))
 #       acc = self._vehicle_model.to_SNAME(self._accel_estimate_b)
 #       self._f_eq = self._vehicle_model.compute_force(acc, use_sname=False)
 #       self._f_lin = - np.multiply(self._k, self._s_b)
 #       self._rho_total = self._adaptive_bounds * self._rho_adapt + self._constant_bound * self._rho_constant
 #       self._rho_adapt = self._rho_adapt + \
 #                         (self._adapt_slope[0] * np.abs(self._s_b) +
 #                         (self._adapt_slope[1] * np.abs(self._s_b) * np.abs(e_p_b) * np.abs(e_p_b)) +
 #                         (self._adapt_slope[2] * np.abs(self._s_b) * np.abs(e_v_b) * np.abs(e_v_b)) +
 #                          self._drift_prevent * (self._rho_0 - self._rho_adapt)) * dt
 #       self._f_robust = - np.multiply(self._rho_total, (2 / np.pi) * np.arctan(np.multiply(self._c, self._s_b)))
 #       self.F_tau = self._ctrl_eq * self._f_eq + self._ctrl_lin * self._f_lin + self._ctrl_robust * self._f_robust
 #       kp_x=0.1
 #       ki_x=0.1
 #       kd_x=0.1
 #       mu_x=0.1
#
#        kp_y=0.1
#        ki_y=0.1
#        kd_y=0.1
#        mu_y=0.1
#
#        kp_z=.1
#        ki_z=.1
#        kd_z=0.1
#        mu_z=0.1
#
#        kp_p=0.1
#        ki_p=0.1
#        kd_p=0.1
#        mu_p=0.1
#
#        kp_q=0.1
#        ki_q=0.1
#        kd_q=0.1
#        mu_q=0.1
#
#        kp_r=0.1
#        ki_r=0.1
#        kd_r=0.1
#        mu_r=0.1
#        m_bar_x=2042#100, 2642
#        m_bar_y=3000#300, 3084
#        m_bar_z=3500#200, 5522
#        m_bar_p=300#725
#        m_bar_q=300#1094
#        m_bar_r=300#791
#        delta=1.0001
#        beta=5#.00001
#        Ldelta_c_x=1000
#        Ldelta_c_y=1000
#        Ldelta_c_z=1000
#        Ldelta_c_p=500
#        Ldelta_c_q=500
#        Ldelta_c_r=500
#        LDelta_d_x=100
#        LDelta_d_y=100
#        LDelta_d_z=100
#        H_para_x=.2
#        H_para_y=.2
#        H_para_z=.2
#        H_para_p=0
#        H_para_q=0
#        H_para_r=0
#        delta_z=.003
#        acc_angular=self._vehicle_model.to_SNAME(np.dot(self._vehicle_model.rotItoB, np.dot(self._rotBtoI_dot, self._vehicle_model._vel[3:6]))) 
#        acc_angular_1=self.acc_angular_prev1
#        acc_angular_2=self.acc_angular_prev2
#        acc_angular_3=self.acc_angular_prev3
#        acc_angular_4=self.acc_angular_prev4 
#        acc_angular_5=self.acc_angular_prev5
#        acc_angular_6=self.acc_angular_prev6 
#        acc_angular_7=self.acc_angular_prev7 
#        #acc_ref=self._vehicle_model.to_SNAME(self._reference['acc'])
#        acc_ref=self._reference['acc']
#        acc_ref_diff=acc_ref-self.acc_ref_prev
#        e_p_b_diff=e_p_b-self.e_p_b_prev
#        e_v_b_diff=e_v_b-self.e_v_b_prev
#        error_pose=self.error_pose_euler_veh_d
#        self._int_e += 0.5 * (error_pose - self._error_pose_prev) * self._dt
#        s_x=kp_x*e_p_b[0]+ki_x*self._int_e[0]+kd_x*e_v_b[0]        
#        s_y=kp_y*e_p_b[1]+ki_y*self._int_e[1]+kd_y*e_v_b[1]        
#        s_z=kp_z*e_p_b[2]+ki_z*self._int_e[2]+kd_z*e_v_b[2] 
#        s_p=kp_p*e_p_b[3]+ki_p*self._int_e[3]+kd_p*e_v_b[3]  
#        s_q=kp_q*e_p_b[4]+ki_q*self._int_e[4]+kd_q*e_v_b[4]
#        s_r=kp_r*e_p_b[5]+ki_r*self._int_e[5]+kd_r*e_v_b[5]         
#        s_diff_x=s_x-self.s_x_prev
#        s_diff_y=s_y-self.s_y_prev
#        s_diff_z=s_z-self.s_z_prev
#        S=np.hstack((s_x,s_y,s_z,0,0,0))
#        #LDelta_d_x=np.abs(acc_ref_diff[0]+kp/kd*e_v_b_diff[0]+ki/kd*e_p_b_diff[0]+1/kd/mu*s_diff_x)
#        #LDelta_d_y=np.abs(acc_ref_diff[1]+kp/kd*e_v_b_diff[1]+ki/kd*e_p_b_diff[1]+1/kd/mu*s_diff_y)
#        #LDelta_d_z=np.abs(acc_ref_diff[2]+kp/kd*e_v_b_diff[2]+ki/kd*e_p_b_diff[2]+1/kd/mu*s_diff_z)
#        rho_x=(kd_x*delta+m_bar_x*delta_z)/(kd_x*m_bar_x-kd_x*delta-m_bar_x*delta_z)*(self.rho_x_prev+kd_x*LDelta_d_x+kd_x*Ldelta_c_x)
#        rho_y=(kd_y*delta+m_bar_y*delta_z)/(kd_y*m_bar_y-kd_y*delta-m_bar_y*delta_z)*(self.rho_y_prev+kd_y*LDelta_d_y+kd_y*Ldelta_c_y)
#        rho_z=(kd_z*delta+m_bar_y*delta_z)/(kd_z*m_bar_y-kd_z*delta-m_bar_z*delta_z)*(self.rho_z_prev+kd_z*LDelta_d_z+kd_z*Ldelta_c_z)
#        rho_p=(kd_p*delta+m_bar_p*delta_z)/(kd_p*m_bar_p-kd_p*delta-m_bar_p*delta_z)*(self.rho_p_prev+kd_p*LDelta_d_x+kd_p*Ldelta_c_p)
#        rho_q=(kd_q*delta+m_bar_q*delta_z)/(kd_q*m_bar_q-kd_q*delta-m_bar_q*delta_z)*(self.rho_q_prev+kd_q*LDelta_d_y+kd_q*Ldelta_c_q)
#        rho_r=(kd_r*delta+m_bar_r*delta_z)/(kd_r*m_bar_r-kd_r*delta-m_bar_r*delta_z)*(self.rho_r_prev+kd_r*LDelta_d_z+kd_r*Ldelta_c_r)
#        v_x=acc_ref[0]+1/mu_x/kd_x*s_x+rho_x/kd_x*np.tanh(s_x/beta)
#        v_y=acc_ref[1]+1/mu_y/kd_y*s_y+rho_y/kd_y*np.tanh(s_y/beta)
#        v_z=acc_ref[2]+1/mu_z/kd_z*s_z+rho_x/kd_z*np.tanh(s_z/beta)
#        v_p=acc_ref[3]+1/mu_p/kd_p*s_p+rho_p/kd_p*np.tanh(s_p/beta)
#        v_q=acc_ref[4]+1/mu_q/kd_q*s_q+rho_q/kd_q*np.tanh(s_q/beta)
#        v_r=acc_ref[5]+1/mu_r/kd_r*s_r+rho_r/kd_r*np.tanh(s_r/beta)
#        F_x_prev_1=self.F_x_prev
#        F_x_prev_2=self.F_x_prev1
#        F_x_prev_3=self.F_x_prev2
#        F_y_prev_1=self.F_y_prev
#        F_y_prev_2=self.F_y_prev1
#        F_y_prev_3=self.F_y_prev2
#        F_z_prev_1=self.F_z_prev
#        F_z_prev_2=self.F_z_prev1
#        F_z_prev_3=self.F_z_prev2
#        if t > 40 and t<90:            
#            self.h_hat_x=H_para_x*self.F_x_prev-m_bar_x*self.acc_imu_prev3[0]
#            self.h_hat_y=H_para_y*self.F_y_prev-m_bar_y*self.acc_imu_prev3[1]
#            self.h_hat_z=H_para_z*self.F_z_prev-m_bar_z*self.acc_imu_prev3[2]
#            self.h_hat_p=H_para_p*(self.F_p_prev-m_bar_p*self.acc_angular_prev3[0])
#            self.h_hat_q=H_para_q*(self.F_q_prev-m_bar_q*self.acc_angular_prev3[1])
#            self.h_hat_r=H_para_r*(self.F_r_prev-m_bar_r*self.acc_angular_prev3[2])
#        else:
#            self.h_hat_x=H_para_x*self.F_x_prev-m_bar_x*self.acc_imu_prev2[0]
#            self.h_hat_y=H_para_y*self.F_y_prev-m_bar_y*self.acc_imu_prev2[1]
#            self.h_hat_z=H_para_z*self.F_z_prev-m_bar_z*self.acc_imu_prev2[2]
#            self.h_hat_p=H_para_p*(self.F_p_prev-m_bar_p*self.acc_angular_prev2[0])
#            self.h_hat_q=H_para_q*(self.F_q_prev-m_bar_q*self.acc_angular_prev2[1])
#            self.h_hat_r=H_para_r*(self.F_r_prev-m_bar_r*self.acc_angular_prev2[2])##
#
#        F_x=m_bar_x*v_x+self.h_hat_x
#        F_y=m_bar_y*v_y+self.h_hat_y
#        F_z=m_bar_z*v_z+self.h_hat_z
#        F_p=m_bar_p*v_p+self.h_hat_p 
#        F_q=m_bar_q*v_q+self.h_hat_q    
#        F_r=m_bar_r*v_r+self.h_hat_r    
#        F=np.hstack((F_x,F_y,F_z,self.F_tau[3],self.F_tau[4],self.F_tau[5]))
#        self.acc = self._vehicle_model.compute_acc(
#            self._vehicle_model.to_SNAME(F), use_sname=False)
#        acc_veh_esti_prev1=self.acc
#        acc_veh_esti_prev2=self.acc_prev1
#        self.acc_prev1=acc_veh_esti_prev1
#        self.acc_prev2=acc_veh_esti_prev2
#        self._error_pose_prev = error_pose
#        self.acc_ref_prev=acc_ref
#        self.e_p_b_prev=e_p_b
#        self.e_v_b_prev=e_v_b
#        self.s_x_prev=s_x
#        self.s_y_prev=s_y
#        self.s_z_prev=s_z
#        self.rho_x_prev=rho_x  
#        self.rho_y_prev=rho_y
#        self.rho_z_prev=rho_z
#        self.rho_p_prev=rho_p
#        self.rho_q_prev=rho_q
#        self.rho_r_prev=rho_r
#        self.F_x_prev=F_x
#        self.F_y_prev=F_y
#        self.F_z_prev=F_z
#        self.F_p_prev=F_p
#        self.F_q_prev=F_q
#        self.F_r_prev=F_r
#        self.F_x_prev1=F_x_prev_1
#        self.F_y_prev1=F_y_prev_1
#        self.F_z_prev1=F_z_prev_1
#        self.F_x_prev2=F_x_prev_2
#        self.F_y_prev2=F_y_prev_2
#        self.F_z_prev2=F_z_prev_2

#    	self._slidingSurface=S
#        self.acc_imu_prev1=acc_imu
#        self.acc_imu_prev2=acc_imu_1
#        self.acc_imu_prev3=acc_imu_2
#        self.acc_imu_prev4=acc_imu_3
#        self.acc_imu_prev5=acc_imu_4
#        self.acc_imu_prev6=acc_imu_5
#        self.acc_imu_prev7=acc_imu_6
#        self.acc_imu_prev8=acc_imu_7
#        self.acc_angular_prev1=acc_angular
#        self.acc_angular_prev2=acc_angular_1
#        self.acc_angular_prev3=acc_angular_2
#        self.acc_angular_prev4=acc_angular_3
#        self.acc_angular_prev5=acc_angular_4
#        self.acc_angular_prev6=acc_angular_5
#        self.acc_angular_prev7=acc_angular_6
#        self.acc_angular_prev8=acc_angular_7

#        self._tau[0]=F_x
#        self._tau[1]=F_y
#        self._tau[2]=F_z
#        self._tau[3]=F_p
#        self._tau[4]=F_q
#        self._tau[5]=F_r

       # self._tau[0]=self.F_tau[0]
       # self._tau[1]=self.F_tau[1]
       # self._tau[2]=self.F_tau[2]
       # self._tau[3]=self.F_tau[3]
       # self._tau[4]=self.F_tau[4]
       # self._tau[5]=self.F_tau[5]




	self._vehi=self._vehicle_model._Ma[5,5]
	#self._generalForce=self._vehicle_model.f
	self._restoring=self._vehicle_model._g
        self._MPara=self._vehicle_model._Mtotal
        self._CPara=self._vehicle_model._C
        self._DPara=self._vehicle_model._D
        self._velocity=self._vehicle_model._vel
        self._dt_=self._vehicle_model.quat[1]
        self._dt_1=self._vehicle_model.quat[2]

        self.pub_dt(self._dt_)
        self.pub_dt1(self._dt_1)
        self.publish_control_wrench(self._tau)
	self.publish_slidingSurface(self._slidingSurface)
	#self.publish_restoring(F_SNAME)
	self.publish_vehiclePara(self._vehi)
	self.publish_MPara(self._MPara)
	self.publish_CPara(self._CPara)
	self.publish_DPara(self._DPara)
	#self.publish_vel(acc_cal_fromVel)
	self.publish_vel1(acc_imu)
	self.publish_generalForce(self._vehicle_model._vel)
	self.publish_equivalentControl(self._vehicle_model.vel)
        self._prev_t = t






if __name__ == '__main__':
    print('Starting Model-based Sliding Mode Controller')
    rospy.init_node('rov_mb_sm_controller')



    try:
        node = ROV_MB_SMController()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')
    print('exiting')
