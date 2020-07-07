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
import rospy
import numpy as np
from uuv_control_interfaces import DPPIDControllerBase


class ROV_MBFLController(DPPIDControllerBase):
    """PID controller for the dynamic positioning of ROVs."""

    _LABEL = 'PID'
    def __init__(self):
        self._tau = np.zeros(6)
        self._tau1 = np.zeros(6)
        DPPIDControllerBase.__init__(self, False)
        self._is_init = True
        self._pid_control = np.zeros(6)
        self._last_vel = np.zeros(6)
        self._last_t = None
        self._logger.info(self._LABEL + ' ready')

    def _reset_controller(self):
        super(ROV_MBFLController, self).reset_controller()
        self._pid_control = np.zeros(6)
        self._tau = np.zeros(6)
        self._tau1 = np.zeros(6)

    def update_controller(self):
        if not self._is_init:
            return False

        t = rospy.get_time()
        if self._last_t is None:
            self._last_t = t
            self._last_vel = self._vehicle_model.to_SNAME(self._reference['vel']) 
          
            return False

        dt = t - self._last_t
        if dt <= 0:
            self._last_t = t
            self._last_vel = self._vehicle_model.to_SNAME(self._reference['vel']) 
       
            return False
        self._pid_control = self.update_pid()

        
        vel = self._vehicle_model.to_SNAME(self._reference['vel'])
    
        acc = (vel - self._last_vel) / dt

        self._vehicle_model._update_damping(vel)
        self._vehicle_model._update_coriolis(vel)
        self._vehicle_model._update_restoring(q=self._reference['rot'], use_sname=True)
      
        self._tau1 = np.dot(self._vehicle_model.Mtotal, acc) + \
                    np.dot(self._vehicle_model.Ctotal, vel) + \
                    np.dot(self._vehicle_model.Dtotal, vel) + \
                    self._vehicle_model.restoring_forces
                    
        self._tau[0] = self._pid_control[0]
        self._tau[1] = 200
        self._tau[2] = self._pid_control[2]
        self._tau[3] = self._pid_control[3]
        self._tau[4] = self._pid_control[4]
        self._tau[5] = self._pid_control[5]

        _MPara=self._vehicle_model._linear_damping
	self.publish_MPara(_MPara)

        # Publish control forces and torques
        #self.publish_control_wrench(self._pid_control + self._vehicle_model.from_SNAME(self._tau))
        self.publish_control_wrench(self._tau)
        #self.publish_control_wrench(self._pid_control)
        self._last_t = t
        self._last_vel = self._vehicle_model.to_SNAME(self._reference['vel'])
        return True


if __name__ == '__main__':
    print('Starting PID')
    rospy.init_node('rov_pid_controller')

    try:
        node = ROV_MBFLController()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')
    print('exiting')
