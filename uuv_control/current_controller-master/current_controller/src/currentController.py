#!/usr/bin/env python

"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import rospy
from uuv_world_ros_plugins_msgs.srv import SetCurrentVelocity
import sys


#T = np.array([[0, 0, 0, vx],[0, 0, 0, vz],[0, 0, 0, vy],[0, 0, 0, 1]])
class currentController():
    def __init__(self):
        try:
            rospy.wait_for_service('/hydrodynamics/set_current_velocity', timeout=20)
        except rospy.ROSException:
            print 'Current velocity services not available! Closing node...'
            sys.exit(-1)
        self.setCurrent = rospy.ServiceProxy('/hydrodynamics/set_current_velocity', SetCurrentVelocity)
        self.vel = 0.0
        self.v_dir = 0.0
        self.h_dir = 0.0
        self.periodicUpdate = rospy.Timer(rospy.Duration(1.0), self.updateCurrent)
        self.counter = 0.0
    def updateCurrent(self, event):
        #self.vel = np.random.random(1)[0]
	self.vel = np.random.uniform(0.5, 1.5)
        self.v_dir = np.sin(self.counter)
        #self.h_dir = 0.0
	self.h_dir = np.cos(self.counter)
        print (self.vel, "   ", self.v_dir, "   ", self.h_dir)
        self.setCurrent(self.vel,self.v_dir, self.h_dir)
        self.counter += 0.1
        
if __name__ == '__main__':
    try:
        rospy.init_node('currentController', log_level=rospy.DEBUG)
    except rospy.ROSInterruptException as error:
        print('pubs error with ROS: ', error)
        exit(1)

    curCon = currentController()
    rospy.spin()

