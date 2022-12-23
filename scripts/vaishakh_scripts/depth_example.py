#!/usr/bin/env python

import numpy as np
import cv2

import rospy
from sensor_msgs.msg import Image
import ros_numpy

class DepthSubscriber:
    def __init__(self):
        self.sub = rospy.Subscriber('/xtion/depth_registered/image', Image, self.callback)
    
    def callback(self, msg):
        depth_data = ros_numpy.numpify(msg)
        print(depth_data.shape)
        
if __name__ == '__main__':
    rospy.init_node('depth')

    try:
        while not rospy.is_shutdown():
            pass
    except KeyboardInterrupt:
        pass