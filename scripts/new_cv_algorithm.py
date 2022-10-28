#!/usr/bin/env python

# Python libs
import numpy as np
import cv2

# ROS libs/packages
import rospy
import ros_numpy

# ROS messages
from sensor_msgs.msg import Image


class ImageProcessing(object):
    def __init__(self):
        self.img_sub = rospy.Subscriber('/xtion/rgb/image_rect_color', Image, self.callback)
        self.img = None

    def callback(self, msg):
        rgb_img = ros_numpy.numpify(msg)
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        self.img = bgr_img


if __name__ == "__main__":
    # Initialize the node
    rospy.init_node('img_proc')
    
    # Initialize the class object
    ip = ImageProcessing()

    try:
        # Main Loop
        while not rospy.is_shutdown():
            if ip.img is not None:
                cv2.imshow('robot view', ip.img)
                cv2.waitKey(1) #[ms]

    except KeyboardInterrupt:
        rospy.loginfo('Image processing stopped by the user.')