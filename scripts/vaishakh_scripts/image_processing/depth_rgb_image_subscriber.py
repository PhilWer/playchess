#!/usr/bin/env python2

import rospy
import cv2
import pickle
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class ImageSubscriber:
    def __init__(self):
        rospy.init_node('depth_and_rgb_image_subscriber', anonymous=True)
        self.bridge = CvBridge()
        self.depth_image_sub = rospy.Subscriber(
            "/xtion/depth/image_raw", Image, self.depth_image_callback)
        self.rgb_image_sub = rospy.Subscriber(
            "/xtion/rgb/image_raw", Image, self.rgb_image_callback)

    def depth_image_callback(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
            #converting to np array
            self.depth_array = np.array(self.depth_image, dtype=np.float32)   
        except CvBridgeError as e:
            print(e)

    

    def rgb_image_callback(self, data):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(data,"bgr8")
            #converting to np array
            self.rgb_array=np.array(self.rgb_image,dtype=np.int32)
            
            
        except CvBridgeError as e:
            print(e)

    def save_images(self):
        #saving depth image
        np.savetxt("empty_depth_image.csv", self.depth_array, delimiter=",")
        cv2.imwrite('empty_depth_image.png', self.depth_image)

        #saving rgb_image
        np.savetxt("empty_rgb_image.csv", self.rgb_array.ravel(), delimiter=",", fmt='%.6e')
        cv2.imwrite('empty_rgb_image.png', self.rgb_image)


    def display_images(self):
        
        #display depth_image
        cv2.imshow('depth_image.png', self.depth_image)
        print(self.depth_image.shape)

        #display rgb_image
        cv2.imshow('rgb_image.png', self.rgb_image)
        print(self.rgb_image.shape[0])
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        imagesub = ImageSubscriber()
        rospy.spin()
        imagesub.save_images()
        imagesub.display_images()
    except rospy.ROSInterruptException:
        pass
