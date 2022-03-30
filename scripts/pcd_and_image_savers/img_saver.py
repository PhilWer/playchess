#!/usr/bin/env python

# Python libs
import numpy as np
import os
import cv2
import sys
# ROS libs
import rospy
from cv_bridge import CvBridge, CvBridgeError
import message_filters
# ROS messages
from sensor_msgs.msg import Image, CompressedImage

# TODO: understand how to properly pass command line arguments when rosrunning a node instead of roslaunching it

SAVE_DIR = 'img_save_path'

TIMESTAMP_AS_NAME = True
SINGLE_IMG = False

class ImgSaver:
    def __init__(self, save_path):
        self.rgb_subscriber = message_filters.Subscriber('/xtion/rgb/image_rect_color/compressed', CompressedImage)
        self.depth_subscriber = message_filters.Subscriber('/xtion/depth/image', Image) #/compressed', CompressedImage)
        # synchronize the subscribers and define a unique callback
        self.filter_pointcloud = message_filters.ApproximateTimeSynchronizer([self.rgb_subscriber, self.depth_subscriber], 30, slop = 0.1)
        self.filter_pointcloud.registerCallback(self.save_rgbd)

        #self.save_path = save_path
        # check if /depth, /depth_norm and /rgb subfolders exist, otherwise create them
        self.depth_subfolder = os.path.join(save_path, 'depth')
        self.depth_norm_subfolder = os.path.join(save_path, 'depth_norm')
        self.rgb_subfolder = os.path.join(save_path, 'rgb')
        
        if not os.path.exists(self.depth_subfolder):
            os.makedirs(self.depth_subfolder)
        if not os.path.exists(self.depth_norm_subfolder):
            os.makedirs(self.depth_norm_subfolder)
        if not os.path.exists(self.rgb_subfolder):
            os.makedirs(self.rgb_subfolder)

        self.count = 1
        self.acquire = True     # enable/disable the callback
                                # actually the callback does not stop to be triggered, but its execution does not have any effect
        self.bridge = CvBridge()

    def save_rgbd(self, rgb_msg, depth_msg):
        if self.acquire:

            ###### DEPTH (for geometrical information)
            try:
                # try to decode the Image message from the depth camera stream to a CV2 image
                cv2_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding = '32FC1')
                #cv2_img = self.bridge.compressed_imgmsg_to_cv2(depth_msg, desired_encoding = '32FC1')
            
            except CvBridgeError as e:
                rospy.logerr(e)

            else:
                img_array = np.array(cv2_img, dtype = np.dtype('f8'))
                # save the array to a .csv file
                progressive_number = str(depth_msg.header.stamp.secs) + '_' + str(depth_msg.header.stamp.nsecs) if TIMESTAMP_AS_NAME else str(self.count)
                filename = os.path.join(self.depth_subfolder, 'depth' + progressive_number + '.csv')
                np.savetxt(filename, img_array.reshape(480, 640), fmt = '%.4e', delimiter = ',') #fmt should be adapted/adaptable depending on the input...

                rospy.loginfo('Raw depth information saved to .txt file.')  # maybe add info about distance measurement unit (depending on the topic)

            ###### DEPTH NORMALIZED (for visualization)
                # normalize the data in the [0 1] range 
                cv2_img_norm = cv2.normalize(cv2_img, cv2_img, 0, 255, cv2.NORM_MINMAX)
                cv2.imwrite(os.path.join(self.depth_norm_subfolder, 'depth_norm' + progressive_number + '.png'), cv2_img_norm)
                rospy.loginfo('Depth information saved to as .png file.')  # maybe add info about distance measurement unit (depending on the topic)

            ######## RGB
            try:
                # try to decode the CompressedImage message from the RGB camera stream to a CV2 image
                cv2_image = self.bridge.compressed_imgmsg_to_cv2(rgb_msg, desired_encoding = 'passthrough') # maybe revert to Image messages just to be consistent with the depth part

            except CvBridgeError as e:
                rospy.logerr(e)

            else:
                # save the image to a .png file
                progressive_number = str(rgb_msg.header.stamp.secs) + '_' + str(rgb_msg.header.stamp.nsecs) if TIMESTAMP_AS_NAME else str(self.count)
                filename = os.path.join(self.rgb_subfolder, 'rgb' + progressive_number + '.png')
                cv2.imwrite(filename, cv2_image)

                rospy.loginfo('RGB image saved as .png file.\n')

            self.count += 1
        self.acquire = False if SINGLE_IMG else True    # get a single pair of RGB and depth images if the SINGLE_IMG flag is on
        # TODO: in single mode shut down the node after a single acquisition
        #       maybe print the save files once at the beginning

if __name__ == '__main__':
    if len(sys.argv) > 1:
        save_path = sys.argv[1]
        # maybe check if it does exist...
    else:
        save_path = SAVE_DIR

    # initialize a ROS node
    rospy.init_node('img_saver', anonymous = True)
    img_saver = ImgSaver(save_path)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS image saver module.")