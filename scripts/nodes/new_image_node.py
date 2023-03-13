#!/home/myvenv_p3/bin/python3
import rospy
from sensor_msgs.msg import PointCloud2, PointField, Image, CompressedImage
from std_msgs.msg import Bool, Int16, String
import numpy as np
import open3d as o3d
import os
import cv2
import csv
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from time import time


def CallbackState():
        PLAYCHESS_PKG_DIR = '/home/pal/tiago_public_ws/src/playchess'
        MOVES_DATA_DIR = PLAYCHESS_PKG_DIR + '/data/moves'

            # Get RGB (+ depth, for reproducibility with v1.0)
        rgb_img_before, __, __, depth_arr_before = _get_rgb_and_depth()

        move = 1
        mv_folder = MOVES_DATA_DIR + 'move_{}'.format(move)
        try:
            os.mkdir(mv_folder)
        except FileExistsError:
            rospy.logwarn(
                'The content of the ' + 'move_{}'.format(move) + ' will be overwritten.')
        # Save the RGB and depth data BEFORE the execution of the opponent's move
        cv2.imwrite(mv_folder + '/rgb_before.png', rgb_img_before)
        np.savetxt(mv_folder + '/depth_before.csv',
                   depth_arr_before, delimiter=",")
        # NOTE. Retrieve the depth data with depth_image = np.genfromtxt(mv_folder + '/depth_before.csv', delimiter=',')

        # Get RGB and depth
        rgb_img_after, __, __, depth_arr_after = _get_rgb_and_depth()

        mv_folder = MOVES_DATA_DIR + 'move_{}'.format(move)
        # Save the RGB and depth data BEFORE the execution of the opponent's move
        cv2.imwrite(mv_folder + '/rgb_after.png', rgb_img_after)
        np.savetxt(mv_folder + '/depth_after.csv',
                   depth_arr_after, delimiter=",")
            # Retrieve the depth data with depth_image = np.genfromtxt(mv_folder + '/depth_after.csv', delimiter=',')
        # Identify the move and broadcast messages accordingly
        def _get_rgb_and_depth(self):
        # Get RGB and depth data
        img_msg = rospy.wait_for_message('/xtion/rgb/image_raw', Image)
        rgb_img, rgb_arr = self._convert_image_msg(img_msg, rgb=True)
        depth_msg = rospy.wait_for_message('/xtion/depth/image_raw', Image)
        depth_img, depth_arr = self._convert_image_msg(depth_msg, rgb=False)
        return rgb_img, rgb_arr, depth_img, depth_arr

    def _convert_image_msg(self, msg, rgb=True):
        if rgb:
            encoding = "bgr8"
            dtype = np.int32
        else:   # if depth
            encoding = "32FC1"
            dtype = np.float32

        try:
            img = self.bridge.imgmsg_to_cv2(msg, encoding)
            # Converting to np array
            np_arr = np.array(img, dtype=dtype)
        except CvBridgeError as e:
            print(e)

        return img, np_arr

       

def main():
    rospy.init_node('pcl_processor')
    depth_processor = DepthProcessing()

    # Initialize a subscriber to monitor the state.
    rospy.Subscriber("/state", Int16, CallbackState)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down the CV module.")


if __name__ == '__main__':
    main()