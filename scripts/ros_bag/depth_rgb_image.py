#!/home/vaishakh/myenv/bin/python3

import rospy
import cv2
import pickle
import numpy as np

from sensor_msgs.msg import Image


class ImageSubscriber:
    def __init__(self):
        rospy.init_node('depth_and_rgb_image_subscriber', anonymous=True)

        data = rospy.wait_for_message('/xtion/depth/image_raw', Image)
        # self.depth_image_sub = rospy.Subscriber(
        #     "/xtion/depth/image_raw", Image, self.depth_image_callback)
        # self.rgb_image_sub = rospy.Subscriber(
        #     "/xtion/rgb/image_raw", Image, self.rgb_image_callback)
        # adjust this value to normalize the depth image range
        self.depth_image_norm_factor = 1
        self.depth_image = np.frombuffer(
            data.data, dtype=np.uint16).reshape(data.height, data.width)
        # converting to float and scaling to 0-1 range
        self.depth_image = self.depth_image.astype(
            np.float32) / self.depth_image_norm_factor
        # save depth data to file
        np.savetxt("empty_depth_image.csv", self.depth_image, delimiter=",")
        cv2.imwrite('empty_depth_image.png', self.depth_image * 255)
        data = rospy.wait_for_message('/xtion/rgb/image_raw', Image)
        # extract raw RGB data from ROS Image message
        self.rgb_image = np.frombuffer(data.data, dtype=np.uint8).reshape(
            data.height, data.width, -1)
        self.rgb_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB)
        # save RGB data to file
        np.savetxt("empty_rgb_image.csv", self.rgb_image.ravel(),
                   delimiter=",", fmt='%.6e')
        cv2.imwrite('empty_rgb_image.png', self.rgb_image)

    def depth_image_callback(self, data):
        try:
            # extract raw depth data from ROS Image message
            self.depth_image = np.frombuffer(
                data.data, dtype=np.uint16).reshape(data.height, data.width)
            # converting to float and scaling to 0-1 range
            self.depth_image = self.depth_image.astype(
                np.float32) / self.depth_image_norm_factor
            # save depth data to file
            np.savetxt("empty_depth_image.csv",
                       self.depth_image, delimiter=",")
            cv2.imwrite('empty_depth_image.png', self.depth_image * 255)

        except Exception as e:
            print(e)

    def rgb_image_callback(self, data):
        try:
            # extract raw RGB data from ROS Image message
            self.rgb_image = np.frombuffer(data.data, dtype=np.uint8).reshape(
                data.height, data.width, -1)
            self.rgb_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB)
            # save RGB data to file
            np.savetxt("empty_rgb_image.csv", self.rgb_image.ravel(),
                       delimiter=",", fmt='%.6e')
            cv2.imwrite('empty_rgb_image.png', self.rgb_image)

        except Exception as e:
            print(e)

    def display_images(self):

        # display depth_image
        cv2.imshow('depth_image.png', self.depth_image)
        print(self.depth_image.shape)

        # display rgb_image
        cv2.imshow('rgb_image.png', self.rgb_image)
        print(self.rgb_image.shape[0])

        cv2.waitKey(1000)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        imagesub = ImageSubscriber()

        rospy.spin()
        imagesub.display_images()
    except rospy.ROSInterruptException:
        pass
