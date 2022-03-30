#!/usr/bin/env python
# make the robot perform a pre-defined motion

# Rospy for the subscriber
import rospy
import numpy as np
# ROS Image messages
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import Quaternion, PointStamped, Pose, Point, PoseStamped, WrenchStamped
#ROS Image message --> Open CV2 image converter
from cv_bridge import CvBridge, CvBridgeError
#Open CV2 for saving an image
import cv2

bridge = CvBridge()
counter = 0
depth = []


def image_callback(msg):
    print('Received an image!')
    global counter
    counter = counter + 1
    '''
    #For simple Images
    try:
        #Convert the ROS image to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError, e:
        print(e)
    else:
        #Save the OpenCV2 image as a jpeg
        cv2.imwrite('camera_image' + str(counter) + '.jpeg', cv2_img)
    '''

    #For compressed images
    np_arr = np.fromstring(msg.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    cv2.imwrite('camera_image' + str(counter) + '.jpeg', image_np)
    
    #save depth pointcloud points
    #f = open("depth_points.txt", "w+")
    #print(depth)
    #f.write(depth)
    #f.close()
'''
def depth_callback(msg):
    global depth
    depth = msg
    print(depth)
'''

def main():
    rospy.init_node('image_listener')
    #Define your image tpic
    image_topic = "/xtion/rgb/image_rect_color/compressed" #/compressed
    #depth_topic = "/xtion/depth_registered/points"
    #Setup your subscriber and define its callback
    rospy.Subscriber(image_topic, CompressedImage, image_callback) #CompressedImage
    #rospy.Subscriber(depth_topic, PointCloud2, image_callback)
    #Spin until ctrl+c
    rospy.spin()

if __name__ == '__main__':
     main()

