#!/usr/bin/env python3

# from https://github.com/felixchenfy/open3d_ros_pointcloud_conversion/blob/master/lib_cloud_conversion_between_Open3D_and_ROS.py

import open3d as o3d
import cv2
import numpy as np
from ctypes import * # convert float to uint32
import os
import sys

import rospy
from std_msgs.msg import Header, Bool
from sensor_msgs.msg import PointCloud2, PointField, CompressedImage, Image
import sensor_msgs.point_cloud2 as pc2
import message_filters


# Path and file format definition
SAVE_DIR = '/home/pal/tiago_public_ws/src/playchess/scripts/PointcloudForCalibration' #'pcd_save_path' # TODO: receive save dir from launch and create /cloud subfolder if it does not exist
                                                                        #       in this way, you can use this node along with img_saver (img_single) to save
                                                                        #       'all' the outputs of the vision sensor
#FILENAME = 'cloud'
FILENAME = 'calibrationCloud'
FILE_FORMAT = '.ply'

VERBOSE = False
TIMESTAMP_AS_NAME = True

# The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + \
    [PointField(name='rgb', offset=16, datatype=PointField.FLOAT32, count=1)]

# Bit operations
BIT_MOVE_16 = 2**16
BIT_MOVE_8 = 2**8
convert_rgbUint32_to_tuple = lambda rgb_uint32: (
    (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, (rgb_uint32 & 0x000000ff)
)
convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
    int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
)

# Convert the datatype of point cloud from Open3D to ROS PointCloud2 (XYZRGB only)
def convertCloudFromOpen3dToRos(open3d_cloud, frame_id="odom"):
    # Set "header"
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    # Set "fields" and "cloud_data"
    points=np.asarray(open3d_cloud.points)
    if not open3d_cloud.colors: # XYZ only
        fields=FIELDS_XYZ
        cloud_data=points
    else: # XYZ + RGB
        fields=FIELDS_XYZRGB
        # -- Change rgb color from "three float" to "one 24-byte int"
        # 0x00FFFFFF is white, 0x00000000 is black.
        colors = np.floor(np.asarray(open3d_cloud.colors)*255) # nx3 matrix
        colors = colors[:,0] * BIT_MOVE_16 +colors[:,1] * BIT_MOVE_8 + colors[:,2]  
        cloud_data=np.c_[points, colors]
    
    # create ros_cloud
    return pc2.create_cloud(header, fields, cloud_data)

def convertCloudFromRosToOpen3d(ros_cloud):
    
    # Get cloud data from ros_cloud
    field_names=[field.name for field in ros_cloud.fields]
    cloud_data = list(pc2.read_points(ros_cloud, skip_nans = False, field_names = field_names))
    if VERBOSE:
        print('Cloud data: ', str(len(cloud_data)))

    # Check empty
    open3d_cloud = o3d.geometry.PointCloud()
    if len(cloud_data)==0:
        print("Converting an empty cloud")
        return None

    # Set open3d_cloud
    if "rgb" in field_names:
        IDX_RGB_IN_FIELD=3 # x, y, z, rgb
        
        # Get xyz
        xyz = [(x,y,z) for x,y,z,rgb in cloud_data ] # (why cannot put this line below rgb?)
        if VERBOSE:
            print('XYZ: ', str(len(xyz)))
        # Get rgb
        # Check whether int or float
        if type(cloud_data[0][IDX_RGB_IN_FIELD])==float: # if float (from pcl::toROSMsg)
            rgb = [convert_rgbFloat_to_tuple(rgb) for x,y,z,rgb in cloud_data ]
        else:
            rgb = [convert_rgbUint32_to_tuple(rgb) for x,y,z,rgb in cloud_data ]
        if VERBOSE:
            print('RGB: ', str(len(rgb)))
        # combine
        open3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))
        open3d_cloud.colors = o3d.utility.Vector3dVector(np.array(rgb)/255.0)
    else:
        xyz = [(x,y,z) for x,y,z in cloud_data ] # get xyz
        open3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))

    # return
    return open3d_cloud

class PointCloud2Saver:
    def __init__(self, save_path):
        #Publishers initialization
        self.pc_saved_publisher = rospy.Publisher('/pc_saved', Bool, queue_size = 10)

        # change the topic name with a the '/pointcloud2' placeholder and use remap at launch
        self.pointcloud2_sub = rospy.Subscriber('/xtion/depth_registered/points', PointCloud2, self.save_pointcloud2)   # /xtion/depth/points               no RGB
        self.count = 0 #1

        #self.save_path = save_path                                                                                          # /xtion/depth_registered/points    with RGB
        # check if /cloud subfolder exists, otherwise create it
        '''
        self.cloud_subfolder = os.path.join(save_path, 'cloud')
        if not os.path.exists(self.cloud_subfolder):
            os.makedirs(self.cloud_subfolder)
        '''

    def save_pointcloud2(self, ros_point_cloud): # a service maybe better than a callback
        out_pcd = convertCloudFromRosToOpen3d(ros_point_cloud)
        #progressive_number = str(ros_point_cloud.header.stamp.secs) + '_' + str(ros_point_cloud.header.stamp.nsecs) if TIMESTAMP_AS_NAME else str(self.count)        
        #o3d.io.write_point_cloud(os.path.join(self.cloud_subfolder, FILENAME + progressive_number + FILE_FORMAT), out_pcd)
        print(save_path)
        o3d.io.write_point_cloud(os.path.join(SAVE_DIR, FILENAME + FILE_FORMAT), out_pcd) #save_path al posto di SAVE_DIR
        self.count += 1
        rospy.loginfo('PointCloud2 message saved to .ply file.')
        if self.count == 1: #If one pointcloud has been saved
            self.pc_saved_publisher.publish(True)



if __name__ == '__main__':
    if len(sys.argv) > 1:
        save_path = sys.argv[1]
        # maybe check if it does exist...
    else:
        save_path = SAVE_DIR

    rospy.init_node('pcl_listener', anonymous = True)
    try:
        print(SAVE_DIR)
        pointcloud2_saver = PointCloud2Saver(SAVE_DIR)
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo('Shutting down pointcloud2_saver node.')