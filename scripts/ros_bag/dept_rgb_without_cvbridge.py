#!/usr/bin/env python

import rospy
import cv2
import numpy as np
import os
import open3d as o3d
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class MyNode:
    def __init__(self):
        rospy.init_node('my_node', anonymous=True)
        self.image_sub = rospy.Subscriber('/xtion/rgb/image_raw', Image, self.callback)
        self.bridge = CvBridge()
        self.move = 0

    def callback(self, data):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        # Get RGB and depth data
        rgb_img, depth_arr = self._get_rgb_and_depth(cv_image)

        # Save RGB and depth data
        self.save_data(rgb_img, depth_arr)

        # Convert depth data to point cloud
        pcd = self.convert_depth_to_point_cloud(depth_arr)

        # Perform additional processing as needed

    def _get_rgb_and_depth(self, cv_image):
        # Get RGB and depth data from OpenCV image
        rgb_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        depth_arr = cv2.imread('/path/to/depth/image.png', cv2.IMREAD_ANYDEPTH)
        return rgb_img, depth_arr

    def save_data(self, rgb_img, depth_arr):
        self.move += 1
        mv_folder = 'move_{}'.format(self.move)
        try:
            os.mkdir(mv_folder)
        except FileExistsError:
            rospy.logwarn(
                'The content of the ' + 'move_{}'.format(self.move) + ' will be overwritten.')
        # Save the RGB and depth data
        cv2.imwrite(mv_folder + '/rgb.png', rgb_img)
        depth_arr = depth_arr.reshape((depth_arr.shape[0], 1))
        np.savetxt(mv_folder + '/depth.csv', depth_arr, delimiter=",")
        # NOTE. Retrieve the depth data with depth_image = np.genfromtxt(mv_folder + '/depth.csv', delimiter=',')

    def convert_depth_to_point_cloud(self, depth_arr):
        # Convert depth image to point cloud
        intrinsics = self._get_intrinsics()
        # Normalize depth values
        depth_arr = depth_arr / 1000.0
        # Create Open3D depth image
        depth_img = o3d.geometry.Image(depth_arr)
        # Create Open3D RGB image (set to black)
        w, h = depth_arr.shape
        rgb_arr = np.zeros((w, h, 3), dtype=np.uint8)
        rgb_img = o3d.geometry.Image(rgb_arr)
        # Create Open3D rgbd image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img, depth_img, depth_scale=1.0/1000.0,
                                                                   depth_trunc=2.0, convert_rgb_to_intensity=False)
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
        return pcd

    def _get_intrinsics(self):
        # Set intrinsics (modify to match your camera)
        fx = 570.3422241210938
        fy = 570.3422241210938
        cx = 314.5
        cy = 235.5
        intrinsics = o3d.camera.PinholeCameraIntrinsic(width=640, height=480, fx=fx, fy=fy, cx=cx, cy=cy)
        return intrinsics

    def _publish_point_cloud(self, pcd):
      # Create and publish ROS point cloud message
      header = Header()
      header.stamp = rospy.Time.now()
      header.frame_id = 'camera_link'
      pc2_msg = o3d_to_ros(pcd, header)
      self.pc_pub.publish(pc2_msg)
    
    def visualize_point_cloud(self, pcd):
        # Create visualizer object
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Add point cloud to visualizer
        vis.add_geometry(pcd)

        # Set visualizer camera parameters
        ctr = vis.get_view_control()
        ctr.rotate(0.0, 0.0)
        ctr.set_zoom(0.5)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # Run visualizer until window is closed
        while not vis.was_destroyed():
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

        # Destroy visualizer object
        vis.destroy_window()


if __name__ == '__main__':
    try:
        node = MyNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass