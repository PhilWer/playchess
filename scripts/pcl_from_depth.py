#!/home/silvia/venv/pcd/bin/python3
#Processing of the pointcloud over the chessboard

# Rospy for the subscriber
import rospy
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import open3d as o3d
import os
import sensor_msgs.point_cloud2 as pc2
import ctypes
import struct
import yaml
import copy
import cv2
import csv

#My scripts
import config as cfg

#Defines
imported_rgb_image = '/home/silvia/tiago_public_ws/src/tiago_playchess/pcl_depth_synchro/03/rgb03.png' #/root/tiago_public_ws/src/tiago_playchess/pcl_depth_synchro/03/rgb03.png
imported_depth_image = '/home/silvia/tiago_public_ws/src/tiago_playchess/pcl_depth_synchro/03/depth_norm03.png' #/root/tiago_public_ws/src/tiago_playchess/pcl_depth_synchro/03/depth_norm03.png
imported_depth_image_csv = '/home/silvia/tiago_public_ws/src/tiago_playchess/pcl_depth_synchro/03/depth03.csv' #/root/tiago_public_ws/src/tiago_playchess/pcl_depth_synchro/03/depth03.csv
imported_configurations = r'/home/silvia/tiago_public_ws/src/tiago_playchess/scripts/config/simulation_config.yaml' #/root/tiago_public_ws/src/tiago_playchess/scripts/config/simulation_config.yaml
imported_depth_img_raw = '/home/silvia/tiago_public_ws/src/tiago_playchess/pcl_depth_synchro/03/prova_cm.png' #/root/tiago_public_ws/src/tiago_playchess/pcl_depth_synchro/03/prova_cm.png
imported_chessboard_squares = r'/home/silvia/tiago_public_ws/src/tiago_playchess/config/simul_config_not_transformed.yaml' #/root/tiago_public_ws/src/tiago_playchess/config/simul_config_not_transformed.yaml
#without color information: r'/home/silvia/tiago_public_ws/src/tiago_playchess/Chessboard_data_2021.11.02/simul_config_not_transformed.yaml'

class DepthProcessing:
#Class containing functions to process chessboard images and detect squares.
	def __init__(self):
		self.verbose = False
		#with open(imported_depth_image_csv) as file:
		depth_data = np.genfromtxt(imported_depth_image_csv, delimiter = ',')
		#print(np.nanmin(depth_data), np.nanmax(depth_data))
		#print(depth_data.shape)
		#cv2.imshow('depth', depth_data)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
		#cv2.imwrite('/root/tiago_public_ws/src/tiago_playchess/pcl_depth_synchro/03/prova_cm.png', depth_data)
		#Import depth image
		#self.depth_image = cv2.imread(imported_depth_image)
		self.depth_image = o3d.io.read_image(imported_depth_img_raw)#imported_depth_image)
		#self.rgb_image = cv2.imread(imported_rgb_image)
		self.rgb_image = o3d.io.read_image(imported_rgb_image)
		self.rgb_im = cv2.imread(imported_rgb_image) 
		self.rgb_im = cv2.cvtColor(self.rgb_im, cv2.COLOR_BGR2RGB) / 255.0
		#cv2.imshow('RGB image', self.rgb_image)
		#cv2.imshow('depth image', self.depth_image)

		#Import the chessboard squares centers as computed with CV
		with open(imported_chessboard_squares) as file:
			self.squares_centers = yaml.load(file, Loader=yaml.Loader)

		#Array creation
		self.pts = []
		self.pts_color = []
		for u in range(640):
			col = u
			for v in range(480):
				row = v
				if not np.isnan(depth_data[row, col]):
					pt = [u, v, depth_data[row, col]]
					pt_rgb = self.rgb_im[row, col]
					self.pts_color.append(pt_rgb)
					self.pts.append(pt)
		self.pts = np.array(self.pts)

		#Constants
		self.rotation_in_quaternions = [-0.675, 0.677, -0.202, -0.215]
		self.translation = [0.040, 0.934, 0.957]
		self.K = np.array([523.2994491601762, 0.0, 312.0471722095908, 0.0, 524.1979376240457, 249.9013550067579, 0.0, 0.0, 1.0]) #Intrinsic matrix
		self.width = 640
		self.height = 480

		#Load configurations
		with open(imported_configurations) as file:
			self.config = yaml.load(file, Loader=yaml.Loader)
		self.color = self.config.get('color')
		self.live_chessboard_situation_complete = cfg.live_chessboard_situation_complete
		self.starting_chessboard_situation_complete = cfg.starting_chessboard_situation_complete
		self.pieces_coordinates = cfg.pieces_coordinates
		self.columns_indexes = cfg.columns
		if self.color == 'white':
			self.squares_to_index = cfg.squares_to_index_white
			self.rows_indexes = cfg.rows_white
			self.opposite_color = 'black'
			self.castle_squares = ['A8', 'E8', 'H8']
			self.seventh_row = 'row7'
			self.second_row = 'row2'
		else:
			self.squares_to_index = cfg.squares_to_index_black
			self.rows_indexes = cfg.rows_black
			self.opposite_color = 'white'
			self.seventh_row = 'row2'
			self.second_row = 'row7'

	def pcl_creation(self, rgb_image, depth_image):
		rot_matrix = self.quaternion_rotation_matrix(self.rotation_in_quaternions)
		immagine = o3d.geometry.Image(np.array(np.asarray(rgb_image)[:, :, :3]).astype('uint8')) #Create the Image geometry element
		rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(immagine, depth_image, convert_rgb_to_intensity = True, depth_scale=0.01)

		#Extrinsic matrix creation
		extrinsic = np.array([[rot_matrix[0][0], rot_matrix[0][1], rot_matrix[0][2], self.translation[0]], [rot_matrix[1][0], rot_matrix[1][1], rot_matrix[1][2], self.translation[1]], [rot_matrix[2][0], rot_matrix[2][1], rot_matrix[2][2], self.translation[2]], [0, 0, 0, 1]])
		#Set the intrinsic parameters of the camera.
		phc = o3d.camera.PinholeCameraIntrinsic()
		phc.set_intrinsics(self.width, self.height, self.K[0], self.K[4], self.K[2], self.K[5])

		#pcd = o3d.geometry.PointCloud.create_from_depth_image(immagine, phc, extrinsic, depth_scale = 1000.0, depth_trunc = 1000.0, stride = 1, project_valid_depth_only = True)
		pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, phc, extrinsic, project_valid_depth_only = True)
		o3d.visualization.draw_geometries([pcd], zoom = 1000, front = [0, 0, -1], lookat = [0, 0, 1], up = [0, -1, 0])
		pts = np.asarray(pcd.points)
		if self.verbose:
			print(pts[:, 0].min(axis = 0), pts[:, 0].max(axis = 0))
			print(pts[:, 1].min(axis = 0), pts[:, 1].max(axis = 0))
			print(pts[:, 2].min(axis = 0), pts[:, 2].max(axis = 0))

	def uvz_to_xyz(self, uv, z, K):
	    ##########
	    # from: https://medium.com/yodayoda/from-depth-map-to-point-cloud-7473721d3f
	    # uv:   [list] pixel coordinates of the point in the image
	    # z:    [float] depth value in the same measurement unit of the output
	    # K:    [list[list]] intrinsic camera matrix (3x3)
	    ##########
	    # format the arrays
	    z = z[:, np.newaxis]                                        # (N, 1)
	    ones = np.ones((z.shape))                                   # (N, 1)
	    uv = np.hstack((uv, ones, np.reciprocal(z)))                # (N, 4)
	    # attach a dummy dimension so that matmul sees it as a stack of (4, 1) vectors
	    uv = np.expand_dims(uv, axis = 2)                           # (N, 4, 1)
	    
	    # invert the intrinsic matrix
	    fx, S, cx, fy, cy = K[0], K[1], K[2], K[4], K[5] 
	    K_inv = [   [1/fx, -S/(fx*fy), (S*cy-cx*fy)/(fx*fy), 0],
	                [0, 1/fy, -cy/fy, 0],
	                [0, 0, 1, 0],
	                [0, 0, 0, 1]
	            ]
	    # compute the spacial 3D coordinates for the points
	    xyz = z[:, np.newaxis] * np.matmul(K_inv, uv)   # (N, 4, 1)
	    return xyz[:, :3].reshape(-1, 3)  

	def quaternion_rotation_matrix(self, Q):
		"""
		Covert a quaternion into a full three-dimensional rotation matrix.
	 
		Input
		:param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
	 
		Output
		:return: A 3x3 element matrix representing the full 3D rotation matrix. 
		This rotation matrix converts a point in the local reference 
		frame to a point in the global reference frame.
		"""
		# Extract the values from Q
		q0 = Q[0]
		q1 = Q[1]
		q2 = Q[2]
		q3 = Q[3]
	     
		# First row of the rotation matrix
		r00 = 2 * (q0 * q0 + q1 * q1) - 1
		r01 = 2 * (q1 * q2 - q0 * q3)
		r02 = 2 * (q1 * q3 + q0 * q2)
	     
		# Second row of the rotation matrix
		r10 = 2 * (q1 * q2 + q0 * q3)
		r11 = 2 * (q0 * q0 + q2 * q2) - 1
		r12 = 2 * (q2 * q3 - q0 * q1)
	     
		# Third row of the rotation matrix
		r20 = 2 * (q1 * q3 - q0 * q2)
		r21 = 2 * (q2 * q3 + q0 * q1)
		r22 = 2 * (q0 * q0 + q3 * q3) - 1
	     
		# 3x3 rotation matrix
		rot_matrix = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
		                    
		return rot_matrix

	def visualize_points(self, pcl, points):
		#Function to visualize specific points of the PointCloud in red.
		#points. [list] of points like [[x,y,z], [x,y,z]]
		pcd_red = o3d.geometry.PointCloud()
		xyz = np.asarray(points)
		pcd_red.points = o3d.utility.Vector3dVector(xyz)
		#Paint them to red
		pcd_red.paint_uniform_color([1, 0, 0])
		o3d.visualization.draw_geometries([pcl, pcd_red])
		return pcd_red

def main():
	rospy.init_node('pcl_processor')
	depth_processor = DepthProcessing()

	xyz = depth_processor.uvz_to_xyz(depth_processor.pts[:, :2], depth_processor.pts[:, -1], depth_processor.K)
	pcd_red = o3d.geometry.PointCloud()
	pcd_red.points = o3d.utility.Vector3dVector(xyz)
	pcd_red.colors = o3d.utility.Vector3dVector(depth_processor.pts_color)
	pts = np.asarray(pcd_red.points)
	'''
	print(pts[:, 0].min(axis = 0), pts[:, 0].max(axis = 0))
	print(pts[:, 1].min(axis = 0), pts[:, 1].max(axis = 0))
	print(pts[:, 2].min(axis = 0), pts[:, 2].max(axis = 0))
	'''


	depth_processor.visualize_points(pcd_red, depth_processor.squares_centers)


	#o3d.visualization.draw_geometries([pcd_red])
	'''
	depth_processor.pcl_creation(depth_processor.rgb_image, depth_processor.depth_image)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	'''

if __name__ == '__main__':
     main()

