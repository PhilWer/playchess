#!/usr/bin/env python3
#Processing of the pointcloud over the chessboard

# Rospy for the subscriber
import rospy
from sensor_msgs.msg import PointCloud2, PointField, Image, CompressedImage
from std_msgs.msg import Bool, Int16, String
import numpy as np
import open3d as o3d
import os
import sensor_msgs.point_cloud2 as pc2
import ctypes
import struct
import yaml
import pickle
import copy
import cv2
import csv
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from time import time
from decoder import decode_CompressedImage_depth

#My scripts
import config as cfg

ready = False

PLAYCHESS_PKG_DIR = '/home/pal/tiago_public_ws/src/tiago_playchess'

#Publishers initialization
state_publisher = rospy.Publisher('/state', Int16, queue_size = 10)
promotion_happened_publisher = rospy.Publisher('/promotion_happened', String, queue_size = 10)
en_passant_square_publisher = rospy.Publisher('/en_passant_square', String, queue_size = 10)
castle_square_publisher = rospy.Publisher('/castle_square', String, queue_size = 10)
opponent_move_start_square_publisher = rospy.Publisher('/opponent_move_start_square', String, queue_size = 10)
opponent_move_end_square_publisher = rospy.Publisher('/opponent_move_end_square', String, queue_size = 10)

#Defines
imported_chessboard_squares = PLAYCHESS_PKG_DIR + '/config/simul_config_not_transformed.yaml'
imported_chessboard_vertices = PLAYCHESS_PKG_DIR + '/config/chessboard_vertices_not_transformed.yaml'
imported_chessboard_vertices_pickle = PLAYCHESS_PKG_DIR + '/config/vertices_not_transformed.pickle'
imported_configurations = PLAYCHESS_PKG_DIR + '/scripts/config/simulation_config.yaml'
imported_chessboard_squares_pickle = PLAYCHESS_PKG_DIR + '/config/simul_config_not_transformed.pickle'
pointcloud_for_calibration = PLAYCHESS_PKG_DIR + '/PointcloudForCalibration/calibrationCloud.ply'

occupied_thresh_file = PLAYCHESS_PKG_DIR + '/config/occupied_thresh.pickle'
color_thresh_file = PLAYCHESS_PKG_DIR + '/config/color_thresh.pickle'
is_color_over_file = PLAYCHESS_PKG_DIR + '/config/is_color_over.pickle'


#Flags initialization
analysis_not_done_yet = True
calibration_not_done_yet = True

class DepthProcessing:
#Class containing functions to process depth and rgb images to detect the opponent move.
	def __init__(self):
		
		#Message initialiation
		self.state = Int16()

		self.verbose = False

		#Publishers initialization
		self.done_publisher = rospy.Publisher('/synchronization_done', Bool, queue_size = 10)

		#Subscribers initialization
		self.done_subscriber = rospy.Subscriber("/synchronization_done", Bool, self.identify_moved_piece)

		with open(imported_chessboard_squares_pickle, 'rb') as fin:
			self.squares_centers = pickle.load(fin, encoding = 'latin1')

		#Import the chessboard vertices as computed with CV
		with open(imported_chessboard_vertices_pickle, 'rb') as file:
			self.vertices = pickle.load(file, encoding = 'latin1')

		self.bridge = CvBridge()

		#Constants
		self.K = np.array([523.2994491601762, 0.0, 312.0471722095908, 0.0, 524.1979376240457, 249.9013550067579, 0.0, 0.0, 1.0]) #Intrinsic matrix
		self.width = 640
		self.height = 480

		#Load configurations
		with open(imported_configurations) as file:
			self.config = yaml.load(file, Loader = yaml.Loader)
		self.color = self.config.get('color')
		self.dir_live_chessboard_situation = PLAYCHESS_PKG_DIR + "/scripts/live_chessboard_situation.yaml"
		self.starting_chessboard_situation_complete = cfg.starting_chessboard_situation_complete
		self.pieces_coordinates = cfg.pieces_coordinates
		if self.color == 'white':
			self.squares_to_index = cfg.squares_to_index_white
			self.rows_indexes = cfg.rows_white
			self.opposite_color = 'black'
			self.castle_squares = ['a8', 'e8', 'h8', 'king_e8', 'rook_a8', 'rook_h8']
			self.castle_ending_squares = ['d8', 'c8', 'g8', 'f8']
			self.seventh_row = 'row7'
			self.second_row = 'row2'
			self.columns_indexes = cfg.columns
		else:
			self.squares_to_index = cfg.squares_to_index_black
			self.rows_indexes = cfg.rows_black
			self.opposite_color = 'white'
			self.castle_squares = ['a1', 'e1', 'h1', 'king_e1', 'rook_a1', 'rook_h1']
			self.castle_ending_squares = ['d1', 'c1', 'g1', 'f1']
			self.seventh_row = 'row2'
			self.second_row = 'row7'
			self.columns_indexes = cfg.columns_black

	def downsample(self, pcl, voxel_size):
		#Function to downsample the PointCloud with a voxel of voxel_size.
		downsample = pcl.voxel_down_sample(voxel_size = 0.01)
		if self.verbose:
			o3d.visualization.draw_geometries([downsample])

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

	def normals(self, pcl):
		#Function that computes the normals of each point of a PointCloud.
		pcl.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 0.1, max_nn = 30))
		if self.verbose:
			o3d.visualization.draw_geometries([pcl], point_show_normal = True)
		return pcl.normals

	def get_points_in_matrix_from_pointstamped(self, pointsStamped):
		#Function to transform a list of PointStamped objects in a matrix of nx3 coordinates
		points_number = len(pointsStamped)
		points_matrix = np.zeros((points_number, 3)) #Initialize the empty matrix
		for i in range(0, points_number):
			points_matrix[i] = [pointsStamped[i].point.x, pointsStamped[i].point.y, pointsStamped[i].point.z]
		return points_matrix

	def get_points_in_matrix_from_array(self, array):
		#Function to transform a list of array objects in a matrix of nx3 coordinates
		points_number = len(array)
		points_matrix = np.zeros((points_number, 3)) #Initialize the empty matrix
		for i in range(0, points_number):
			points_matrix[i] = [array[i][0], array[i][1], array[i][2]]
		return points_matrix

	def synchronize_rgb_depth(self):
		self.t_synchro = self.t_recognition = time()
		self.acquire = True
		#Function that call a callback whenever an rgb image and a depth image are acquired contemporary from TIAGo.
		self.rgb_subscriber = message_filters.Subscriber('/xtion/rgb/image_rect_color/compressed', CompressedImage)
		self.depth_subscriber = message_filters.Subscriber('/xtion/depth/image_rect/compressedDepth', CompressedImage) #/compressed', CompressedImage)    xtion/depth/image_rect/compressedDepth  PRIMA ERA: /xtion/depth/image e Image
		#Synchronize the subscribers and define a unique callback
		self.filter_pointcloud = message_filters.ApproximateTimeSynchronizer([self.rgb_subscriber, self.depth_subscriber], 30, slop = 0.1)
		self.filter_pointcloud.registerCallback(self.create_pointcloud_from_depth)

	
	def create_pointcloud_from_depth(self, rgb_msg, depth_msg):
		#Callback function to recreate the desired pointcloud starting from the acquired rgb and depth images.
		if self.acquire:
			rospy.sleep(2)
			#Import the rgb image to save colors of the points
			self.rgb_im = self.bridge.compressed_imgmsg_to_cv2(rgb_msg, desired_encoding = 'passthrough') #cv2.imread(imported_rgb_image) #DA VERIFICARE SE FUNZIONA, SE NO DEVO SALVARE LE IMMAGINI ACQUISITE DA TIAGO
			self.rgb_im = cv2.cvtColor(self.rgb_im, cv2.COLOR_BGR2RGB) / 255.0 #Convert the image from BGR to RGB
			depth_data, __ = decode_CompressedImage_depth(depth_msg)

			#Array creation
			self.pts = []
			self.pts_color = []
			for u in range(640):
				col = u
				for v in range(480):
					row = v
					if not np.isnan(depth_data[row, col]): #Save the points' coordinates, depth information and color if the depth value is not 'nan'
						pt = [u, v, depth_data[row, col]]
						pt_rgb = self.rgb_im[row, col]
						self.pts_color.append(pt_rgb)
						self.pts.append(pt)
			self.pts = np.array(self.pts)
			

			xyz = self.uvz_to_xyz(self.pts[:, :2], self.pts[:, -1], self.K)
			self.pcd = o3d.geometry.PointCloud()
			self.pcd.points = o3d.utility.Vector3dVector(xyz)
			self.pcd.colors = o3d.utility.Vector3dVector(self.pts_color)

			if self.verbose:
				o3d.visualization.draw_geometries([self.pcd])

			self.acquire = False

			rospy.loginfo('Synchro took ' + str(time() - self.t_synchro) + 's')

			self.done_publisher.publish(True)

	def create_pointcloud(self, points, color):
		#Function that creates a PointCloud starting from a matrix of xyz points [nx3 matrix].
		#color: array that tells the desired color for the PointCloud (Red: [1,0,0], Green: [0, 1, 0])
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(points)
		pcd.paint_uniform_color(color) #Paint the points i red
		if self.verbose:
			o3d.visualization.draw_geometries([pcd])
		return pcd

	def uvz_to_xyz(self, uv, z, K):
	    #From: https://medium.com/yodayoda/from-depth-map-to-point-cloud-7473721d3f
	    #uv: [list] pixel coordinates of the point in the image
	    #z: [float] depth value in the same measurement unit of the output
	    #K: [list[list]] intrinsic camera matrix (3x3)
	    #Format the arrays

		z = z[:, np.newaxis] # (N, 1)
		ones = np.ones((z.shape)) # (N, 1)
		uv = np.hstack((uv, ones, np.reciprocal(z))) # (N, 4)
		#Attach a dummy dimension so that matmul sees it as a stack of (4, 1) vectors
		uv = np.expand_dims(uv, axis = 2) # (N, 4, 1)

		#Invert the intrinsic matrix
		fx, S, cx, fy, cy = K[0], K[1], K[2], K[4], K[5] 
		K_inv = [   [1/fx, -S/(fx*fy), (S*cy-cx*fy)/(fx*fy), 0],
					[0, 1/fy, -cy/fy, 0],
					[0, 0, 1, 0],
					[0, 0, 0, 1]
				]
		#Compute the spacial 3D coordinates for the points
		xyz = z[:, np.newaxis] * np.matmul(K_inv, uv)   # (N, 4, 1)
		return xyz[:, :3].reshape(-1, 3)  

	def create_bounding_box(self, pcl, length, width, R, centro, reference_normal, box_extent, margin, translation):
		#Function to create a bounding box given information about the PointCloud and the location of the desired box.
		#length: extension of the chessboard in the y direction.
		#width: extension of the chessboard in the x direction.
		#R: rotation matrix computed looking at the normals of the chessboard squares.
		#centro: center of the chessboard.
		#reference normal: dimensions of the normal to the chessboard in the xyz directions.
		#box_extent: extension in the z direction of the bounding box with which I want to crop the PointCloud
		#margin: external margin to make sure to take all the chessboard.
		#translation: 
		
		old_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.1)
		old_frame.translate(centro)
		self_created_frame = copy.deepcopy(old_frame)
		self_created_frame.rotate(R, center = centro)

		#Creation of the bounding box
		centro[2] = centro[2] - translation
		bounding_box = o3d.geometry.OrientedBoundingBox(centro, R, [length + margin, width + margin, box_extent])

		#Visualize the bounding box over the original PointCloud
		if self.verbose:
			#pass
			o3d.visualization.draw_geometries([pcl, bounding_box, self_created_frame])
		return bounding_box, self_created_frame

	def crop(self, pcl, bounding_box):
		#Function to crop a PointCloud given a bounding box.

		#Crop the PointCloud
		point_cloud_crop = pcl.crop(bounding_box)

		#Visualize the cropped pointcloud
		if self.verbose:
			o3d.visualization.draw_geometries([point_cloud_crop, bounding_box])

		return point_cloud_crop

	def get_rot_matrix(self, pcl, normals, centro):
		#Function to compute the rotation matrix to rotate the reference frame of the bounding box that I want to create and allign it to the ref frame of the pointcloud seen by TIAGo.

		#Choose the reference normal to keep to compute rotation
		for n in normals:
			if n[0] > 0:
				reference_normal = n
				break
		if self.verbose:
			print('NORMAL: ' + str(reference_normal))

		R = pcl.get_rotation_matrix_from_xyz((- reference_normal[1], 0, 0))
		pcl_r = copy.deepcopy(pcl)
		pcl_r.rotate(R, center=(centro[0], centro[1], centro[2]))
		if self.verbose:
			o3d.visualization.draw_geometries([pcl, pcl_r])
		return R, reference_normal
	
	def king_checking(self, square, chessboard, threshold, squares_boundboxes, color_threshold, is_color_over):
		t_king = time()
		print('KING CHECKING...')
		#square: square from which the king has been moved.

		self.found = False
		possible_squares = []
		higher_points_number = 0
		higher_points_square = 'none'
		second_higher_points_number = 0
		second_higher_points_square = 'none'
		piece_color = 'none'

		en_passant = False
		en_passant_square = 'none'
		promotion = False
		ending_square = 'none'
		#Save the index of the starting square.
		index = self.squares_to_index[square]

		#Check if castle happened:
		if square == 'e8':
			if self.live_chessboard_situation['f8'][0] == 'none' and self.live_chessboard_situation['g8'][0] == 'none' and self.live_chessboard_situation['h8'][0] == 'rook_h8':
				square_box = self.crop(chessboard, squares_boundboxes[self.squares_to_index['f8']])
				points_number = len(np.asarray(square_box.points))
				if points_number > threshold:
					ending_square = 'castle'
					self.castle_squares_found = ['g8', 'f8']
			if ending_square == 'none' and self.live_chessboard_situation['d8'][0] == 'none' and self.live_chessboard_situation['c8'][0] == 'none' and self.live_chessboard_situation['b8'][0] == 'none' and self.live_chessboard_situation['a8'] == 'rook_a8':
				square_box = self.crop(chessboard, squares_boundboxes[self.squares_to_index['d8']])
				points_number = len(np.asarray(square_box.points))
				if points_number > threshold:
					ending_square = 'castle'
					self.castle_squares_found = ['c8', 'd8']

		if square == 'e1':
			if self.live_chessboard_situation['f1'][0] == 'none' and self.live_chessboard_situation['g1'][0] == 'none' and self.live_chessboard_situation['h1'][0] == 'rook_h1':
				square_box = self.crop(chessboard, squares_boundboxes[self.squares_to_index['f1']])
				points_number = len(np.asarray(square_box.points))
				if points_number > threshold:
					ending_square = 'castle'
					self.castle_squares_found = ['g1', 'f1']
			if ending_square == 'none' and self.live_chessboard_situation['d1'][0] == 'none' and self.live_chessboard_situation['c1'][0] == 'none' and self.live_chessboard_situation['b1'][0] == 'none' and self.live_chessboard_situation['a1'][0] == 'rook_a1':
				square_box = self.crop(chessboard, squares_boundboxes[self.squares_to_index['d1']])
				points_number = len(np.asarray(square_box.points))
				if points_number > threshold:
					ending_square = 'castle'
					self.castle_squares_found = ['c1', 'd1']

		if ending_square == 'none':
			#Save the indexes of the reachable squares for the king.
			if (index + 7) in range(64): 
				possible_squares.append(index + 7)
			if (index - 1) in range(64): 
				possible_squares.append(index - 1)
			if (index - 9) in range(64): 
				possible_squares.append(index - 9)
			if (index - 7) in range(64): 
				possible_squares.append(index - 7)
			if (index + 1) in range(64): 
				possible_squares.append(index + 1)
			if (index + 9) in range(64): 
				possible_squares.append(index + 9)
			if (index - 8) in range(64): 
				possible_squares.append(index - 8)
			if (index + 8) in range(64): 
				possible_squares.append(index + 8)
			#Check the squares on the diagonal.
			for element in possible_squares:
				#Save the name of the current square
				for key, value in self.squares_to_index.items():
					if value == element:
						current_square = key
				square_box = self.crop(chessboard, squares_boundboxes[element])
				points_number = len(np.asarray(square_box.points))
				if self.verbose:
					print('square: '+str(element))
					print(points_number)
				if points_number > threshold * 1.3 and points_number > higher_points_number  and self.live_chessboard_situation[current_square][1] != self.opposite_color: #The square is occupied and is the one with the higher number of points.
					second_higher_points_number = higher_points_number
					higher_points_number = points_number
					second_higher_points_square = higher_points_square
					higher_points_square = current_square
					RGB_colors = square_box.colors
					mean_RGB = self.mean_RGB_points([RGB_colors])
					second_piece_color = piece_color
					piece_color = self.recognize_color(mean_RGB, color_threshold, is_color_over)
					if self.verbose:
						print('higher: '+str(higher_points_square))
						print('second higher: '+str(second_higher_points_square))

			#Check if the square was occupied before too.
			if higher_points_square != 'none':
				if self.live_chessboard_situation[higher_points_square][1] == 'none':
					#This means that the square was not occupied before. I just found the ending square of the king.
					ending_square = higher_points_square
					#Update the live_chessboard_situation
					self.live_chessboard_situation[higher_points_square][0] = self.live_chessboard_situation[square][0]
					self.live_chessboard_situation[higher_points_square][1] = self.opposite_color
					self.live_chessboard_situation[square][0] = 'none'
					self.live_chessboard_situation[square][1] = 'none'
					self.found = True

				elif self.live_chessboard_situation[higher_points_square][1] == self.color and piece_color == self.opposite_color:
					#The piece has been captured.
					ending_square = higher_points_square
					#Update the live_chessboard_situation
					self.live_chessboard_situation[higher_points_square][0] = self.live_chessboard_situation[square][0]
					self.live_chessboard_situation[higher_points_square][1] = self.opposite_color
					self.live_chessboard_situation[square][0] = 'none'
					self.live_chessboard_situation[square][1] = 'none'
					self.found = True
				elif second_higher_points_square != 'none':
					if self.live_chessboard_situation[second_higher_points_square][1] != self.opposite_color and second_piece_color == self.opposite_color:
						ending_square = second_higher_points_square
						#Update the live_chessboard_situation
						self.live_chessboard_situation[second_higher_points_square][0] = self.live_chessboard_situation[square][0]
						self.live_chessboard_situation[second_higher_points_square][1] = self.opposite_color
						self.live_chessboard_situation[square][0] = 'none'
						self.live_chessboard_situation[square][1] = 'none'
						self.found = True

		if not self.found: #If the end square has not been found again, it means that the pcl processing has not worked properly.
			rospy.loginfo('PCL processing has not worked properly. Try processing again.')

		if self.verbose:
			rospy.loginfo('checking_king took ' + str(time() - t_king) + 's')

		return ending_square, en_passant, en_passant_square, promotion
	
	def queen_checking(self, square, chessboard, threshold, squares_boundboxes, color_threshold, is_color_over):
		t_queen = time()
		print('QUEEN CHECKING...')
		#square: square from which the queen has been moved.

		self.found = False
		possible_squares = []
		diagonal_squares = []
		count = 1
		higher_points_number = 0
		second_higher_points_number = 0
		higher_points_square = 'none'
		second_higher_points_square = 'none'
		piece_color = 'none'

		en_passant = False
		en_passant_square = 'none'
		promotion = False

		#Save the index of the starting square.
		index = self.squares_to_index[square]

		#Save the row and the column of the square
		for key, values in self.rows_indexes.items():
			if index in values:
				row = self.rows_indexes[key]

		for key, values in self.columns_indexes.items():
			if index in values:
				column = self.columns_indexes[key]
		
		#Get the position of the square in the row and column arrays
		ind_riga = row.index(index)
		ind_colonna = column.index(index)
		if self.verbose:
			print('ind riga: '+str(ind_riga))
			print('ind col: '+str(ind_colonna))

		#Remove from the row and column array the starting square,cause i will not search on that one for the end square.
		#row.remove(index)
		#column.remove(index)

		for numb in range(7 - ind_riga):
			for key, value in self.squares_to_index.items():
				if value == row[ind_riga + (numb + 1)]:
					current_square_row = key
			if self.live_chessboard_situation[current_square_row][1] == 'none':
				possible_squares.append(row[ind_riga + (numb + 1)])
			else:
				possible_squares.append(row[ind_riga + (numb + 1)])
				break

		for numb in range(ind_riga):
			for key, value in self.squares_to_index.items():
				if value == row[ind_riga - (numb + 1)]:
					current_square_row = key
			if self.live_chessboard_situation[current_square_row][1] == 'none':
				possible_squares.append(row[ind_riga - (numb + 1)])
			else:
				possible_squares.append(row[ind_riga - (numb + 1)])
				break

		for numb in range(7 - ind_colonna):
			for key, value in self.squares_to_index.items():
				if value == column[ind_colonna + (numb + 1)]:
					current_square_col = key
			if self.live_chessboard_situation[current_square_col][1] == 'none':
				possible_squares.append(column[ind_colonna + (numb + 1)])
			else:
				possible_squares.append(column[ind_colonna + (numb + 1)])
				break

		for numb in range(ind_colonna):
			for key, value in self.squares_to_index.items():
				if value == column[ind_colonna - (numb + 1)]:
					current_square_col = key
			if self.live_chessboard_situation[current_square_col][1] == 'none':
				possible_squares.append(column[ind_colonna - (numb + 1)])
			else:
				possible_squares.append(column[ind_colonna - (numb + 1)])
				break


		#possible_squares = np.append(row, column)

		#Save the indexes of the squares on the diagonal of the queen.
		
		flag1 = False
		flag2 = False
		flag3 = False
		flag4 = False
		
		while count < 8:
			if (index + 7 * count) in range(64) and not flag1: # and count1 <= ciao:
				for key, value in self.squares_to_index.items():
					if value == (index + 7 * count):
						current_square_diag = key
				if self.live_chessboard_situation[current_square_diag][1] == 'none':
					possible_squares = np.append(possible_squares, index + 7 * count)
				else:
					possible_squares = np.append(possible_squares, index + 7 * count)
					flag1 = True
			if (index - 9 * count) in range(64) and not flag2: # and count2 <= (7-ciao1):
				for key, value in self.squares_to_index.items():
					if value == (index - 9 * count):
						current_square_diag = key
				if self.live_chessboard_situation[current_square_diag][1] == 'none':
					possible_squares = np.append(possible_squares, index - 9 * count)
				else:
					possible_squares = np.append(possible_squares, index - 9 * count)
					flag2 = True
			if (index - 7 * count) in range(64) and not flag3: # and count3 <= (7 - ciao1):
				for key, value in self.squares_to_index.items():
					if value == (index - 7 * count):
						current_square_diag = key
				if self.live_chessboard_situation[current_square_diag][1] == 'none':
					possible_squares = np.append(possible_squares, index - 7 * count)
				else:
					possible_squares = np.append(possible_squares, index - 7 * count)
					flag3 = True
			if (index + 9 * count) in range(64) and not flag4: # and count4 <= (7 - ciao):
				for key, value in self.squares_to_index.items():
					if value == (index + 9 * count):
						current_square_diag = key
				if self.live_chessboard_situation[current_square_diag][1] == 'none':
					possible_squares = np.append(possible_squares, index + 9 * count)
				else:
					possible_squares = np.append(possible_squares, index + 9 * count)
					flag4 = True
			count += 1
		if self.verbose:
			print('POSSIBLE SQUARES: '+str(possible_squares))

		#Check the reachable squares.
		for element in possible_squares:
			#Save the name of the current square
			for key, value in self.squares_to_index.items():
				if value == element:
					current_square = key
			square_box = self.crop(chessboard, squares_boundboxes[element])
			points_number = len(np.asarray(square_box.points))
			if self.verbose:
				print('curr square: '+str(current_square))
				print(points_number)
				print(self.live_chessboard_situation[current_square][1])
			if points_number > threshold * 1.3 and points_number > higher_points_number and self.live_chessboard_situation[current_square][1] != self.opposite_color: #The square is occupied and is the one with the higher number of points.
				second_higher_points_number = higher_points_number
				higher_points_number = points_number
				second_higher_points_square = higher_points_square
				higher_points_square = current_square
				RGB_colors = square_box.colors
				mean_RGB = self.mean_RGB_points([RGB_colors])
				second_piece_color = piece_color
				piece_color = self.recognize_color(mean_RGB, color_threshold, is_color_over)
				if self.verbose:
					print(higher_points_square)
					print(second_higher_points_square)

		#Check if the square was occupied before too.
		if higher_points_square != 'none':
			if self.live_chessboard_situation[higher_points_square][1] == 'none':
				#This means that the square was not occupied before. I just found the ending square of the queen.
				ending_square = higher_points_square
				#Update the live_chessboard_situation
				self.live_chessboard_situation[higher_points_square][0] = self.live_chessboard_situation[square][0]
				self.live_chessboard_situation[higher_points_square][1] = self.opposite_color
				self.live_chessboard_situation[square][0] = 'none'
				self.live_chessboard_situation[square][1] = 'none'
				self.found = True

			elif self.live_chessboard_situation[higher_points_square][1] == self.color and piece_color == self.opposite_color:
				#The piece has been captured.
				ending_square = higher_points_square
				#Update the live_chessboard_situation
				self.live_chessboard_situation[higher_points_square][0] = self.live_chessboard_situation[square][0]
				self.live_chessboard_situation[higher_points_square][1] = self.opposite_color
				self.live_chessboard_situation[square][0] = 'none'
				self.live_chessboard_situation[square][1] = 'none'
				self.found = True

			elif second_higher_points_square != 'none':
				if self.live_chessboard_situation[second_higher_points_square][1] != self.opposite_color and second_piece_color == self.opposite_color:
					ending_square = second_higher_points_square
					#Update the live_chessboard_situation
					self.live_chessboard_situation[second_higher_points_square][0] = self.live_chessboard_situation[square][0]
					self.live_chessboard_situation[second_higher_points_square][1] = self.opposite_color
					self.live_chessboard_situation[square][0] = 'none'
					self.live_chessboard_situation[square][1] = 'none'
					self.found = True
				else:
					ending_square = 'none'

			else:
				ending_square = 'none'
		else:
			ending_square = 'none'

		if not self.found: #If the end square has not been found again, it means that the pcl processing has not worked properly.
			rospy.loginfo('PCL processing has not worked properly. Try processing again.')

		if self.verbose:
			rospy.loginfo('checking_queen took ' + str(time() - t_queen) + 's')

		return ending_square, en_passant, en_passant_square, promotion
	
	def knight_checking(self, square, chessboard, threshold, squares_boundboxes, color_threshold, is_color_over):
		t_knight = time()
		print('KNIGHT CHECKING...')
		#square: square from which the knight has been moved.

		self.found = False
		possible_squares = []
		higher_points_number = 0
		higher_points_square = 'none'

		en_passant = False
		en_passant_square = 'none'
		promotion = False

		#Save the index of the starting square.
		index = self.squares_to_index[square]

		#Save the row and the column of the square
		for key, values in self.rows_indexes.items():
			if index in values:
				row = key

		for key, values in self.columns_indexes.items():
			if index in values:
				column = key
		columns = list(self.columns_indexes.keys())
		ind_column = columns.index(column)
		prev1_column_ind = ind_column - 1
		next1_column_ind = ind_column + 1
		prev2_column_ind = ind_column - 2
		next2_column_ind = ind_column + 2
		rows = list(self.rows_indexes.keys())
		ind_row = rows.index(row)
		prev1_row_ind = ind_row - 1
		next1_row_ind = ind_row + 1
		prev2_row_ind = ind_row - 2
		next2_row_ind = ind_row + 2

		#Save the indexes of the reachable squares for the knight.
		if prev2_column_ind in range(8):
			if prev1_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev2_column_ind]]:
					if casella_colonna in self.rows_indexes[rows[prev1_row_ind]]:
						possible_squares.append(casella_colonna)
			if next1_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev2_column_ind]]:
					if casella_colonna in self.rows_indexes[rows[next1_row_ind]]:
						possible_squares.append(casella_colonna)
		if prev1_column_ind in range(8):
			if prev2_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev1_column_ind]]:
					if casella_colonna in self.rows_indexes[rows[prev2_row_ind]]:
						possible_squares.append(casella_colonna)
			if next2_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev1_column_ind]]:
					if casella_colonna in self.rows_indexes[rows[next2_row_ind]]:
						possible_squares.append(casella_colonna)
		if next1_column_ind in range(8):
			if prev2_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next1_column_ind]]:
					if casella_colonna in self.rows_indexes[rows[prev2_row_ind]]:
						possible_squares.append(casella_colonna)
			if next2_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next1_column_ind]]:
					if casella_colonna in self.rows_indexes[rows[next2_row_ind]]:
						possible_squares.append(casella_colonna)
		if next2_column_ind in range(8):
			if prev1_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next2_column_ind]]:
					if casella_colonna in self.rows_indexes[rows[prev1_row_ind]]:
						possible_squares.append(casella_colonna)
			if next1_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next2_column_ind]]:
					if casella_colonna in self.rows_indexes[rows[next1_row_ind]]:
						possible_squares.append(casella_colonna)
		#Check the squares.
		if self.verbose:
			print('possible square: '+str(possible_squares))
		for element in possible_squares:
			#Save the name of the current square
			for key, value in self.squares_to_index.items():
				if value == element:
					current_square = key
					square_box = self.crop(chessboard, squares_boundboxes[element])
					points_number = len(np.asarray(square_box.points))
					if self.verbose:
						print('++++ curr square: '+str(current_square))
						print('++++ points: '+str(points_number))
					if points_number > threshold * 0.8 and points_number > higher_points_number  and self.live_chessboard_situation[current_square][1] != self.opposite_color: #The square is occupied and is the one with the higher number of points.
						RGB_colors = square_box.colors
						mean_RGB = self.mean_RGB_points([RGB_colors])
						piece_color = self.recognize_color(mean_RGB, color_threshold, is_color_over)
						if self.verbose:
							print('piece color: '+str(piece_color))
						if piece_color == self.opposite_color:
							higher_points_number = points_number
							higher_points_square = current_square
		#Check if the square was occupied before too.
		if higher_points_square != 'none':
			if self.live_chessboard_situation[higher_points_square][1] == 'none':
				#This means that the square was not occupied before. I just found the ending square of the knight.
				ending_square = higher_points_square
				self.found = True

			elif self.live_chessboard_situation[higher_points_square][1] == self.color: # and piece_color == self.opposite_color:
				#The piece has been captured.
				ending_square = higher_points_square
				self.found = True
		else:
			ending_square = 'none'

		if not self.found: #If the end square has not been found again, it means that the pcl processing has not worked properly.
			rospy.loginfo('PCL processing has not worked properly. Try processing again.')
			ending_square = 'none'

		if self.verbose:
			rospy.loginfo('checking_knight took ' + str(time() - t_knight) + 's')

		return ending_square, en_passant, en_passant_square, promotion
	
	def rook_checking(self, square, chessboard, threshold, squares_boundboxes, color_threshold, is_color_over):
		t_rook = time()
		print('ROOK CHECKING...')
		#square: square from which the rook has been moved.

		self.found = False
		higher_points_number = 0
		higher_points_square = 'none'
		second_higher_points_number = 0
		second_higher_points_square = 'none'
		possible_squares = []
		piece_color = 'none'

		en_passant = False
		en_passant_square = 'none'
		promotion = False
		ending_square = 'none'
		ind_opp_color_on_row = []

		#Save the index of the starting square.
		index = self.squares_to_index[square]

		#Save the row and the column of the square
		for key, values in self.rows_indexes.items():
			if index in values:
				row = self.rows_indexes[key]

		for key, values in self.columns_indexes.items():
			if index in values:
				column = self.columns_indexes[key]
		
		#Get the position of the square in the row and column arrays
		ind_riga = row.index(index)
		ind_colonna = column.index(index)
		if self.verbose:
			print('ind riga: '+str(ind_riga))
			print('ind col: '+str(ind_colonna))

		#Remove from the row and column array the starting square,cause i will not search on that one for the end square.
		#row.remove(index)
		#column.remove(index)

		for numb in range(7 - ind_riga):
			for key, value in self.squares_to_index.items():
				if value == row[ind_riga + (numb + 1)]:
					current_square_row = key
			if self.live_chessboard_situation[current_square_row][1] == 'none':
				possible_squares.append(row[ind_riga + (numb + 1)])
			else:
				possible_squares.append(row[ind_riga + (numb + 1)])
				break

		for numb in range(ind_riga):
			for key, value in self.squares_to_index.items():
				if value == row[ind_riga - (numb + 1)]:
					current_square_row = key
			if self.live_chessboard_situation[current_square_row][1] == 'none':
				possible_squares.append(row[ind_riga - (numb + 1)])
			else:
				possible_squares.append(row[ind_riga - (numb + 1)])
				break

		for numb in range(7 - ind_colonna):
			for key, value in self.squares_to_index.items():
				if value == column[ind_colonna + (numb + 1)]:
					current_square_col = key
			if self.live_chessboard_situation[current_square_col][1] == 'none':
				possible_squares.append(column[ind_colonna + (numb + 1)])
			else:
				possible_squares.append(column[ind_colonna + (numb + 1)])
				break

		for numb in range(ind_colonna):
			for key, value in self.squares_to_index.items():
				if value == column[ind_colonna - (numb + 1)]:
					current_square_col = key
			if self.live_chessboard_situation[current_square_col][1] == 'none':
				possible_squares.append(column[ind_colonna - (numb + 1)])
			else:
				possible_squares.append(column[ind_colonna - (numb + 1)])
				break
		if self.verbose:
			print('POSSIBLE SQUARES: '+str(possible_squares))

		#Check if castle happened:
		if square == 'a8':
			if self.live_chessboard_situation['d8'][0] == 'none' and self.live_chessboard_situation['e8'][0] == 'king_e8' and self.live_chessboard_situation['b8'][0] == 'none' and self.live_chessboard_situation['c8'][0] == 'none':
				square_box = self.crop(chessboard, squares_boundboxes[self.squares_to_index['d8']])
				points_number = len(np.asarray(square_box.points))
				if points_number > threshold:
					ending_square = 'castle'
					self.castle_squares_found = ['c8', 'd8']

		if square == 'h8':
			if self.live_chessboard_situation['f8'][0] == 'none' and self.live_chessboard_situation['e8'][0] == 'king_e8' and self.live_chessboard_situation['g8'][0] == 'none':
				square_box = self.crop(chessboard, squares_boundboxes[self.squares_to_index['f8']])
				points_number = len(np.asarray(square_box.points))
				if points_number > threshold:
					ending_square = 'castle'
					self.castle_squares_found = ['g8', 'f8']

		if square == 'a1':
			if self.live_chessboard_situation['d1'][0] == 'none' and self.live_chessboard_situation['e1'][0] == 'king_e1' and self.live_chessboard_situation['b1'][0] == 'none' and self.live_chessboard_situation['c1'][0] == 'none':
				square_box = self.crop(chessboard, squares_boundboxes[self.squares_to_index['d1']])
				points_number = len(np.asarray(square_box.points))
				if points_number > threshold:
					ending_square = 'castle'
					self.castle_squares_found = ['c1', 'd1']

		if square == 'h1':
			if self.live_chessboard_situation['f1'][0] == 'none' and self.live_chessboard_situation['e1'][0] == 'king_e1' and self.live_chessboard_situation['g1'][0] == 'none':
				square_box = self.crop(chessboard, squares_boundboxes[self.squares_to_index['f1']])
				points_number = len(np.asarray(square_box.points))
				if points_number > threshold:
					ending_square = 'castle'
					self.castle_squares_found = ['g1', 'f1']

		if ending_square == 'none':
			#Check which squares on the row are occupied.
			for element in possible_squares:
				#Save the name of the current square
				for key, value in self.squares_to_index.items():
					if value == element:
						current_square = key
				square_box = self.crop(chessboard, squares_boundboxes[element])
				points_number = len(np.asarray(square_box.points))
				if self.verbose:
					print('++++ curr square: '+str(current_square))
					print('++++ points: '+str(points_number))
					print(higher_points_number)
					print(self.live_chessboard_situation[current_square][1])

				if points_number > threshold * 1.3 and points_number > higher_points_number  and self.live_chessboard_situation[current_square][1] != self.opposite_color: #The square is occupied and is the one with the higher number of points.
					second_higher_points_number = higher_points_number
					higher_points_number = points_number
					second_higher_points_square = higher_points_square
					higher_points_square = current_square #element
					RGB_colors = square_box.colors
					mean_RGB = self.mean_RGB_points([RGB_colors])
					second_piece_color = piece_color
					piece_color = self.recognize_color(mean_RGB, color_threshold, is_color_over)
					if self.verbose:
						print(higher_points_square)



			#Check if the square was occupied before too.
			if higher_points_square != 'none':
				if self.live_chessboard_situation[higher_points_square][1] == 'none':
					#This means that the square was not occupied before. I just found the ending square of the queen.
					ending_square = higher_points_square
					#Update the live_chessboard_situation
					self.live_chessboard_situation[higher_points_square][0] = self.live_chessboard_situation[square][0]
					self.live_chessboard_situation[higher_points_square][1] = self.opposite_color
					self.live_chessboard_situation[square][0] = 'none'
					self.live_chessboard_situation[square][1] = 'none'
					self.found = True

				elif self.live_chessboard_situation[higher_points_square][1] == self.color and piece_color == self.opposite_color:
					#The piece has been captured.
					ending_square = higher_points_square
					#Update the live_chessboard_situation
					if self.conta_torri != 2:
						self.live_chessboard_situation[higher_points_square][0] = self.live_chessboard_situation[square][0]
						self.live_chessboard_situation[higher_points_square][1] = self.opposite_color
						self.live_chessboard_situation[square][0] = 'none'
						self.live_chessboard_situation[square][1] = 'none'
					self.found = True

				elif second_higher_points_square != 'none':
					if self.live_chessboard_situation[second_higher_points_square][1] != self.opposite_color and second_piece_color == self.opposite_color:
						ending_square = second_higher_points_square
						#Update the live_chessboard_situation
						self.live_chessboard_situation[second_higher_points_square][0] = self.live_chessboard_situation[square][0]
						self.live_chessboard_situation[second_higher_points_square][1] = self.opposite_color
						self.live_chessboard_situation[square][0] = 'none'
						self.live_chessboard_situation[square][1] = 'none'
						self.found = True
					else:
						ending_square = 'none'
				else:
					ending_square = 'none'
			else:
				ending_square = 'none'

		if not self.found: #If the end square has not been found again, it means that the pcl processing has not worked properly.
			rospy.loginfo('PCL processing has not worked properly. Try processing again.')

		if self.verbose:
			rospy.loginfo('checking_rook took ' + str(time() - t_rook) + 's')

		return ending_square, en_passant, en_passant_square, promotion
			
	def bishop_checking(self, square, chessboard, threshold, squares_boundboxes, color_threshold, is_color_over):
		t_bishop = time()
		print('BISHOP CHECKING...')
		#square: square from which the bishop has been moved.

		self.found = False
		exist = True
		diagonal_squares = []
		count = 1
		promotion = False
		en_passant = False
		en_passant_square = 'none'
		higher_points_number = 0
		higher_points_square = 'none'
		higher_piece_color = 'none'

		#Save the index of the starting square.
		index = self.squares_to_index[square]

		#Save the row and the column of the square
		for key, values in self.rows_indexes.items():
			if index in values:
				row = key

		for key, values in self.columns_indexes.items():
			if index in values:
				column = key

		columns = list(self.columns_indexes.keys())
		ind_column = columns.index(column)
		prev1_column_ind = ind_column - 1
		next1_column_ind = ind_column + 1
		prev2_column_ind = ind_column - 2
		next2_column_ind = ind_column + 2
		prev3_column_ind = ind_column - 3
		next3_column_ind = ind_column + 3
		prev4_column_ind = ind_column - 4
		next4_column_ind = ind_column + 4
		prev5_column_ind = ind_column - 5
		next5_column_ind = ind_column + 5
		prev6_column_ind = ind_column - 6
		next6_column_ind = ind_column + 6
		prev7_column_ind = ind_column - 7
		next7_column_ind = ind_column + 7
		rows = list(self.rows_indexes.keys())
		ind_row = rows.index(row)
		prev1_row_ind = ind_row - 1
		next1_row_ind = ind_row + 1
		prev2_row_ind = ind_row - 2
		next2_row_ind = ind_row + 2
		prev3_row_ind = ind_row - 3
		next3_row_ind = ind_row + 3
		prev4_row_ind = ind_row - 4
		next4_row_ind = ind_row + 4
		prev5_row_ind = ind_row - 5
		next5_row_ind = ind_row + 5
		prev6_row_ind = ind_row - 6
		next6_row_ind = ind_row + 6
		prev7_row_ind = ind_row - 7
		next7_row_ind = ind_row + 7

		#Save the indexes of the squares on the diagonal of the bishop.
		if prev1_column_ind in range(8):
			if prev1_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev1_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[prev1_row_ind]] and self.live_chessboard_situation[square[0]][1] != self.opposite_color:
						diagonal_squares.append(casella_colonna)
			if next1_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev1_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[next1_row_ind]] and self.live_chessboard_situation[square[0]][1] != self.opposite_color:
						diagonal_squares.append(casella_colonna)
		if prev2_column_ind in range(8):
			if prev2_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev2_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[prev2_row_ind]] and self.live_chessboard_situation[square[0]][1] != self.opposite_color:
						diagonal_squares.append(casella_colonna)
			if next2_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev2_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[next2_row_ind]] and self.live_chessboard_situation[square[0]][1] != self.opposite_color:
						diagonal_squares.append(casella_colonna)
		if prev3_column_ind in range(8):
			if prev3_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev3_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[prev3_row_ind]] and self.live_chessboard_situation[square[0]][1] != self.opposite_color:
						diagonal_squares.append(casella_colonna)
			if next3_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev3_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[next3_row_ind]] and self.live_chessboard_situation[square[0]][1] != self.opposite_color:
						diagonal_squares.append(casella_colonna)
		if prev4_column_ind in range(8):
			if prev4_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev4_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[prev4_row_ind]] and self.live_chessboard_situation[square[0]][1] != self.opposite_color:
						diagonal_squares.append(casella_colonna)
			if next4_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev4_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[next4_row_ind]] and self.live_chessboard_situation[square[0]][1] != self.opposite_color:
						diagonal_squares.append(casella_colonna)
		if prev5_column_ind in range(8):
			if prev5_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev5_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[prev5_row_ind]] and self.live_chessboard_situation[square[0]][1] != self.opposite_color:
						diagonal_squares.append(casella_colonna)
			if next5_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev5_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[next5_row_ind]] and self.live_chessboard_situation[square[0]][1] != self.opposite_color:
						diagonal_squares.append(casella_colonna)
		if prev6_column_ind in range(8):
			if prev6_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev6_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[prev6_row_ind]] and self.live_chessboard_situation[square[0]][1] != self.opposite_color:
						diagonal_squares.append(casella_colonna)
			if next6_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev6_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[next6_row_ind]] and self.live_chessboard_situation[square[0]][1] != self.opposite_color:
						diagonal_squares.append(casella_colonna)
		if prev7_column_ind in range(8):
			if prev7_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev7_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[prev7_row_ind]] and self.live_chessboard_situation[square[0]][1] != self.opposite_color:
						diagonal_squares.append(casella_colonna)
			if next7_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev7_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[next7_row_ind]] and self.live_chessboard_situation[square[0]][1] != self.opposite_color:
						diagonal_squares.append(casella_colonna)
		if next1_column_ind in range(8):
			if prev1_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next1_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[prev1_row_ind]] and self.live_chessboard_situation[square[0]][1] != self.opposite_color:
						diagonal_squares.append(casella_colonna)
			if next1_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next1_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[next1_row_ind]] and self.live_chessboard_situation[square[0]][1] != self.opposite_color:
						diagonal_squares.append(casella_colonna)
		if next2_column_ind in range(8):
			if prev1_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next2_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[prev2_row_ind]] and self.live_chessboard_situation[square[0]][1] != self.opposite_color:
						diagonal_squares.append(casella_colonna)
			if next2_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next2_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[next2_row_ind]] and self.live_chessboard_situation[square[0]][1] != self.opposite_color:
						diagonal_squares.append(casella_colonna)
		if next3_column_ind in range(8):
			if prev1_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next3_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[prev3_row_ind]] and self.live_chessboard_situation[square[0]][1] != self.opposite_color:
						diagonal_squares.append(casella_colonna)
			if next3_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next3_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[next3_row_ind]] and self.live_chessboard_situation[square[0]][1] != self.opposite_color:
						diagonal_squares.append(casella_colonna)
		if next4_column_ind in range(8):
			if prev4_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next4_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[prev4_row_ind]] and self.live_chessboard_situation[square[0]][1] != self.opposite_color:
						diagonal_squares.append(casella_colonna)
			if next4_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next4_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[next4_row_ind]] and self.live_chessboard_situation[square[0]][1] != self.opposite_color:
						diagonal_squares.append(casella_colonna)
		if next5_column_ind in range(8):
			if prev5_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next5_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[prev5_row_ind]] and self.live_chessboard_situation[square[0]][1] != self.opposite_color:
						diagonal_squares.append(casella_colonna)
			if next5_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next5_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[next5_row_ind]] and self.live_chessboard_situation[square[0]][1] != self.opposite_color:
						diagonal_squares.append(casella_colonna)
		if next6_column_ind in range(8):
			if prev6_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next6_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[prev6_row_ind]] and self.live_chessboard_situation[square[0]][1] != self.opposite_color:
						diagonal_squares.append(casella_colonna)
			if next6_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next6_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[next6_row_ind]] and self.live_chessboard_situation[square[0]][1] != self.opposite_color:
						diagonal_squares.append(casella_colonna)
		if next7_column_ind in range(8):
			if prev7_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next7_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[prev7_row_ind]] and self.live_chessboard_situation[square[0]][1] != self.opposite_color:
						diagonal_squares.append(casella_colonna)
			if next7_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next7_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[next7_row_ind]] and self.live_chessboard_situation[square[0]][1] != self.opposite_color:
						diagonal_squares.append(casella_colonna)

		#Check the squares on the diagonal.
		for element in diagonal_squares:
			#Save the name of the current square
			for key, value in self.squares_to_index.items():
				if value == element:
					current_square = key
			square_box = self.crop(chessboard, squares_boundboxes[element])
			points_number = len(np.asarray(square_box.points))
			if self.verbose:
				print('++++ curr square: '+str(current_square))
				print('++++ points: '+str(points_number))
			if points_number > threshold * 0.8 and points_number > higher_points_number  and self.live_chessboard_situation[current_square][1] != self.opposite_color: #The square is occupied and is the one with the higher number of points.
				#higher_points_number = points_number
				#higher_points_square = current_square
				RGB_colors = square_box.colors
				mean_RGB = self.mean_RGB_points([RGB_colors])
				piece_color = self.recognize_color(mean_RGB, color_threshold, is_color_over)						
				if piece_color == self.opposite_color:
					higher_points_number = points_number
					higher_points_square = current_square
					higher_piece_color = piece_color
					print(higher_points_square)

		#Check if the square was occupied before too.
		if higher_points_square != 'none':
			if self.verbose:
				print('COLORE:  '+str(piece_color))
			if self.live_chessboard_situation[higher_points_square][1] == 'none':
				#This means that the square was not occupied before. I just found the ending square of the bishop.
				ending_square = higher_points_square
				self.found = True

			elif self.live_chessboard_situation[higher_points_square][1] == self.color and higher_piece_color == self.opposite_color:
				#The piece has been captured.
				ending_square = higher_points_square
				self.found = True
		else:
			ending_square = 'none'

		if not self.found: #If the end square has not been found again, it means that the pcl processing has not worked properly.
			rospy.loginfo('PCL processing has not worked properly. Try processing again.')

		if self.verbose:
			rospy.loginfo('checking_bishop took ' + str(time() - t_bishop) + 's')

		return ending_square, en_passant, en_passant_square, promotion

	def pawn_checking(self, square, chessboard, threshold, squares_boundboxes, color_threshold, is_color_over):
		t_pawn = time()
		print('PAWN CHECKING...')
		if self.verbose:
			print(square)
		#square: square from which the pawn has been moved.
		#global en_passant
		#global en_passant_square

		self.found = False
		promotion = False
		en_passant = False
		en_passant_square = 'none'

		#Save the index of the starting square.
		index = self.squares_to_index[square]

		#Save the row and the column of the square
		for key, values in self.rows_indexes.items():
			if index in values:
				row = key

		for key, values in self.columns_indexes.items():
			if index in values:
				column = key

		if row == self.second_row: #If the starting row of the pawn is the second row, his move will be a promotion
			promotion = True

		#Check the next square in the column to see if it's occupied.
		self.columns_indexes[column].sort() #Sort the squares of the column in ascending order.
		ind = self.columns_indexes[column].index(index) #Save the index of the current square of the piece in the column array.
		next_square = self.columns_indexes[column][ind + 1]

		square_box = self.crop(chessboard, squares_boundboxes[next_square])
		points_number = len(np.asarray(square_box.points))
		if self.verbose:
			print(points_number)
		if points_number > threshold * 0.8: #The square is occupied.
			#Check if the square  was occupied before.
			#Save the name of the current square
			for key, value in self.squares_to_index.items():
				if value == next_square:
					ending_square = key
			if self.live_chessboard_situation[ending_square][0] == 'none':
				self.found = True

		if not self.found and index in self.rows_indexes[self.seventh_row]: #If the pawn is in its starting position, it could be moved straight for two steps. Check this possibility.
			next_square_x2 = self.columns_indexes[column][ind + 2]

			square_box = self.crop(chessboard, squares_boundboxes[next_square_x2])
			points_number = len(np.asarray(square_box.points))
			if self.verbose:
				print('PUNTI. '+str(points_number))
			RGB_colors = square_box.colors
			mean_RGB = self.mean_RGB_points([RGB_colors])
			piece_color = self.recognize_color(mean_RGB, color_threshold, is_color_over)
			if self.verbose:
				print(piece_color)
			if points_number > threshold * 0.7 and piece_color == self.opposite_color: #The square is occupied.
				#Save the name of the current square
				for key, value in self.squares_to_index.items():
					if value == next_square_x2:
						ending_square = key
						if self.verbose:
							print(ending_square)
				self.found = True

		if not self.found: #If the move has not been identified yet, a capture has happened.
			ending_square = 'none'
			#Check the 2 possible squares of capture of the pawn.
			capture_square_1 = index + 7
			capture_square_2 = index + 9

			rows = self.rows_indexes.keys()
			list_rows = list(rows)
			ind_current_row = list_rows.index(row)
			next_row = list_rows[ind_current_row + 1]
			if capture_square_1 in self.rows_indexes[next_row] and capture_square_2 not in self.rows_indexes[next_row]:
				#If the only possibility of caputre is in square_1, the pawn should be there. TODO SHOULD I CHECK ANYWAY?
				#Save the name of the current square.
				for key, value in self.squares_to_index.items():
					if value == capture_square_1:
						ending_square = key
				if self.live_chessboard_situation[ending_square][1] == self.opposite_color:
					#The move was not possible
					ending_square = 'none'
				else:
					#Check if the square was occupied before. If it was not, en-passant happened.
					if self.live_chessboard_situation[ending_square][0] == 'none':
						en_passant = True
						indice = self.squares_to_index[ending_square]
						for key, values in self.columns_indexes.items():
							if indice in values:
								punto_fisso = values.index(indice)
								colonna = key
						en_passant_square_index = self.columns_indexes[colonna][punto_fisso + 1]
						for key, value in self.squares_to_index.items():
							if value == en_passant_square_index:
								en_passant_square = key
				self.found = True
			
			if capture_square_2 in self.rows_indexes[next_row] and capture_square_1 not in self.rows_indexes[next_row]:
				#If the only possibility of caputer is in square_2, the pawn should be there.
				#Save the name of the current square.
				for key, value in self.squares_to_index.items():
					if value == capture_square_2:
						ending_square = key
				if self.live_chessboard_situation[ending_square][1] == self.opposite_color:
					#The move was not possible
					ending_square = 'none'
				else:
					#Check if the square was occupied before. If it was not, en-passant happened.
					if self.live_chessboard_situation[ending_square][0] == 'none':
						en_passant = True
						indice = self.squares_to_index[ending_square]
						for key, values in self.columns_indexes.items():
							if indice in values:
								punto_fisso = values.index(indice)
								colonna = key
						en_passant_square_index = self.columns_indexes[colonna][punto_fisso + 1]
						for key, value in self.squares_to_index.items():
							if value == en_passant_square_index:
								en_passant_square = key
				self.found = True

			if capture_square_2 in self.rows_indexes[next_row] and capture_square_1 in self.rows_indexes[next_row]:
				#If both the squares are possible, check which one of the 2 has a piece in it.
				square_box_2 = self.crop(chessboard, squares_boundboxes[capture_square_2])
				square_box_1 = self.crop(chessboard, squares_boundboxes[capture_square_1])
				points_number_2 = len(np.asarray(square_box_2.points))
				points_number_1 = len(np.asarray(square_box_1.points))
				RGB_colors_2 = square_box_2.colors
				RGB_colors_1 = square_box_1.colors
				mean_RGB_2 = self.mean_RGB_points([RGB_colors_2])
				mean_RGB_1 = self.mean_RGB_points([RGB_colors_1])
				piece_color_2 = self.recognize_color(mean_RGB_2, color_threshold, is_color_over)
				piece_color_1 = self.recognize_color(mean_RGB_1, color_threshold, is_color_over)
				if self.verbose:
					print('DEBUG points 2: '+str(points_number_2))
					print('DEBUG points 1: '+str(points_number_1))
					print('capt square 1: '+str(capture_square_1))
					print('capt square 2: '+str(capture_square_2))
				if points_number_2 > threshold*0.7 and points_number_1 < threshold*0.7: #Square 2 is occupied, square 1 is not.
					#Save the name of the current square
					for key, value in self.squares_to_index.items():
						if value == capture_square_2:
							ending_square = key
					if self.verbose:
						print('capt square 2:'+str(capture_square_2))
						print('end square: '+str(ending_square))
					if self.live_chessboard_situation[ending_square][1] == self.opposite_color:
						#The move was not possible
						ending_square = 'none'
					else:
						#Check if the square was occupied before. If it was not, en-passant happened.
						if self.live_chessboard_situation[ending_square][0] == 'none':
							en_passant = True
							indice = self.squares_to_index[ending_square]
							for key, values in self.columns_indexes.items():
								if indice in values:
									punto_fisso = values.index(indice)
									colonna = key
							en_passant_square_index = self.columns_indexes[colonna][punto_fisso + 1]
							for key, value in self.squares_to_index.items():
								if value == en_passant_square_index:
									en_passant_square = key

				if points_number_1 > threshold*0.7 and points_number_2 < threshold*0.7: #Square 1 is occupied, square 2 is not.
					#Save the name of the current square
					for key, value in self.squares_to_index.items():
						if value == capture_square_1:
							ending_square = key
					if self.verbose:
						print('capt square 1:'+str(capture_square_1))
						print('end square: '+str(ending_square))
					if self.live_chessboard_situation[ending_square][1] == self.opposite_color:
						#The move was not possible
						ending_square = 'none'
					else:
						#Check if the square was occupied before. If it was not, en-passant happened.
						if self.live_chessboard_situation[ending_square][0] == 'none':
							en_passant = True
							indice = self.squares_to_index[ending_square]
							for key, values in self.columns_indexes.items():
								if indice in values:
									punto_fisso = values.index(indice)
									colonna = key
							en_passant_square_index = self.columns_indexes[colonna][punto_fisso + 1]
							for key, value in self.squares_to_index.items():
								if value == en_passant_square_index:
									en_passant_square = key

				if points_number_2 > threshold*0.7 and points_number_1 > threshold*0.7: #Both square 1 and 2 are occupied.
					#Check the color of the pointcloud.
					if piece_color_1 == self.color and piece_color_2 == self.opposite_color:
						#The ending square of the pawn is square 2.
						for key, value in self.squares_to_index.items():
							if value == capture_square_2:
								ending_square = key
						if self.live_chessboard_situation[ending_square][1] == self.opposite_color:
							#The move was not possible
							ending_square = 'none'
						else:
							#Check if the square was occupied before. If it was not, en-passant happened.
							if self.live_chessboard_situation[ending_square][0] == 'none':
								en_passant = True
								indice = self.squares_to_index[ending_square]
								for key, values in self.columns_indexes.items():
									if indice in values:
										punto_fisso = values.index(indice)
										colonna = key
								en_passant_square_index = self.columns_indexes[colonna][punto_fisso + 1]
								for key, value in self.squares_to_index.items():
									if value == en_passant_square_index:
										en_passant_square = key

					if piece_color_2 == self.color and piece_color_1 == self.opposite_color:
						#The ending square of the pawn is square 1.
						for key, value in self.squares_to_index.items():
							if value == capture_square_1:
								ending_square = key
						if self.live_chessboard_situation[ending_square][1] == self.opposite_color:
							#The move was not possible
							ending_square = 'none'
						else:
							#Check if the square was occupied before. If it was not, en-passant happened.
							if self.live_chessboard_situation[ending_square][0] == 'none':
								en_passant = True
								indice = self.squares_to_index[ending_square]
								for key, values in self.columns_indexes.items():
									if indice in values:
										punto_fisso = values.index(indice)
										colonna = key
								en_passant_square_index = self.columns_indexes[colonna][punto_fisso + 1]
								for key, value in self.squares_to_index.items():
									if value == en_passant_square_index:
										en_passant_square = key

					if piece_color_1 == self.opposite_color and piece_color_2 == self.opposite_color:
						#Both the ending squares are occupied by pieces of the opposite color. I need to check the situation before.
						#Save the square 1 name.
						for key, value in self.squares_to_index.items():
							if value == capture_square_1:
								square_1 = key
						#Save the square 2 name.
						for key, value in self.squares_to_index.items():
							if value == capture_square_1:
								square_2 = key
						if self.live_chessboard_situation[square_1][1] == self.opposite_color or self.live_chessboard_situation[square_1][1] == 'none':
							#In this case the pawn has captured in square 2. En-passant could have happened.
							ending_square = square_2
							if self.live_chessboard_situation[ending_square][1] == self.opposite_color:
								#The move was not possible
								ending_square = 'none'
							else:
								#Check if the square was occupied before. If it was not, en-passant happened.
								if self.live_chessboard_situation[ending_square] == 'none':
									en_passant = True
									indice = self.squares_to_index[ending_square]
									for key, values in self.columns_indexes.items():
										if indice in values:
											punto_fisso = values.index(indice)
											colonna = key
									en_passant_square_index = self.columns_indexes[colonna][punto_fisso + 1]
									for key, value in self.squares_to_index.items():
										if value == en_passant_square_index:
											en_passant_square = key

						if self.live_chessboard_situation[square_2][1] == self.opposite_color or self.live_chessboard_situation[square_1][1] == 'none':
							#In this case the pawn has captured in square 1. En-passant could have happened.
							ending_square = square_1
							if self.live_chessboard_situation[ending_square][1] == self.opposite_color:
								#The move was not possible
								ending_square = 'none'
							else:
								#Check if the square was occupied before. If it was not, en-passant happened.
								if self.live_chessboard_situation[ending_square][0] == 'none':
									en_passant = True
									indice = self.squares_to_index[ending_square]
									for key, values in self.columns_indexes.items():
										if indice in values:
											punto_fisso = values.index(indice)
											colonna = key
									en_passant_square_index = self.columns_indexes[colonna][punto_fisso + 1]
									for key, value in self.squares_to_index.items():
										if value == en_passant_square_index:
											en_passant_square = key
				self.found = True

		if not self.found: #If the end square has not been found again, it means that the pcl processing has not worked properly.
			rospy.loginfo('PCL processing has not worked properly. Try processing again.')
			ending_square = 'none'

		if self.verbose:
			rospy.loginfo('checking_pawn took ' + str(time() - t_pawn) + 's')

		return ending_square, en_passant, en_passant_square, promotion

	def calibrate_points(self):
		#Function to estabilish how many points in the boundbox can be considered enough to decide that there is a piece there.
		#This should be used when pieces are in the starting position.

		starting_pcl = o3d.io.read_point_cloud(pointcloud_for_calibration) #PointCloud with information for calibration.

		start_nan_flag = False
		start_bottom_left_flag = False
		start_bottom_right_flag = False
		start_top_left_flag = False
		start_top_right_flag = False

		starting_squares_matrix = self.get_points_in_matrix_from_pointstamped(self.squares_centers) #To use if I'm using the original points in the eyes reference frame.
		starting_vertices_matrix = self.get_points_in_matrix_from_pointstamped(self.vertices) #To use if I'm usign the original points in the eyes reference frame.
		starting_squares_pointcloud = self.create_pointcloud(starting_squares_matrix, [1, 0, 0]) #Creation of a red PointCloud of the squares centers computed segmenting the chessboard.
		starting_vertices_pointcloud = self.create_pointcloud(starting_vertices_matrix, [0, 1, 0]) #Creation of a red PointCloud of the vertices computed segmenting the chessboard.
		#Normals estimation
		starting_normals = self.normals(starting_squares_pointcloud)
		#Save the coordinates of the centrale squares to compute the coordinates of the center.
		starting_d4 = starting_squares_pointcloud.points[35]
		starting_e4 = starting_squares_pointcloud.points[36]
		starting_d5 = starting_squares_pointcloud.points[27]
		starting_center = [(starting_e4[0] + starting_d5[0])/2, (starting_d5[1] + starting_d4[1])/2, (starting_d4[2] + starting_d5[2] + starting_e4[2])/3]
		#Save the bottom left vertex and the length and width of the chessboard:
		starting_length = 0
		starting_width = 0
		print('CHESSBOARD VERTICES: ' + str(starting_vertices_matrix))
		for point in starting_vertices_matrix:
			if (point[0] < 0 and point[1] > 0):
				starting_bottom_left_vertex = point
				start_bottom_left_flag = True
			if (point[0] < 0 and point[1] < 0):
				starting_bottom_right_vertex = point
				start_bottom_right_flag = True
			if (point[0] > 0 and point[1] > 0):
				starting_top_left_vertex = point
				start_top_left_flag = True
			if (point[0] > 0 and point[1] < 0):
				starting_top_right_vertex = point
				start_top_right_flag = True
			if np.isnan(point[0]):
				start_nan_flag = True
				rospy.logwarn('There is a nan in the computed vertices')
		if start_nan_flag == False:
			starting_length = starting_bottom_left_vertex[1] - starting_bottom_right_vertex[1]
			starting_width = starting_top_left_vertex[0] - starting_bottom_left_vertex[0]
		elif start_nan_flag == True:
			if start_bottom_left_flag and start_bottom_right_flag:
				starting_length = starting_bottom_left_vertex[1] - starting_bottom_right_vertex[1]
			elif start_top_left_flag and start_top_right_flag and start_bottom_left_flag:
				offset = starting_bottom_left_vertex[1] - starting_top_left_vertex[1]
				starting_length = starting_top_left_vertex[1] - starting_top_right_vertex[1] + (offset * 2)
			elif start_top_left_flag and start_top_right_flag and start_bottom_right_flag:
				offset = starting_top_right_vertex[1] - starting_bottom_right_vertex[1]
				starting_length = starting_top_left_vertex[1] - starting_top_right_vertex[1] + (offset * 2)
			elif start_top_left_flag and start_bottom_right_flag:
				starting_length = (starting_top_left_vertex[1] - starting_bottom_right_vertex[1]) + 0.05
			elif start_top_right_flag and start_bottom_left_flag:
				starting_length = (starting_bottom_left_vertex[1] - starting_top_right_vertex[1]) + 0.05

			if start_top_left_flag and start_bottom_left_flag:
				starting_width = starting_top_left_vertex[0] - starting_bottom_left_vertex[0]
			elif start_top_right_flag and start_bottom_right_flag:
				starting_width = starting_top_right_vertex[0] - starting_bottom_right_vertex[0]

			if starting_length == 0:
				starting_length = starting_width + 0.05

			if starting_width == 0:
				starting_width = starting_length

			if self.verbose:
				print(starting_width)
				print(starting_length)

		#Rotation matrix
		if start_bottom_left_flag:
			starting_R, starting_reference_normal = self.get_rot_matrix(starting_pcl, starting_normals, starting_bottom_left_vertex)
		elif start_bottom_right_flag:
			#starting_bottom_left_vertex = starting_bottom_right_vertex
			#starting_bottom_left_vertex[1] = starting_bottom_right_vertex[1] + starting_length
			starting_R, starting_reference_normal = self.get_rot_matrix(starting_pcl, starting_normals, starting_bottom_right_vertex)

		#Create a bounding box around the chessboard
		starting_bounding_box_chessboard, starting_frame = self.create_bounding_box(starting_pcl, starting_length, starting_width, starting_R, starting_center, starting_reference_normal, 0.25, 0.15, 0)
		#o3d.visualization.draw_geometries([starting_pcl, starting_bounding_box_chessboard])
		#Crop the PointCloud to extract the chessboard
		starting_chessboard = self.crop(starting_pcl, starting_bounding_box_chessboard)
		#Compute the mean distance between two consecutive squares in the y direction and in the x direction
		sum_width = 0
		sum_length = 0
		for i in range(1, 7):
			sum_width = sum_width + abs(starting_squares_pointcloud.points[self.rows_indexes['row8'][i]][0] - starting_squares_pointcloud.points[self.rows_indexes['row8'][i-1]][0])
			sum_width = sum_width + abs(starting_squares_pointcloud.points[self.rows_indexes['row5'][i]][0] - starting_squares_pointcloud.points[self.rows_indexes['row5'][i-1]][0])
			sum_width = sum_width + abs(starting_squares_pointcloud.points[self.rows_indexes['row1'][i]][0] - starting_squares_pointcloud.points[self.rows_indexes['row1'][i-1]][0])
			sum_length = sum_length + abs(starting_squares_pointcloud.points[self.columns_indexes['columnA'][i]][1] - starting_squares_pointcloud.points[self.columns_indexes['columnA'][i-1]][1])
			sum_length = sum_length + abs(starting_squares_pointcloud.points[self.columns_indexes['columnD'][i]][1] - starting_squares_pointcloud.points[self.columns_indexes['columnD'][i-1]][1])
			sum_length = sum_length + abs(starting_squares_pointcloud.points[self.columns_indexes['columnH'][i]][1] - starting_squares_pointcloud.points[self.columns_indexes['columnH'][i-1]][1])

		mean_y_dist = sum_width / 21
		mean_x_dist = sum_length / 21
		
		#Create bounding boxes for each square
		squares_bound_boxes = []
		squares_frames = []
		for squares in starting_squares_pointcloud.points:
			bounding_box_square, frame = self.create_bounding_box(starting_chessboard, mean_y_dist, mean_x_dist, starting_R, squares, starting_reference_normal, 0.05, 0.02, 0.035)
			squares_bound_boxes.append(bounding_box_square)
			squares_frames.append(frame)
		#o3d.visualization.draw_geometries([starting_chessboard, squares_bound_boxes[0], squares_frames[0]])

		#Save the squares that are currently occupied.
		self.currently_occupied = []
		self.opponent_occupied = []
		self.tiago_occupied = []
		for square in self.starting_chessboard_situation_complete:
			if self.starting_chessboard_situation_complete[square][1] == self.opposite_color:
				self.currently_occupied.append(square)
				self.opponent_occupied.append(square)

		for square in self.starting_chessboard_situation_complete:
			if self.starting_chessboard_situation_complete[square][1] == self.color:
				self.currently_occupied.append(square)
				self.tiago_occupied.append(square)

		#Save the squares that are currently not occupied.
		self.currently_not_occupied = []
		for square in self.starting_chessboard_situation_complete:
			if self.starting_chessboard_situation_complete[square][1] == 'none':
				self.currently_not_occupied.append(square)

		#Check the squares occupied and compute the mean of the point contained in their bounding boxes.
		mean_points_occupied = 0
		tiago_colors = []
		opponent_colors = []
		for occupied in self.currently_occupied:
			square = self.crop(starting_chessboard, squares_bound_boxes[self.squares_to_index[occupied]])
			points_number = len(np.asarray(square.points))
			mean_points_occupied = mean_points_occupied + points_number
			if occupied in self.tiago_occupied:
				tiago_colors.append(square.colors)
			elif occupied in self.opponent_occupied:
				#print('OPPONENT OCCUPIED: '+str(self.opponent_occupied))###
				opponent_colors.append(square.colors)
		mean_points_occupied = mean_points_occupied // len(self.currently_occupied)
		#Compute the mean of the R, G and B values for white and black pieces.
		color_RGB = self.mean_RGB_points(tiago_colors)
		opponent_color_RGB = self.mean_RGB_points(opponent_colors)

		mean_points_not_occupied = 0
		#Check the squares not occupied and compute the mean of the point contained in their bounding boxes.
		for not_occupied in self.currently_not_occupied:
			square = self.crop(starting_chessboard, squares_bound_boxes[self.squares_to_index[not_occupied]])
			points_number = len(np.asarray(square.points))
			mean_points_not_occupied = mean_points_not_occupied + points_number
		mean_points_not_occupied = mean_points_not_occupied // len(self.currently_not_occupied)

		return mean_points_occupied, mean_points_not_occupied, color_RGB, opponent_color_RGB

	def mean_RGB_points(self, points_colors):
		#Function to compute the mean of the RGB values of the points of white pieces' pointclouds and of black poeces' pointclouds.
		#point_colors: list of lists. Each list corresponds to one piece, and it contains triplets of RGB values corresponding to each point of the piece.
		sumR = 0
		sumG = 0
		sumB = 0
		point_count = 0
		for piece in points_colors:
			for point in piece:
				sumR += point[0]
				sumG += point[1]
				sumB += point[2]
				point_count += 1
		if point_count != 0:
			meanR = sumR / point_count
			meanG = sumG / point_count
			meanB = sumB / point_count
		else:
			meanR = 0
			meanG = 0
			meanB = 0
		mean_RGB = [meanR, meanG, meanB]

		return mean_RGB

	def recognize_color(self, mean_RGB, threshold, is_color_over):
		#Function to recognize if a piece id white or black.
		#threshold: array of three RGB values that are the threshold to consider a piece white or black.
		#is_color_over: array of three True/False flags that tells if TIAGo's color pieces' RGB values are over or under the threshold.
		#mean_RGB: array of the three RGB values corresponding to the mean of the RGB values of the point of the considered piece.
	
		RGB = [False, False, False]
		if mean_RGB[0] > threshold[0]:
				RGB[0] = True
		if mean_RGB[1] > threshold[1]:
				RGB[1] = True
		if mean_RGB[2] > threshold[2]:
				RGB[2] = True
		opposite_RGB = [not RGB[0], not RGB[1], not RGB[2]]
		if RGB == is_color_over:
			piece_color = self.color
		elif opposite_RGB == is_color_over:
			piece_color = self.opposite_color
		else:
			num = 0
			count = 0
			for boolean in RGB:
				if boolean == is_color_over[num]:
					count = count + 1
				num = num + 1
			if count >= 2:
				piece_color = self.color
			else:
				piece_color = self.opposite_color

		return piece_color
		

	def look_for_pieces(self, pcl, chessboard, squares_boundboxes, threshold, color_threshold, is_color_over):
		#Function to look for the presence of pieces over the chessboard squares.
		#chessboard: PointCloud of the isolated chessboard.
		#squares_boundboxes: a list containing the 64 bounding boxes to isolate each square and the things inside them.
		#threshold: threshold of points to consider a square occupied or not

		castle = False
		self.castle_squares_found = 'none'

		#Save the live chessboard situation
		with open(self.dir_live_chessboard_situation, 'rb') as live_file:
			self.live_chessboard_situation = yaml.load(live_file.read(), Loader = yaml.Loader)
		#Save the squares that are currently occupied by opposite's pieces
		self.currently_occupied = []
		for square in self.live_chessboard_situation:
			if self.live_chessboard_situation[square][1] == self.opposite_color:
				self.currently_occupied.append(square)

		#Check the squares occupied by the opposite's pieces.
		checking_flag = True
		count_empty = 0
		count_occupied = 0
		check_squares = []
		check_pieces = []
		check_columns = []
		check_promotion = []
		less_points = 2000
		less_points_not_pawn = 2000
		less_points_piece = 'none'
		less_points_square = 'none'
		second_less_points_piece = 'none'
		second_less_points_square = 'none'
		less_points_not_pawn_piece = 'none'
		less_points_not_pawn_square = 'none'
		indices_occupied = []
		if self.verbose:
			print('CURRENTLY OCCUPIED: '+str(self.currently_occupied))
		
		while checking_flag:
			for occupied in self.currently_occupied:
				if self.verbose:
					print('THRESHOLD: '+str(threshold))
				square = self.crop(pcl, squares_boundboxes[self.squares_to_index[occupied]])
				#o3d.visualization.draw_geometries([chessboard, squares_boundboxes[self.squares_to_index[occupied]]])
				points_number = len(np.asarray(square.points))
				moved_piece_debug = self.live_chessboard_situation[occupied][0]
				if self.verbose:
					print('Moved piece debug: '+str(moved_piece_debug))
					print('Points: '+str(points_number))

				if points_number < less_points:
					less_points = points_number
					#less_points_piece, second_less_points_piece = (moved_piece_debug, less_point_piece)
					second_less_points_piece = less_points_piece
					less_points_piece = moved_piece_debug
					second_less_points_square = less_points_square
					less_points_square = occupied

				if self.pieces_coordinates[moved_piece_debug][1]['name'] != 'pawn' and points_number < less_points_not_pawn:
					less_points_not_pawn = points_number
					less_points_not_pawn_piece = moved_piece_debug
					less_points_not_pawn_square = occupied

				'''
				if moved_piece_debug == 'king_e8' or moved_piece_debug == 'rook_h8':
					o3d.visualization.draw_geometries([square])
				'''

				#Save the index of the starting square.
				index = self.squares_to_index[occupied]
				for key, values in self.columns_indexes.items():
					if index in values:
						column = key
				if points_number < (threshold * 0.7): #* 0.5): #If the points over the square are less than a threshold, it means that the piece have been moved. take the 50% of the threshold to exclude the pieces covered by pieces in front of them.
					move_square = occupied
					moved_piece = self.live_chessboard_situation[move_square][0]
					checking_flag = False
					check_squares.append(move_square)
					check_pieces.append(moved_piece)
					check_columns.append(column)
					count_empty += 1
					if self.verbose:
						print('Moved piece: '+str(moved_piece))
						print('Points number : '+str(points_number))

			if self.verbose:
				print('CHECK SQUARES: '+str(check_squares))
				print('CHECK PIECES: '+str(check_pieces))
				print('Less points piece: '+str(less_points_piece))

			self.conta_torri = 0
			for piece in check_pieces:
				if self.pieces_coordinates[piece][1]['name'] == 'rook':
					self.conta_torri += 1

			if count_empty > 1:
				check_ending_square = []
				cast_sq = ['none', 'none']
				conteggio = 0
				is_castle = 0
				for k in check_squares:
					if k in self.castle_squares:
						is_castle += 1
						if k == self.castle_squares[1]:
							cast_sq[1] = k
						else:
							cast_sq[0] = k
				#If the empty squares are 2 and these squares are the castle squares, castle happened
				if is_castle == 2:
					castle = True
					if cast_sq[0] == 'h8' or cast_sq[1] == 'h8':
						self.castle_squares_found = ['g8', 'f8']
					elif cast_sq[0] == 'a8' or cast_sq[1] == 'a8':
						self.castle_squares_found = ['c8', 'd8']
					elif cast_sq[0] == 'h1' or cast_sq[1] == 'h1':
						self.castle_squares_found = ['g1', 'f1']
					elif cast_sq[0] == 'a1' or cast_sq[1] == 'a1':
						self.castle_squares_found = ['c1', 'd1']

					#self.castle_squares_found = cast_sq
					checking_flag = False
					en_passant = False
					en_passant_square = 'none'
					promotion = False
					promoted_piece = 'none'
					ending_square = 'castle'
					move_square = 'castle'
					moved_piece = 'castle'
					ending_square = 'castle'

				#If castle didn't happen, continue checking the situation
				if not castle:
					for piece in check_pieces:
						if self.pieces_coordinates[piece][1]['name'] == 'pawn':
							ending_square, en_passant, en_passant_square, promotion = self.pawn_checking(check_squares[conteggio], chessboard, threshold, squares_boundboxes, color_threshold, is_color_over)
						elif self.pieces_coordinates[piece][1]['name'] == 'knight':
							ending_square, en_passant, en_passant_square, promotion = self.knight_checking(check_squares[conteggio], chessboard, threshold, squares_boundboxes, color_threshold, is_color_over)
						elif self.pieces_coordinates[piece][1]['name'] == 'bishop':
							ending_square, en_passant, en_passant_square, promotion = self.bishop_checking(check_squares[conteggio], chessboard, threshold, squares_boundboxes, color_threshold, is_color_over)
						elif self.pieces_coordinates[piece][1]['name'] == 'rook':
							ending_square, en_passant, en_passant_square, promotion = self.rook_checking(check_squares[conteggio], chessboard, threshold, squares_boundboxes, color_threshold, is_color_over)
						elif self.pieces_coordinates[piece][1]['name'] == 'queen':
							ending_square, en_passant, en_passant_square, promotion = self.queen_checking(check_squares[conteggio], chessboard, threshold, squares_boundboxes, color_threshold, is_color_over)
						elif self.pieces_coordinates[piece][1]['name'] == 'king':
							ending_square, en_passant, en_passant_square, promotion = self.king_checking(check_squares[conteggio], chessboard, threshold, squares_boundboxes, color_threshold, is_color_over)
						conteggio += 1
						check_ending_square.append(ending_square)
						check_promotion.append(promotion)
						'''
						if self.found:
							break
						'''

					for i in range(0, len(check_ending_square)):
						if check_ending_square[i] == 'castle':
							castle = True

						if check_ending_square[i] not in check_columns[i]: #Controllo per evitare i pedoni coperti da altri pezzi
							moved_piece = check_pieces[i]
							move_square = check_squares[i]

						if check_ending_square[i] != 'none':
							count_occupied += 1
							index_occupied = i
							indices_occupied.append(index_occupied)
					if count_occupied == 0:
						rospy.logwarn('The opponent move has not been identified')
					elif count_occupied == 1:
						moved_piece = check_pieces[index_occupied]
						move_square = check_squares[index_occupied]
						ending_square = check_ending_square[index_occupied]
						promotion = check_promotion[index_occupied]
					elif count_occupied > 1:
						if less_points_piece in check_pieces and check_pieces.index(less_points_piece) in indices_occupied:
							ind = check_pieces.index(less_points_piece)
							moved_piece = less_points_piece
							move_square = less_points_square
							ending_square = check_ending_square[ind]
							promotion = check_promotion[ind]
						elif second_less_points_piece in check_pieces:
							ind = check_pieces.index(second_less_points_piece)
							moved_piece = second_less_points_piece
							move_square = second_less_points_square
							ending_square = check_ending_square[ind]
							promotion = check_promotion[ind]

					checking_flag = False

			elif count_empty == 0:
				moved_piece = less_points_piece
				move_square = less_points_square
				if self.pieces_coordinates[less_points_piece][1]['name'] == 'pawn':
					#Check if the pawn has a piece in front of him
					#Save the index of the starting square.
					index = self.squares_to_index[move_square]

					#Save the row and the column of the square
					for key, values in self.rows_indexes.items():
						if index in values:
							row = key

					for key, values in self.columns_indexes.items():
						if index in values:
							column = key
					'''
					#Check the next square in the column to see if it's occupied.
					self.columns_indexes[column].sort() #Sort the squares of the column in ascending order.
					ind = self.columns_indexes[column].index(index) #Save the index of the current square of the piece in the column array.
					next_square = self.columns_indexes[column][ind + 1]
					print('next square *********'+str(next_square))

					square_box = self.crop(chessboard, squares_boundboxes[next_square])
					points_number = len(np.asarray(square_box.points))
					if points_number > threshold: #The square is occupied.
						#Since the square is occupied, the pawn could be hidden by another piece.
						moved_piece = less_points_not_pawn_piece
						move_square = less_points_not_pawn_square
					'''
				checking_flag = False
			

			elif count_empty == 1:
				if self.pieces_coordinates[moved_piece][1]['name'] == 'pawn':
					#Check if the pawn has a piece in front of him.
					#Save the index of the starting square.
					index = self.squares_to_index[move_square]

					#Save the row and the column of the square
					for key, values in self.rows_indexes.items():
						if index in values:
							row = key

					for key, values in self.columns_indexes.items():
						if index in values:
							column = key

					#Check the next square in the column to see if it's occupied.
					self.columns_indexes[column].sort() #Sort the squares of the column in ascending order.
					ind = self.columns_indexes[column].index(index) #Save the index of the current square of the piece in the column array.
					next_square = self.columns_indexes[column][ind + 1]

					square_box = self.crop(chessboard, squares_boundboxes[next_square])
					points_number = len(np.asarray(square_box.points))
					if points_number > threshold: #The square is occupied.
						#Since the square is occupied, the pawn could be hidden by another piece.
						moved_piece = less_points_piece
						move_square = less_points_square
						if self.pieces_coordinates[less_points_piece][1]['name'] == 'pawn':
							#Check if the pawn has a piece in front of him
							#Save the index of the starting square.
							index = self.squares_to_index[move_square]

							#Save the row and the column of the square
							for key, values in self.rows_indexes.items():
								if index in values:
									row = key

							for key, values in self.columns_indexes.items():
								if index in values:
									column = key
							'''
							#Check the next square in the column to see if it's occupied. #DA COMMENTARE (INIZIO)
							self.columns_indexes[column].sort() #Sort the squares of the column in ascending order.
							ind = self.columns_indexes[column].index(index) #Save the index of the current square of the piece in the column array.
							next_square = self.columns_indexes[column][ind + 1]

							square_box = self.crop(chessboard, squares_boundboxes[next_square])
							points_number = len(np.asarray(square_box.points))
							if points_number > threshold: #The square is occupied.
								#Since the square is occupied, the pawn could be hidden by another piece.
								moved_piece = less_points_not_pawn_piece
								move_square = less_points_not_pawn_square  #DA COMMENTARE (FINE)
							'''
				checking_flag = False
			

		if not castle and count_empty < 2: #and checking_flag: #AGGIUNTA LA PARTE DOPO L'AND
			if self.pieces_coordinates[moved_piece][1]['name'] == 'pawn':
				ending_square, en_passant, en_passant_square, promotion = self.pawn_checking(move_square, chessboard, threshold, squares_boundboxes, color_threshold, is_color_over)
			elif self.pieces_coordinates[moved_piece][1]['name'] == 'knight':
				ending_square, en_passant, en_passant_square, promotion = self.knight_checking(move_square, chessboard, threshold, squares_boundboxes, color_threshold, is_color_over)
			elif self.pieces_coordinates[moved_piece][1]['name'] == 'bishop':
				ending_square, en_passant, en_passant_square, promotion = self.bishop_checking(move_square, chessboard, threshold, squares_boundboxes, color_threshold, is_color_over)
			elif self.pieces_coordinates[moved_piece][1]['name'] == 'rook':
				ending_square, en_passant, en_passant_square, promotion = self.rook_checking(move_square, chessboard, threshold, squares_boundboxes, color_threshold, is_color_over)
			elif self.pieces_coordinates[moved_piece][1]['name'] == 'queen':
				ending_square, en_passant, en_passant_square, promotion = self.queen_checking(move_square, chessboard, threshold, squares_boundboxes, color_threshold, is_color_over)
			elif self.pieces_coordinates[moved_piece][1]['name'] == 'king':
				ending_square, en_passant, en_passant_square, promotion = self.king_checking(move_square, chessboard, threshold, squares_boundboxes, color_threshold, is_color_over)

			if ending_square == 'none':
				moved_piece = second_less_points_piece
				move_square = second_less_points_square
				if self.pieces_coordinates[moved_piece][1]['name'] == 'pawn':
					ending_square, en_passant, en_passant_square, promotion = self.pawn_checking(move_square, chessboard, threshold, squares_boundboxes, color_threshold, is_color_over)
				elif self.pieces_coordinates[moved_piece][1]['name'] == 'knight':
					ending_square, en_passant, en_passant_square, promotion = self.knight_checking(move_square, chessboard, threshold, squares_boundboxes, color_threshold, is_color_over)
				elif self.pieces_coordinates[moved_piece][1]['name'] == 'bishop':
					ending_square, en_passant, en_passant_square, promotion = self.bishop_checking(move_square, chessboard, threshold, squares_boundboxes, color_threshold, is_color_over)
				elif self.pieces_coordinates[moved_piece][1]['name'] == 'rook':
					ending_square, en_passant, en_passant_square, promotion = self.rook_checking(move_square, chessboard, threshold, squares_boundboxes, color_threshold, is_color_over)
				elif self.pieces_coordinates[moved_piece][1]['name'] == 'queen':
					ending_square, en_passant, en_passant_square, promotion = self.queen_checking(move_square, chessboard, threshold, squares_boundboxes, color_threshold, is_color_over)
				elif self.pieces_coordinates[moved_piece][1]['name'] == 'king':
					ending_square, en_passant, en_passant_square, promotion = self.king_checking(move_square, chessboard, threshold, squares_boundboxes, color_threshold, is_color_over)
			if promotion:
				state_publisher.publish(16) #Change status to the status of asking the opponent promoted piece.
				promotion_happened_publisher.publish(self.opposite_color)
				rospy.sleep(2)
			'''
			if en_passant:
				#If en-passant happened, send a message to correctly update the GUI.
				en_passant_square_publisher.publish(en_passant_square)
			'''

		elif castle:
			#If castle happened, send a message to correctly update the GUI.
			castle_square_publisher.publish(self.castle_squares_found[0]) #Publish one square of the castle.

		elif promotion:
			state_publisher.publish(16) #Change status to the status of asking the opponent promoted piece.
			promotion_happened_publisher.publish(self.opposite_color)
			rospy.sleep(2)

		if ending_square == 'castle':
			#If castle happened, send a message to correctly update the GUI.
			castle_square_publisher.publish(self.castle_squares_found[0]) #Publish one square of the castle

		if en_passant:
			#If en-passant happened, send a message to correctly update the GUI.
			en_passant_square_publisher.publish(en_passant_square)


		return moved_piece, move_square, ending_square

	def moved_piece_identification(self, pcl):
		t_crop = time()
		squares_matrix = self.get_points_in_matrix_from_pointstamped(self.squares_centers) #To use if I'm using the original points in the eyes reference frame.
		vertices_matrix = self.get_points_in_matrix_from_pointstamped(self.vertices) #To use if I'm usign the original points in the eyes reference frame.
		squares_pointcloud = self.create_pointcloud(squares_matrix, [1, 0, 0]) #Creation of a red PointCloud of the squares centers computed segmenting the chessboard.
		vertices_pointcloud = self.create_pointcloud(vertices_matrix, [0, 1, 0]) #Creation of a green PointCloud of the vertices computed segmenting the chessboard.

		nan_flag = False
		bottom_left_flag = False
		bottom_right_flag = False
		top_left_flag = False
		top_right_flag = False

		#Normals estimation
		t_norm = time()
		normals = self.normals(squares_pointcloud)
		if self.verbose:
			rospy.loginfo('Normals estimation took ' +str(time() -  t_norm) + 's')

		#Save the coordinates of the central squares to compute the coordinates of the center.
		d4 = squares_pointcloud.points[35]
		e4 = squares_pointcloud.points[36]
		d5 = squares_pointcloud.points[27]
		center = [(e4[0] + d5[0])/2, (d5[1] + d4[1])/2, (d4[2] + d5[2] + e4[2])/3]

		#Save the bottom left vertex and the length and width of the chessboard:
		length = 0
		width = 0
		for point in vertices_matrix:
			if (point[0] < 0 and point[1] > 0):
				bottom_left_vertex = point
				bottom_left_flag = True
			if (point[0] < 0 and point[1] < 0):
				bottom_right_vertex = point
				bottom_right_flag = True
			if (point[0] > 0 and point[1] > 0):
				top_left_vertex = point
				top_left_flag = True
			if (point[0] > 0 and point[1] < 0):
				top_right_vertex = point
				top_right_flag = True
			if np.isnan(point[0]):
				nan_flag = True
				#rospy.logwarn('There is a nan in the computed vertices')
		if nan_flag == False:
			length = bottom_left_vertex[1] - bottom_right_vertex[1]
			width = top_left_vertex[0] - bottom_left_vertex[0]
		elif nan_flag == True:
			if bottom_left_flag and bottom_right_flag:
				length = bottom_left_vertex[1] - bottom_right_vertex[1]
			elif top_left_flag and top_right_flag and bottom_left_flag:
				offset = bottom_left_vertex[1] - top_left_vertex[1]
				length = top_left_vertex[1] - top_right_vertex[1] + (offset * 2)
			elif top_left_flag and top_right_flag and bottom_right_flag:
				offset = top_right_vertex[1] - bottom_right_vertex[1]
				length = top_left_vertex[1] - top_right_vertex[1] + (offset * 2)
			elif top_left_flag and bottom_right_flag:
				length = (top_left_vertex[1] - bottom_right_vertex[1]) + 0.05
			elif top_right_flag and bottom_left_flag:
				length = (bottom_left_vertex[1] - top_right_vertex[1]) + 0.05

			if top_left_flag and bottom_left_flag:
				width = top_left_vertex[0] - bottom_left_vertex[0]
			elif top_right_flag and bottom_right_flag:
				width = top_right_vertex[0] - bottom_right_vertex[0]

			if length == 0:
				length = width + 0.05 ###DA VERIFICARE

			if width == 0:
				width = length ###DA VERIFICARE

		#Rotation matrix
		if bottom_left_flag:
			R, reference_normal = self.get_rot_matrix(pcl, normals, bottom_left_vertex)
		elif bottom_right_flag:
			#starting_bottom_left_vertex = starting_bottom_right_vertex
			#starting_bottom_left_vertex[1] = starting_bottom_right_vertex[1] + starting_length
			R, reference_normal = self.get_rot_matrix(pcl, normals, bottom_right_vertex)

		#Create a bounding box around the chessboard
		bounding_box_chessboard, frame = self.create_bounding_box(pcl, length, width, R, center, reference_normal, 0.25, 0.15, 0)

		#Crop the PointCloud to extract the chessboard
		chessboard = self.crop(pcl, bounding_box_chessboard)

		#Compute the mean distance between two consecutive squares in the y direction and in the x direction
		sum_width = 0
		sum_length = 0
		for i in range(1, 7):
			sum_width = sum_width + abs(squares_pointcloud.points[self.rows_indexes['row8'][i]][0] - squares_pointcloud.points[self.rows_indexes['row8'][i-1]][0])
			sum_width = sum_width + abs(squares_pointcloud.points[self.rows_indexes['row5'][i]][0] - squares_pointcloud.points[self.rows_indexes['row5'][i-1]][0])
			sum_width = sum_width + abs(squares_pointcloud.points[self.rows_indexes['row1'][i]][0] - squares_pointcloud.points[self.rows_indexes['row1'][i-1]][0])
			sum_length = sum_length + abs(squares_pointcloud.points[self.columns_indexes['columnA'][i]][1] - squares_pointcloud.points[self.columns_indexes['columnA'][i-1]][1])
			sum_length = sum_length + abs(squares_pointcloud.points[self.columns_indexes['columnD'][i]][1] - squares_pointcloud.points[self.columns_indexes['columnD'][i-1]][1])
			sum_length = sum_length + abs(squares_pointcloud.points[self.columns_indexes['columnH'][i]][1] - squares_pointcloud.points[self.columns_indexes['columnH'][i-1]][1])

		mean_y_dist = sum_width / 21
		mean_x_dist = sum_length / 21

		#Create bounding boxes for each square
		squares_bound_boxes = []
		squares_frames = []
		for squares in squares_pointcloud.points:
			bounding_box_square, frame = self.create_bounding_box(chessboard, mean_y_dist, mean_x_dist, R, squares, reference_normal, 0.05, 0.02, 0.035)
			squares_bound_boxes.append(bounding_box_square)
			squares_frames.append(frame)


		#if ready:
		with open(occupied_thresh_file, 'rb') as file:
				occupied_threshold = pickle.load(file, encoding = 'latin1')
		with open(color_thresh_file, 'rb') as file:
				color_threshold = pickle.load(file, encoding = 'latin1')
		with open(is_color_over_file, 'rb') as file:
				is_color_over = pickle.load(file, encoding = 'latin1')
		if self.verbose:
			rospy.loginfo('Cropping chessboard pointcloud took' + str(time() - t_crop) + 's.')

		#Look for pieces
		#o3d.visualization.draw_geometries([pcl])
		moved_piece, starting_square, ending_square = self.look_for_pieces(pcl, chessboard, squares_bound_boxes, occupied_threshold, color_threshold, is_color_over)
		print('MOVED PIECE: ' + str(moved_piece))
		print('ENDING SQUARE: ' + str(ending_square))

		return moved_piece, starting_square, ending_square

	def identify_moved_piece(self, data):
		global analysis_not_done_yet
		if data:
			#Understand which piece has been moved
			moved_piece, start_square, end_square = self.moved_piece_identification(self.pcd)

			#Send messages regarding the move executed by the opponent to change the GUI.
			rospy.loginfo('Publishing the opponent move...')
			print('Time for move recognition: '+str(time() - self.t_recognition)+' s')
			opponent_move_start_square_publisher.publish(start_square)
			opponent_move_end_square_publisher.publish(end_square)

			analysis_not_done_yet = False


	def CallbackState(self, data):
		self.state.data = data.data
		global analysis_not_done_yet
		global calibration_not_done_yet
		global occupied_threshold
		global color_threshold
		global is_color_over
		if data.data == 15 and analysis_not_done_yet:
			#occupied_threshold = 6.75
			#color_threshold = [0.6893202700397256, 0.5754423843091373, 0.5045072090399367]
			#is_color_over = [True, True, True]
			print('Waiting for images')
			self.synchronize_rgb_depth()

			'''
			#Understand which piece has been moved
			moved_piece, start_square, end_square = self.moved_piece_identification(pcl)

			#Send messages regarding the move executed by the opponent to change the GUI.
			rospy.loginfo('Publishing the opponent move...')
			opponent_move_start_square_publisher.publish(start_square)
			opponent_move_end_square_publisher.publish(end_square)

			analysis_not_done_yet = False
			'''

		elif data.data == 50 and calibration_not_done_yet:
			#Calibrate to save the optimal points threshold to discriminate when a square is occupied or not. #FORSE NON SERVE
			mean_points_occupied, mean_points_not_occupied, color_RGB, opponent_color_RGB = self.calibrate_points()
			occupied_threshold = ((mean_points_occupied + mean_points_not_occupied) // 2) * 0.75 #Take as threshold, the 75% of the mean of points of the occupied squares and the not occupied ones.  * 0.75
			color_threshold = [((color_RGB[0] + opponent_color_RGB[0]) / 2), ((color_RGB[1] + opponent_color_RGB[1]) / 2), ((color_RGB[2] + opponent_color_RGB[2]) / 2)]
			print('THRESHOLD: ' + str(occupied_threshold))
			print('RGB COLOR THRESHOLD: ' + str(color_threshold))
			is_color_over = [False, False, False]
			if color_RGB[0] > color_threshold[0]:
				is_color_over[0] = True

			if color_RGB[1] > color_threshold[1]:
				is_color_over[1] = True

			if color_RGB[2] > color_threshold[2]:
				is_color_over[2] = True

			calibration_not_done_yet = False

			with open(occupied_thresh_file, "wb") as fout:
				pickle.dump(occupied_threshold, fout)
			with open(color_thresh_file, "wb") as fout:
				pickle.dump(color_threshold, fout)
			with open(is_color_over_file, "wb") as fout:
				pickle.dump(is_color_over, fout)

			state_publisher.publish(9) #DA VERIFICARE SE NON LO PUBBLICA TROPPO PRESTO

		elif data.data == 17:
			analysis_not_done_yet = True #Change back the flag
			

def main():
	rospy.init_node('pcl_processor')
	depth_processor = DepthProcessing()

	#Initialize a subscriber to monitor the state.
	rospy.Subscriber("/state", Int16, depth_processor.CallbackState)

	try:
		rospy.spin()    
	except KeyboardInterrupt:
		print("Shutting down the CV module.")





if __name__ == '__main__':
     main()

