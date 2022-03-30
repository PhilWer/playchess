#!/root/venv/pcd/bin/python3
#Processing of the pointcloud over the chessboard

# Rospy for the subscriber
import rospy
from sensor_msgs.msg import PointCloud2, PointField
#from tiago_playchess.msg import chess_move
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
pointcloud_for_calibration = '/home/silvia/tiago_public_ws/src/tiago_playchess/pcl_depth_synchro/01/cloud01.ply' #'/root/tiago_public_ws/src/tiago_playchess/pcl_depth_synchro/01/cloud01.ply'
imported_pointcloud = '/home/silvia/tiago_public_ws/src/tiago_playchess/pcl_depth_synchro/17/cloud17.ply' #'/root/tiago_public_ws/src/tiago_playchess/pcl_depth_synchro/17/cloud17.ply'
imported_rgb_image = '/home/silvia/tiago_public_ws/src/tiago_playchess/pcl_depth_synchro/17/rgb17.png' #'/root/tiago_public_ws/src/tiago_playchess/pcl_depth_synchro/17/rgb17.png'
imported_depth_image = '/home/silvia/tiago_public_ws/src/tiago_playchess/pcl_depth_synchro/17/depth_norm17.png' #'/root/tiago_public_ws/src/tiago_playchess/pcl_depth_synchro/17/depth_norm17.png'
imported_depth_image_csv = '/home/silvia/tiago_public_ws/src/tiago_playchess/pcl_depth_synchro/17/depth17.csv' #'/root/tiago_public_ws/src/tiago_playchess/pcl_depth_synchro/17/depth17.csv'
imported_chessboard_squares = r'/home/silvia/tiago_public_ws/src/tiago_playchess/config/simul_config_not_transformed.yaml' #'/root/tiago_public_ws/src/tiago_playchess/config/simul_config_not_transformed.yaml'
imported_chessboard_vertices = r'/home/silvia/tiago_public_ws/src/tiago_playchess/config/chessboard_vertices_not_transformed.yaml' #'/root/tiago_public_ws/src/tiago_playchess/config/chessboard_vertices_not_transformed.yaml'
imported_configurations = r'/home/silvia/tiago_public_ws/src/tiago_playchess/scripts/config/simulation_config.yaml' #'/root/tiago_public_ws/src/tiago_playchess/scripts/config/simulation_config.yaml'

class PclProcessing:
#Class containing functions to process chessboard images and detect squares.
	def __init__(self):
		self.verbose = True
		#Import the chessboard squares centers as computed with CV
		with open(imported_chessboard_squares) as file:
			self.squares_centers = yaml.load(file, Loader=yaml.Loader)

		#Import the chessboard vertices as computed with CV
		with open(imported_chessboard_vertices) as file:
			self.vertices = yaml.load(file, Loader=yaml.Loader)

		#Constants
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
			self.castle_squares = ['a8', 'e8', 'h8', 'king_e8', 'rook_a8', 'rook_h8']
			self.castle_ending_squares = ['d8', 'c8', 'g8', 'f8']
			self.seventh_row = 'row7'
			self.second_row = 'row2'
		else:
			self.squares_to_index = cfg.squares_to_index_black
			self.rows_indexes = cfg.rows_black
			self.opposite_color = 'white'
			self.castle_squares = ['a1', 'e1', 'h1', 'king_e1', 'rook_a1', 'rook_h1']
			self.castle_ending_squares = ['d1', 'c1', 'g1', 'f1']
			self.seventh_row = 'row2'
			self.second_row = 'row7'

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

	def create_pointcloud_from_depth(self):
		#Function that creates a PointCloud starting from a matrix of xyz points [nx3 matrix].
		#color: list of rgb values corresponding to each pixel of the image)

		#Import the rgb image to save colors of the points
		self.rgb_im = cv2.imread(imported_rgb_image) 
		self.rgb_im = cv2.cvtColor(self.rgb_im, cv2.COLOR_BGR2RGB) / 255.0 #Convert the image from BGR to RGB

		#save depth data from the csv file
		depth_data = np.genfromtxt(imported_depth_image_csv, delimiter = ',')

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
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(xyz)
		pcd.colors = o3d.utility.Vector3dVector(self.pts_color)

		return pcd

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
			o3d.visualization.draw_geometries([pcl, bounding_box, self_created_frame])
		return bounding_box, self_created_frame

	def crop(self, pcl, bounding_box):
		#Function to crop a PointCloud given a bounding box.

		#Crop the PointCloud
		point_cloud_crop = pcl.crop(bounding_box)

		#Visualize the cropped pointcloud
		#o3d.visualization.draw_geometries([point_cloud_crop, bounding_box])

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
	'''
	def king_checking(self, square, chessboard, threshold, squares_boundboxes, color_threshold, is_color_over):
		#square: square from which the king has been moved.

		self.found = False
		possible_squares = []

		en_passant = False
		en_passant_square = 'none'
		promotion = False
		#Save the index of the starting square.
		index = self.squares_to_index[square]

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
			if points_number > threshold: #The square is occupied.
				RGB_colors = square_box.colors
				mean_RGB = self.mean_RGB_points([RGB_colors])
				piece_color = self.recognize_color(mean_RGB, color_threshold, is_color_over)
				#Check if the square was occupied before too.
				if self.live_chessboard_situation_complete[current_square][1] == 'none':
					#This means that the square was not occupied before. I just found the ending square of the king.
					ending_square = current_square
					#Update the live_chessboard_situation
					self.live_chessboard_situation_complete[current_square][0] = self.live_chessboard_situation_complete[square][0]
					self.live_chessboard_situation_complete[current_square][1] = self.opposite_color
					self.live_chessboard_situation_complete[square][0] = 'none'
					self.live_chessboard_situation_complete[square][1] = 'none'
					self.found = True
					break #TODO DA CAPIRE: QUESTO BREAK ESCE DA QUALE FOR LOOP?

				elif self.live_chessboard_situation_complete[current_square][1] == self.color and piece_color == self.opposite_color:
					#The piece has been captured.
					ending_square = current_square
					#Update the live_chessboard_situation
					self.live_chessboard_situation_complete[current_square][0] = self.live_chessboard_situation_complete[square][0]
					self.live_chessboard_situation_complete[current_square][1] = self.opposite_color
					self.live_chessboard_situation_complete[square][0] = 'none'
					self.live_chessboard_situation_complete[square][1] = 'none'
					self.found = True
					break #TODO DA CAPIRE: QUESTO BREAK ESCE DA QUALE FOR LOOP?

		if not self.found: #If the end square has not been found again, it means that the pcl processing has not worked properly.
			rospy.loginfo('PCL processing has not worked properly. Try processing again.')

		return ending_square, en_passant, en_passant_square, promotion
	'''
	def queen_checking(self, square, chessboard, threshold, squares_boundboxes, color_threshold, is_color_over):
		#square: square from which the queen has been moved.

		self.found = False
		possible_squares = []
		diagonal_squares = []
		count = 1

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

		#Remove from the row and column array the starting square,cause i will not search on that one for the end square.
		row.remove(index)
		column.remove(index)

		possible_squares = np.append(row, column)

		#Save the indexes of the squares on the diagonal of the queen.
		while count < 8:
			if (index + 7 * count) in range(64): 
				possible_squares = np.append(possible_squares, index + 7 * count)
			if (index - 9 * count) in range(64): 
				possible_squares = np.append(possible_squares, index - 9 * count)
			if (index - 7 * count) in range(64): 
				possible_squares = np.append(possible_squares, index - 7 * count)
			if (index + 9 * count) in range(64): 
				possible_squares = np.append(possible_squares, index + 9 * count)
			count += 1

		#Check the reachable squares.
		for element in possible_squares:
			#Save the name of the current square
			for key, value in self.squares_to_index.items():
				if value == element:
					current_square = key
			square_box = self.crop(chessboard, squares_boundboxes[element])
			points_number = len(np.asarray(square_box.points))
			if points_number > threshold: #The square is occupied.
				RGB_colors = square_box.colors
				mean_RGB = self.mean_RGB_points([RGB_colors])
				piece_color = self.recognize_color(mean_RGB, color_threshold, is_color_over)
				#Check if the square was occupied before too.
				if self.live_chessboard_situation_complete[current_square][1] == 'none':
					#This means that the square was not occupied before. I just found the ending square of the bishop.
					ending_square = current_square
					#Update the live_chessboard_situation
					self.live_chessboard_situation_complete[current_square][0] = self.live_chessboard_situation_complete[square][0]
					self.live_chessboard_situation_complete[current_square][1] = self.opposite_color
					self.live_chessboard_situation_complete[square][0] = 'none'
					self.live_chessboard_situation_complete[square][1] = 'none'
					self.found = True
					break #TODO DA CAPIRE: QUESTO BREAK ESCE DA QUALE FOR LOOP?

				elif self.live_chessboard_situation_complete[current_square][1] == self.color and piece_color == self.opposite_color:
					#The piece has been captured.
					ending_square = current_square
					#Update the live_chessboard_situation
					self.live_chessboard_situation_complete[current_square][0] = self.live_chessboard_situation_complete[square][0]
					self.live_chessboard_situation_complete[current_square][1] = self.opposite_color
					self.live_chessboard_situation_complete[square][0] = 'none'
					self.live_chessboard_situation_complete[square][1] = 'none'
					self.found = True
					break #TODO DA CAPIRE: QUESTO BREAK ESCE DA QUALE FOR LOOP?

		if not self.found: #If the end square has not been found again, it means that the pcl processing has not worked properly.
			rospy.loginfo('PCL processing has not worked properly. Try processing again.')

		return ending_square, en_passant, en_passant_square, promotion
	
	def knight_checking(self, square, chessboard, threshold, squares_boundboxes, color_threshold, is_color_over):
		#square: square from which the knight has been moved.

		self.found = False
		possible_squares = []

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
		for element in possible_squares:
			#Save the name of the current square
			for key, value in self.squares_to_index.items():
				if value == element:
					current_square = key
					square_box = self.crop(chessboard, squares_boundboxes[element])
					points_number = len(np.asarray(square_box.points))
					if points_number > threshold: #The square is occupied.
						RGB_colors = square_box.colors
						mean_RGB = self.mean_RGB_points([RGB_colors])
						piece_color = self.recognize_color(mean_RGB, color_threshold, is_color_over)
						#Check if the square was occupied before too.
						if self.live_chessboard_situation_complete[current_square][1] == 'none':
							#This means that the square was not occupied before. I just found the ending square of the knight.
							ending_square = current_square
							self.found = True
							break #TODO DA CAPIRE: QUESTO BREAK ESCE DA QUALE FOR LOOP?

						elif self.live_chessboard_situation_complete[current_square][1] == self.color and piece_color == self.opposite_color:
							#The piece has been captured.
							ending_square = current_square
							self.found = True
							break #TODO DA CAPIRE: QUESTO BREAK ESCE DA QUALE FOR LOOP?
			if self.found:
				break

		if not self.found: #If the end square has not been found again, it means that the pcl processing has not worked properly.
			rospy.loginfo('PCL processing has not worked properly. Try processing again.')
			ending_square = 'none'

		return ending_square, en_passant, en_passant_square, promotion
	
	def rook_checking(self, square, chessboard, threshold, squares_boundboxes, color_threshold, is_color_over):
		#square: square from which the rook has been moved.

		self.found = False

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

		#Remove from the row and column array the starting square,cause i will not search on that one for the end square.
		row.remove(index)
		column.remove(index)

		#Check which squares on the row are occupied.
		for element in row:
			#Save the name of the current square
			for key, value in self.squares_to_index.items():
				if value == element:
					current_square = key
			square_box = self.crop(chessboard, squares_boundboxes[element])
			points_number = len(np.asarray(square_box.points))
			if points_number > threshold: #The square is occupied
				RGB_colors = square_box.colors
				mean_RGB = self.mean_RGB_points([RGB_colors])
				piece_color = self.recognize_color(mean_RGB, color_threshold, is_color_over)
				#Check if the square was occupied before too.
				if self.live_chessboard_situation_complete[current_square][1] == 'none':
					#This means that the square was not occupied before.
					ending_square = current_square
					#Update the live_chessboard_situation
					self.live_chessboard_situation_complete[current_square][0] = self.live_chessboard_situation_complete[square][0]
					self.live_chessboard_situation_complete[current_square][1] = self.opposite_color
					self.live_chessboard_situation_complete[square][0] = 'none'
					self.live_chessboard_situation_complete[square][1] = 'none'
					self.found = True
					break #TODO DA CAPIRE: QUESTO BREAK ESCE DA QUALE FOR LOOP?
				elif self.live_chessboard_situation_complete[current_square][1] == self.color and piece_color == self.opposite_color:
					#The piece has been captured.
					ending_square = current_square
					#Update the live_chessboard_situation
					self.live_chessboard_situation_complete[current_square][0] = self.live_chessboard_situation_complete[square][0]
					self.live_chessboard_situation_complete[current_square][1] = self.opposite_color
					self.live_chessboard_situation_complete[square][0] = 'none'
					self.live_chessboard_situation_complete[square][1] = 'none'
					self.found = True
					break #TODO DA CAPIRE: QUESTO BREAK ESCE DA QUALE FOR LOOP?					

		if not self.found: #If the end square has not been found yet
			#Check the columns.
			for element in column:
				for key, value in self.squares_to_index:
					if value == element:
						current_square = key
				square_box = self.crop(chessboard, squares_boundboxes[element])
				points_number = len(np.asarray(square_box.points))
				if points_number > threshold: #The square is occupied
					#Check if the square was occupied before too.
					if self.live_chessboard_situation_complete[current_square][1] == 'none':
						#This means that the square was not occupied before.

						#Castle didn't happen, so I just found the ending square of the rook.
						ending_square = current_square
						#Update the live_chessboard_situation
						self.live_chessboard_situation_complete[current_square][0] = self.live_chessboard_situation_complete[square][0]
						self.live_chessboard_situation_complete[current_square][1] = self.opposite_color
						self.live_chessboard_situation_complete[square][0] = 'none'
						self.live_chessboard_situation_complete[square][1] = 'none'
						self.found = True
						break #TODO DA CAPIRE: QUESTO BREAK ESCE DA QUALE FOR LOOP?
					elif self.live_chessboard_situation_complete[current_square][1] == self.color and piece_color == self.opposite_color:
						#The piece has been captured.
						ending_square = current_square
						#Update the live_chessboard_situation
						self.live_chessboard_situation_complete[current_square][0] = self.live_chessboard_situation_complete[square][0]
						self.live_chessboard_situation_complete[current_square][1] = self.opposite_color
						self.live_chessboard_situation_complete[square][0] = 'none'
						self.live_chessboard_situation_complete[square][1] = 'none'
						self.found = True
						break #TODO DA CAPIRE: QUESTO BREAK ESCE DA QUALE FOR LOOP?

		if not self.found: #If the end square has not been found again, it means that the pcl processing has not worked properly.
			rospy.loginfo('PCL processing has not worked properly. Try processing again.')

		return ending_square, en_passant, en_passant_square, promotion
			
	def bishop_checking(self, square, chessboard, threshold, squares_boundboxes, color_threshold, is_color_over):
		#square: square from which the bishop has been moved.

		self.found = False
		exist = True
		diagonal_squares = []
		count = 1
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
					if casella_colonna in self.rows_indexes[rows[prev1_row_ind]] and self.live_chessboard_situation_complete[square[0]][1] == 'none':
						diagonal_squares.append(casella_colonna)
			if next1_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev1_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[next1_row_ind]] and self.live_chessboard_situation_complete[square[0]][1] == 'none':
						diagonal_squares.append(casella_colonna)
		if prev2_column_ind in range(8):
			if prev2_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev2_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[prev2_row_ind]] and self.live_chessboard_situation_complete[square[0]][1] == 'none':
						diagonal_squares.append(casella_colonna)
			if next2_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev2_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[next2_row_ind]] and self.live_chessboard_situation_complete[square[0]][1] == 'none':
						diagonal_squares.append(casella_colonna)
		if prev3_column_ind in range(8):
			if prev3_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev3_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[prev3_row_ind]] and self.live_chessboard_situation_complete[square[0]][1] == 'none':
						diagonal_squares.append(casella_colonna)
			if next3_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev3_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[next3_row_ind]] and self.live_chessboard_situation_complete[square[0]][1] == 'none':
						diagonal_squares.append(casella_colonna)
		if prev4_column_ind in range(8):
			if prev4_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev4_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[prev4_row_ind]] and self.live_chessboard_situation_complete[square[0]][1] == 'none':
						diagonal_squares.append(casella_colonna)
			if next4_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev4_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[next4_row_ind]] and self.live_chessboard_situation_complete[square[0]][1] == 'none':
						diagonal_squares.append(casella_colonna)
		if prev5_column_ind in range(8):
			if prev5_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev5_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[prev5_row_ind]] and self.live_chessboard_situation_complete[square[0]][1] == 'none':
						diagonal_squares.append(casella_colonna)
			if next5_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev5_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[next5_row_ind]] and self.live_chessboard_situation_complete[square[0]][1] == 'none':
						diagonal_squares.append(casella_colonna)
		if prev6_column_ind in range(8):
			if prev6_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev6_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[prev6_row_ind]] and self.live_chessboard_situation_complete[square[0]][1] == 'none':
						diagonal_squares.append(casella_colonna)
			if next6_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev6_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[next6_row_ind]] and self.live_chessboard_situation_complete[square[0]][1] == 'none':
						diagonal_squares.append(casella_colonna)
		if prev7_column_ind in range(8):
			if prev7_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev7_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[prev7_row_ind]] and self.live_chessboard_situation_complete[square[0]][1] == 'none':
						diagonal_squares.append(casella_colonna)
			if next7_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[prev7_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[next7_row_ind]] and self.live_chessboard_situation_complete[square[0]][1] == 'none':
						diagonal_squares.append(casella_colonna)
		if next1_column_ind in range(8):
			if prev1_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next1_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[prev1_row_ind]] and self.live_chessboard_situation_complete[square[0]][1] == 'none':
						diagonal_squares.append(casella_colonna)
			if next1_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next1_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[next1_row_ind]] and self.live_chessboard_situation_complete[square[0]][1] == 'none':
						diagonal_squares.append(casella_colonna)
		if next2_column_ind in range(8):
			if prev1_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next2_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[prev2_row_ind]] and self.live_chessboard_situation_complete[square[0]][1] == 'none':
						diagonal_squares.append(casella_colonna)
			if next2_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next2_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[next2_row_ind]] and self.live_chessboard_situation_complete[square[0]][1] == 'none':
						diagonal_squares.append(casella_colonna)
		if next3_column_ind in range(8):
			if prev1_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next3_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[prev3_row_ind]] and self.live_chessboard_situation_complete[square[0]][1] == 'none':
						diagonal_squares.append(casella_colonna)
			if next3_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next3_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[next3_row_ind]] and self.live_chessboard_situation_complete[square[0]][1] == 'none':
						diagonal_squares.append(casella_colonna)
		if next4_column_ind in range(8):
			if prev4_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next4_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[prev4_row_ind]] and self.live_chessboard_situation_complete[square[0]][1] == 'none':
						diagonal_squares.append(casella_colonna)
			if next4_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next4_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[next4_row_ind]] and self.live_chessboard_situation_complete[square[0]][1] == 'none':
						diagonal_squares.append(casella_colonna)
		if next5_column_ind in range(8):
			if prev5_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next5_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[prev5_row_ind]] and self.live_chessboard_situation_complete[square[0]][1] == 'none':
						diagonal_squares.append(casella_colonna)
			if next5_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next5_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[next5_row_ind]] and self.live_chessboard_situation_complete[square[0]][1] == 'none':
						diagonal_squares.append(casella_colonna)
		if next6_column_ind in range(8):
			if prev6_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next6_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[prev6_row_ind]] and self.live_chessboard_situation_complete[square[0]][1] == 'none':
						diagonal_squares.append(casella_colonna)
			if next6_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next6_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[next6_row_ind]] and self.live_chessboard_situation_complete[square[0]][1] == 'none':
						diagonal_squares.append(casella_colonna)
		if next7_column_ind in range(8):
			if prev7_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next7_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[prev7_row_ind]] and self.live_chessboard_situation_complete[square[0]][1] == 'none':
						diagonal_squares.append(casella_colonna)
			if next7_row_ind in range(8):
				for casella_colonna in self.columns_indexes[columns[next7_column_ind]]:
					square = [k for k, v in self.squares_to_index.items() if v == (casella_colonna)]
					if casella_colonna in self.rows_indexes[rows[next7_row_ind]] and self.live_chessboard_situation_complete[square[0]][1] == 'none':
						diagonal_squares.append(casella_colonna)

		#Check the squares on the diagonal.
		for element in diagonal_squares:
			#Save the name of the current square
			for key, value in self.squares_to_index.items():
				if value == element:
					current_square = key
			square_box = self.crop(chessboard, squares_boundboxes[element])
			points_number = len(np.asarray(square_box.points))
			if points_number > threshold: #The square is occupied.
				RGB_colors = square_box.colors
				mean_RGB = self.mean_RGB_points([RGB_colors])
				piece_color = self.recognize_color(mean_RGB, color_threshold, is_color_over)
				#Check if the square was occupied before too.
				if self.live_chessboard_situation_complete[current_square][1] == 'none':
					#This means that the square was not occupied before. I just found the ending square of the bishop.
					ending_square = current_square
					self.found = True
					break #TODO DA CAPIRE: QUESTO BREAK ESCE DA QUALE FOR LOOP?

				elif self.live_chessboard_situation_complete[current_square][1] == self.color and piece_color == self.opposite_color:
					#The piece has been captured.
					ending_square = current_square
					self.found = True
					break #TODO DA CAPIRE: QUESTO BREAK ESCE DA QUALE FOR LOOP?

		if not self.found: #If the end square has not been found again, it means that the pcl processing has not worked properly.
			rospy.loginfo('PCL processing has not worked properly. Try processing again.')

		return ending_square, en_passant, en_passant_square, promotion

	def pawn_checking(self, square, chessboard, threshold, squares_boundboxes, color_threshold, is_color_over):
		#square: square from which the pawn has been moved.

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
		if points_number > threshold: #The square is occupied.
			#Save the name of the current square
			for key, value in self.squares_to_index.items():
				if value == next_square:
					ending_square = key
			self.found = True

		if not self.found and index in self.rows_indexes[self.seventh_row]: #If the pawn is in its starting position, it could be moved straight for two steps. Check this possibility.
			next_square_x2 = self.columns_indexes[column][ind + 2]

			square_box = self.crop(chessboard, squares_boundboxes[next_square_x2])
			points_number = len(np.asarray(square_box.points))
			if points_number > threshold: #The square is occupied.
				#Save the name of the current square
				for key, value in self.squares_to_index.items():
					if value == next_square_x2:
						ending_square = key
				self.found = True

		if not self.found: #If the move has not been identified yet, a capture has happened.
			#Check the 2 possible squares of capture of the pawn.
			capture_square_1 = index + 7
			capture_square_2 = index + 9

			rows = self.rows_indexes.keys()
			ind_current_row = rows.index(row)
			next_row = rows[ind_current_row + 1]
			if capture_square_1 in rows_indexes[next_row] and capture_square_2 not in rows_indexes[next_row]:
				#If the only possibility of caputre is in square_1, the pawn should be there. TODO SHOULD I CHECK ANYWAY?
				#Save the name of the current square.
				for key, value in self.squares_to_index:
					if value == capture_square_1:
						ending_square = key
				#Check if the square was occupied before. If it was not, en-passant happened.
				if self.live_chessboard_situation_complete[ending_square] == 'none':
					en_passant = True
					en_passant_square_index = row[index - 1]
					for key, value in self.squares_to_index:
						if value == en_passant_square_index:
							en_passant_square = key
				self.found = True
			
			if capture_square_2 in rows_indexes[next_row] and capture_square_1 not in rows_indexes[next_row]:
				#If the only possibility of caputer is in square_2, the pawn should be there.
				#Save the name of the current square.
				for key, value in self.squares_to_index:
					if value == capture_square_2:
						ending_square = key
				#Check if the square was occupied before. If it was not, en-passant happened.
				if self.live_chessboard_situation_complete[ending_square] == 'none':
					en_passant = True
					en_passant_square_index = row[index + 1]
					for key, value in self.squares_to_index:
						if value == en_passant_square_index:
							en_passant_square = key
				self.found = True

			if capture_square_2 in rows_indexes[next_row] and capture_square_1 in rows_indexes[next_row]:
				#If both the squares are possible, check which one of the 2 has a piece in it.

				square_box_2 = self.crop(chessboard, squares_boundboxes[capture_square_2])
				square_box_1 = self.crop(chessboard, squares_boundboxes[capture_square_1])
				points_number_2 = len(np.asarray(square_box_2.points))
				points_number_1 = len(np.asarray(square_box_1.points))
				RGB_colors_2 = square_box_2.colors
				RGB_colors_1 = square_box_1.colors
				mean_RGB_2 = self.mean_RGB_points([RGB_colors_2])
				mean_RGB_2 = self.mean_RGB_points([RGB_colors_1])
				piece_color_2 = self.recognize_color(mean_RGB_2, color_threshold, is_color_over)
				piece_color_1 = self.recognize_color(mean_RGB_1, color_threshold, is_color_over)
				if points_number_2 > threshold and points_number_1 < threshold: #Square 2 is occupied, square 1 is not.
					#Save the name of the current square
					for key, value in self.squares_to_index:
						if value == capture_square_2:
							ending_square = key
					#Check if the square was occupied before. If it was not, en-passant happened.
					if self.live_chessboard_situation_complete[ending_square] == 'none':
						en_passant = True
						en_passant_square_index = row[index + 1]
						for key, value in self.squares_to_index:
							if value == en_passant_square_index:
								en_passant_square = key

				if points_number_1 > threshold and points_number_2 < threshold: #Square 1 is occupied, square 2 is not.
					#Save the name of the current square
					for key, value in self.squares_to_index:
						if value == capture_square_1:
							ending_square = key

					#Check if the square was occupied before. If it was not, en-passant happened.
					if self.live_chessboard_situation_complete[ending_square] == 'none':
						en_passant = True
						en_passant_square_index = row[index + 1]
						for key, value in self.squares_to_index:
							if value == en_passant_square_index:
								en_passant_square = key

				if points_number_2 > threshold and points_number_1 > threshold: #Both square 1 and 2 are occupied.
					#Check the color of the pointcloud.
					if piece_color_1 == self.color and piece_color_2 == self.opposite_color:
						#The ending square of the pawn is square 2.
						for key, value in self.squares_to_index:
							if value == capture_square_2:
								ending_square = key
						#Check if the square was occupied before. If it was not, en-passant happened.
						if self.live_chessboard_situation_complete[ending_square] == 'none':
							en_passant = True
							en_passant_square_index = row[index + 1]
							for key, value in self.squares_to_index:
								if value == en_passant_square_index:
									en_passant_square = key

					if piece_color_2 == self.color and piece_color_1 == self.opposite_color:
						#The ending square of the pawn is square 1.
						for key, value in self.squares_to_index:
							if value == capture_square_1:
								ending_square = key
						#Check if the square was occupied before. If it was not, en-passant happened.
						if self.live_chessboard_situation_complete[ending_square] == 'none':
							en_passant = True
							en_passant_square_index = row[index + 1]
							for key, value in self.squares_to_index:
								if value == en_passant_square_index:
									en_passant_square = key

					if piece_color_1 == self.opposite_color and piece_color_2 == self.opposite_color:
						#Both the ending squares are occupied by pieces of the opposite color. I need to check the situation before.
						#Save the square 1 name.
						for key, value in self.squares_to_index:
							if value == capture_square_1:
								square_1 = key
						#Save the square 2 name.
						for key, value in self.squares_to_index:
							if value == capture_square_1:
								square_2 = key
						if self.live_chesboard_situation_complete[square_1][1] == self.opposite_color or self.live_chesboard_situation_complete[square_1][1] == 'none':
							#In this case the pawn has captured in square 2. En-passant could have happened.
							ending_square = square_2
							#Check if the square was occupied before. If it was not, en-passant happened.
							if self.live_chessboard_situation_complete[ending_square] == 'none':
								en_passant = True
								en_passant_square_index = row[index + 1]
								for key, value in self.squares_to_index:
									if value == en_passant_square_index:
										en_passant_square = key

						if self.live_chesboard_situation_complete[square_2][1] == self.opposite_color or self.live_chesboard_situation_complete[square_1][1] == 'none':
							#In this case the pawn has captured in square 1. En-passant could have happened.
							ending_square = square_1
							#Check if the square was occupied before. If it was not, en-passant happened.
							if self.live_chessboard_situation_complete[ending_square] == 'none':
								en_passant = True
								en_passant_square_index = row[index + 1]
								for key, value in self.squares_to_index:
									if value == en_passant_square_index:
										en_passant_square = key

		if not self.found: #If the end square has not been found again, it means that the pcl processing has not worked properly.
			rospy.loginfo('PCL processing has not worked properly. Try processing again.')
			ending_square = 'none'

		return ending_square, en_passant, en_passant_square, promotion

	def query_promoted_piece(self, question, default = "queen"):
	    #Ask a question via raw_input() and return their answer.
	    #question: string that is presented to the user.
	    #default: presumed answer if the user just hits <Enter>. It must be "queen" (the default) or None (meaning an answer is required of the user).
	    
	    valid = {"queen": 'queen', "q": 'queen', "rook": 'rook',"r": 'rook', "knight": 'knight', "k": 'knight', "bishop": 'bishop', "b": 'bishop',}
	    if default is None:
	        prompt = " [queen/rook/knight/bishop] "
	    elif default == "queen":
	        prompt = " [Queen/rook/knight/bishop] "
	    else:
	        raise ValueError("Invalid default answer: '%s'" % default)

	    while True:
	        sys.stdout.write(question + prompt)
	        choice = raw_input().lower()
	        if default is not None and choice == "":
	            return valid[default]
	        elif choice in valid:
	            return valid[choice]
	        else:
	            sys.stdout.write("Please respond with 'queen' or 'rook' or 'knight' or 'bishop' " "(or 'q', 'r', 'k', 'b').\n")

	def update_situation(self, en_passant, en_passant_square, castle, castle_squares, promotion, ending_square, start_square):
		#Function to update the live_chessboard_situation_complete dictionary to keep track of the game.
		#en_passant: True if en_passant happened, False if not.
		#en_passant_square: square of the piece that has been captured with en-passant
		#castle: True if castle happened, False if not.
		#castle_squares: squares of the king and the rook that have been castled.
		#promotion: True if promotion happened, False if not.
		#ending_square: end square of the move.
		#start_square: start square of the move.

		if en_passant:
			self.live_chessboard_situation_complete[en_passant_square][0] = 'none'
			self.live_chessboard_situation_complete[en_passant_square][1] = 'none'
			self.live_chessboard_situation_complete[ending_square][0] = self.live_chessboard_situation_complete[start_square][0]
			self.live_chessboard_situation_complete[ending_square][1] = self.opposite_color
			self.live_chessboard_situation_complete[start_square][0] = 'none'
			self.live_chessboard_situation_complete[start_square][1] = 'none'	

		elif castle:
			if castle_squares[0] == 'A8' or castle_squares[0] == 'A1':
				self.live_chessboard_situation_complete[self.castle_ending_squares[0]][0] = self.castle_squares[4] #rook
				self.live_chessboard_situation_complete[self.castle_ending_squares[0]][1] = self.opposite_color
				self.live_chessboard_situation_complete[self.castle_ending_squares[1]][0] = self.castle_squares[3] #king
				self.live_chessboard_situation_complete[self.castle_ending_squares[2]][1] = self.opposite_color
				self.live_chessboard_situation_complete[self.castle_squares[0]][0] = 'none'
				self.live_chessboard_situation_complete[self.castle_squares[0]][1] = 'none'
			elif castle_squares[0] == 'H8' or castle_squares[0] == 'H1':
				self.live_chessboard_situation_complete[self.castle_ending_squares[3]][0] = self.castle_squares[5] #rook
				self.live_chessboard_situation_complete[self.castle_ending_squares[3]][1] = self.opposite_color
				self.live_chessboard_situation_complete[self.castle_ending_squares[2]][0] = self.castle_squares[3] #king
				self.live_chessboard_situation_complete[self.castle_ending_squares[2]][1] = self.opposite_color
				self.live_chessboard_situation_complete[self.castle_squares[2]][0] = 'none'
				self.live_chessboard_situation_complete[self.castle_squares[2]][1] = 'none'
			self.live_chessboard_situation_complete[self.castle_squares[1]][0] = 'none'
			self.live_chessboard_situation_complete[self.castle_squares[1]][1] = 'none'

		elif promotion:
			promoted_piece = self.query_promoted_piece('What piece have been promoted?')
			self.live_chessboard_situation_complete[ending_square][0] = promoted_piece
			self.live_chessboard_situation_complete[ending_square][1] = self.opposite_color
			self.live_chessboard_situation_complete[start_square][0] = 'none'
			self.live_chessboard_situation_complete[start_square][1] = 'none'

		else:
			self.live_chessboard_situation_complete[ending_square][0] = self.live_chessboard_situation_complete[start_square][0]
			self.live_chessboard_situation_complete[ending_square][1] = self.opposite_color
			self.live_chessboard_situation_complete[start_square][0] = 'none'
			self.live_chessboard_situation_complete[start_square][1] = 'none'

	def calibrate_points(self):
		#Function to estabilish how many points in the boundbox can be considered enough to decide that there is a piece there.
		#This should be used when pieces are in the starting position.

		starting_pcl = o3d.io.read_point_cloud(pointcloud_for_calibration) #PointCloud with iformation for calibration.

		starting_squares_matrix = self.get_points_in_matrix_from_array(self.squares_centers) #To use if I'm using the original points in the eyes reference frame.
		starting_vertices_matrix = self.get_points_in_matrix_from_array(self.vertices) #To use if I'm usign the original points in the eyes reference frame.
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
		for point in starting_vertices_matrix:
			if (point[0] < 0 and point[1] > 0):
				starting_bottom_left_vertex = point
			if (point[0] < 0 and point[1] < 0):
				starting_bottom_right_vertex = point
			if (point[0] > 0 and point[1] > 0):
				starting_top_left_vertex = point
			if (point[0] > 0 and point[1] < 0):
				starting_top_right_vertex = point
		starting_length = starting_bottom_left_vertex[1] - starting_bottom_right_vertex[1]
		starting_width = starting_top_left_vertex[0] - starting_bottom_left_vertex[0]
		#Rotation matrix
		starting_R, starting_reference_normal = self.get_rot_matrix(starting_pcl, starting_normals, starting_bottom_left_vertex)
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
		meanR = sumR / point_count
		meanG = sumG / point_count
		meanB = sumB / point_count
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

		return piece_color
		

	def look_for_pieces(self, pcl, chessboard, squares_boundboxes, threshold, color_threshold, is_color_over):
		#Function to look for the presence of pieces over the chessboard squares.
		#chessboard: PointCloud of the isolated chessboard.
		#squares_boundboxes: a list containing the 64 bounding boxes to isolate each square and the things inside them.
		#threshold: threshold of points to consider a square occupied or not

		castle = False
		castle_squares = 'none'
		#Save the squares that are currently occupied by opposite's pieces
		self.currently_occupied = []
		for square in self.live_chessboard_situation_complete:
			if self.live_chessboard_situation_complete[square][1] == self.opposite_color:
				self.currently_occupied.append(square)
		#Check the squares occupied by the opposite's pieces. Whenever one piece is moved, get out of the while loop.
		checking_flag = True
		count_empty = 0
		check_squares = []
		check_pieces = []
		check_columns = []
		while checking_flag:
			for occupied in self.currently_occupied:
				square = self.crop(pcl, squares_boundboxes[self.squares_to_index[occupied]])
				#o3d.visualization.draw_geometries([chessboard, squares_boundboxes[self.squares_to_index[occupied]]])
				points_number = len(np.asarray(square.points))
				#print('Points number: ' + str(points_number))
				#print('casella: ' + str(occupied))
				#Save the index of the starting square.
				index = self.squares_to_index[occupied]
				for key, values in self.columns_indexes.items():
					if index in values:
						column = key
				if points_number < (threshold * 0.3): #If the points over the square are less than a threshold, it means that the piece have been moved. take the 30% of the threshold to exclude the pieces covered by pieces in front of them.
					move_square = occupied
					moved_piece = self.live_chessboard_situation_complete[move_square][0]
					checking_flag = False
					check_squares.append(move_square)
					check_pieces.append(moved_piece)
					check_columns.append(column)
					count_empty += 1
			if count_empty > 1:
				#print('Check squares: ' + str(check_squares))
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
					castle_squares = cast_sq
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
						elif self.pieces_coordinates[moved_piece][1]['name'] == 'bishop':
							ending_square, en_passant, en_passant_square, promotion = self.bishop_checking(move_square, chessboard, threshold, squares_boundboxes, color_threshold, is_color_over)
						elif self.pieces_coordinates[moved_piece][1]['name'] == 'rook':
							ending_square, en_passant, en_passant_square, promotion = self.rook_checking(move_square, chessboard, threshold, squares_boundboxes, color_threshold, is_color_over)
						elif self.pieces_coordinates[moved_piece][1]['name'] == 'queen':
							ending_square, en_passant, en_passant_square, promotion = self.queen_checking(move_square, chessboard, threshold, squares_boundboxes, color_threshold, is_color_over)
						elif self.pieces_coordinates[moved_piece][1]['name'] == 'king':
							ending_square, en_passant, en_passant_square, promotion = self.king_checking(move_square, chessboard, threshold, squares_boundboxes, color_threshold, is_color_over)
						#TODO AGGIUNGERE TUTTI GLI ALTRI PEZZI
						conteggio += 1
						check_ending_square.append(ending_square)
					for i in range(0, len(check_ending_square)):
						if check_ending_square[i] not in check_columns[i]:
							moved_piece = check_pieces[i]
							move_square = check_squares[i]
					checking_flag = False
		if not castle:
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

		#Update the live_chessboard_situation
		self.update_situation(en_passant, en_passant_square, castle, castle_squares, promotion, ending_square, move_square)

		return moved_piece, ending_square


def main():
	rospy.init_node('pcl_processor')
	pcl_processor = PclProcessing()

	#pcl = o3d.io.read_point_cloud(imported_pointcloud) #pcl taken directly from TIAGo. This method works but takes a lot of time for acquiring PointCloud data. 
	#print('Punti: '+str(np.asarray(pcl.colors)))
	#pcl.paint_uniform_color([0, 0, 0]) #Paint the PointCloud seen by TIAGo of black (to do if the pointcloud is without color information)

	#Visualize the PointClud
	#o3d.visualization.draw_geometries([pcl])

	#Visualize the pointcloud with the squares centers points in red.
	#pcl_processor.visualize_points(pcl, pcl_processor.squares_centers)

	'''
	pts = np.asarray(pcl.points)
	print(pts[:, 0].min(axis = 0), pts[:, 0].max(axis = 0))
	print(pts[:, 1].min(axis = 0), pts[:, 1].max(axis = 0))
	print(pts[:, 2].min(axis = 0), pts[:, 2].max(axis = 0))
	'''

	pcl = pcl_processor.create_pointcloud_from_depth()

	#Visualize the PointCloud
	o3d.visualization.draw_geometries([pcl])

	if pcl_processor.verbose:
		o3d.visualization.draw_geometries([pcl])
	#points_matrix = pcl_processor.get_points_in_matrix_from_pointstamped(pcl_processor.squares_centers) #To use if I'm usign the transformed points in the base_footprint reference frame. Otherwise, the other points are in a list of arrays.
	squares_matrix = pcl_processor.get_points_in_matrix_from_array(pcl_processor.squares_centers) #To use if I'm using the original points in the eyes reference frame.
	vertices_matrix = pcl_processor.get_points_in_matrix_from_array(pcl_processor.vertices) #To use if I'm usign the original points in the eyes reference frame.
	squares_pointcloud = pcl_processor.create_pointcloud(squares_matrix, [1, 0, 0]) #Creation of a red PointCloud of the squares centers computed segmenting the chessboard.
	vertices_pointcloud = pcl_processor.create_pointcloud(vertices_matrix, [0, 1, 0]) #Creation of a red PointCloud of the vertices computed segmenting the chessboard.

	#o3d.visualization.draw_geometries([pcl, squares_pointcloud])

	#Normals estimation
	normals = pcl_processor.normals(squares_pointcloud)

	#Save the coordinates of the centrale squares to compute the coordinates of the center.
	d4 = squares_pointcloud.points[35]
	e4 = squares_pointcloud.points[36]
	d5 = squares_pointcloud.points[27]
	center = [(e4[0] + d5[0])/2, (d5[1] + d4[1])/2, (d4[2] + d5[2] + e4[2])/3]

	#Save the bottom left vertex and the length and width of the chessboard:
	for point in vertices_matrix:
		if (point[0] < 0 and point[1] > 0):
			bottom_left_vertex = point
		if (point[0] < 0 and point[1] < 0):
			bottom_right_vertex = point
		if (point[0] > 0 and point[1] > 0):
			top_left_vertex = point
		if (point[0] > 0 and point[1] < 0):
			top_right_vertex = point
	length = bottom_left_vertex[1] - bottom_right_vertex[1]
	width = top_left_vertex[0] - bottom_left_vertex[0]

	#Rotation matrix
	R, reference_normal = pcl_processor.get_rot_matrix(pcl, normals, bottom_left_vertex)

	#Create a bounding box around the chessboard
	bounding_box_chessboard, frame = pcl_processor.create_bounding_box(pcl, length, width, R, center, reference_normal, 0.25, 0.15, 0)
	#o3d.visualization.draw_geometries([pcl, bounding_box_chessboard])

	#Crop the PointCloud to extract the chessboard
	chessboard = pcl_processor.crop(pcl, bounding_box_chessboard)
	#o3d.visualization.draw_geometries([chessboard])

	#Compute the mean distance between two consecutive squares in the y direction and in the x direction
	sum_width = 0
	sum_length = 0
	for i in range(1, 7):
		sum_width = sum_width + abs(squares_pointcloud.points[pcl_processor.rows_indexes['row8'][i]][0] - squares_pointcloud.points[pcl_processor.rows_indexes['row8'][i-1]][0])
		sum_width = sum_width + abs(squares_pointcloud.points[pcl_processor.rows_indexes['row5'][i]][0] - squares_pointcloud.points[pcl_processor.rows_indexes['row5'][i-1]][0])
		sum_width = sum_width + abs(squares_pointcloud.points[pcl_processor.rows_indexes['row1'][i]][0] - squares_pointcloud.points[pcl_processor.rows_indexes['row1'][i-1]][0])
		sum_length = sum_length + abs(squares_pointcloud.points[pcl_processor.columns_indexes['columnA'][i]][1] - squares_pointcloud.points[pcl_processor.columns_indexes['columnA'][i-1]][1])
		sum_length = sum_length + abs(squares_pointcloud.points[pcl_processor.columns_indexes['columnD'][i]][1] - squares_pointcloud.points[pcl_processor.columns_indexes['columnD'][i-1]][1])
		sum_length = sum_length + abs(squares_pointcloud.points[pcl_processor.columns_indexes['columnH'][i]][1] - squares_pointcloud.points[pcl_processor.columns_indexes['columnH'][i-1]][1])

	mean_y_dist = sum_width / 21
	mean_x_dist = sum_length / 21
	
	#Create bounding boxes for each square
	squares_bound_boxes = []
	squares_frames = []
	for squares in squares_pointcloud.points:
		bounding_box_square, frame = pcl_processor.create_bounding_box(chessboard, mean_y_dist, mean_x_dist, R, squares, reference_normal, 0.05, 0.02, 0.035)
		squares_bound_boxes.append(bounding_box_square)
		squares_frames.append(frame)
	#o3d.visualization.draw_geometries([pcl, squares_bound_boxes[6], squares_frames[6]])
	
	#Crop single squares to visualize them
	square = pcl_processor.crop(chessboard, squares_bound_boxes[6])
	#o3d.visualization.draw_geometries([square, squares_bound_boxes[6], squares_frames[6]])
	
	#Calibrate to save the optimal points threshold to discriminate when a square is occupied or not
	mean_points_occupied, mean_points_not_occupied, color_RGB, opponent_color_RGB = pcl_processor.calibrate_points()
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

	#Look for pieces
	moved_piece, ending_square = pcl_processor.look_for_pieces(pcl, chessboard, squares_bound_boxes, occupied_threshold, color_threshold, is_color_over)
	print('MOVED PIECE: ' + str(moved_piece))
	print('ENDING SQUARE: ' + str(ending_square))
	#print('SITUATION: ' + str(pcl_processor.live_chessboard_situation_complete))
	


if __name__ == '__main__':
     main()

