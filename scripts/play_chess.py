#!/usr/bin/env python
# Script to make TIAGo grasp pieces on the chess board

# Import ROS libraries
import rospy, roslaunch, rosnode
import moveit_commander

# Import Python libraries
import sys
import yaml
import math
from time import time

# Import ROS messages
from geometry_msgs.msg import Quaternion, PointStamped, Pose, Point
from std_msgs.msg import Bool, Int16, String
from std_srvs.srv import Empty
from moveit_commander import PlanningSceneInterface

# My scripts
import config as cfg #real_config
import armies as army
import roslauncher as roslauncher
from errors import ArucoException, PlanningException, StraightMovementError
from color import Color
from grasp_piece import Grasping

PACKAGE_DIR = '/home/luca/tiago_public_ws/src/tiago_playchess'
imported_configurations = PACKAGE_DIR + '/scripts/config/simulation_config.yaml'

### TODO: set from launch
ready = True # True if segmentation and search have been already performed, False otherwise

# Callbacks definition
### TODO: move into class to avoid global variables
def Callback_State(data):
    state.data = data.data

def CallbackStartSquare(data):
	global start_square
	start_square = data.data
	#print('START SQUARE: '+str(start_square))

def CallbackEndSquare(data):
	global end_square
	end_square = data.data
	#print('END SQUARE: '+str(end_square))

def CallbackPushbutton(data):
	if data and state.data == 14:
		rospy.loginfo("The button has been pushed, so the opponent move has been executed." + 
					  " Start looking for the just executed move.")
		state_publisher.publish(15)

def CallbackSegmentAgain(data):
	if data:
		rospy.loginfo("Perform segmentation again.")
		#Change the flag
		global segmentation_not_launched
		segmentation_not_launched = True

def CallbackSearchAgain(data):
	if data:
		rospy.loginfo("Perform ARUCO markers search again.")
		#Change the flag
		global aruco_detector_not_launched
		aruco_detector_not_launched = True

def CallbackPcSaved(data):
	if data:
		rospy.loginfo("Initial Pointcloud saved in the tiago_playchess/PointcloudForCalibration folder.")
		#Shutdown the pc_saver node
		pc_saver_launcher.shutdown()

def CallbackEnPassantSquare(data):
	global en_passant_square
	en_passant_square = data.data
	print('EN PASSANT SQUARE: ' + str(en_passant_square))


def squares_from_rows(rows):	### TODO: use instead of hardcoding
	'''
	rows 	[int list] index of the rows
	----------
	squares [str list] squares in the given rows (letter are lowercase)
	'''
	return [chr(97 + i) + str(row) for row in rows for i in range(8)]	# chr(97) -> 'a'

def Color_Callback(data):
		global color
		global last_squares
		global first_squares
		global squares_to_index
		color = data.data
		if color == 'white':
			squares_to_index = cfg.squares_to_index_white
			last_squares = ['a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8', 
							'a7', 'b7', 'c7', 'd7','e7', 'f7', 'g7', 'h7'
						   ]
			first_squares = ['a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1', 
							 'a2', 'b2', 'c2', 'd2','e2', 'f2', 'g2', 'h2'
							]
		elif color == 'black':
			squares_to_index = cfg.squares_to_index_black
			last_squares = ['a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1', 
							'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2'
						   ]
			first_squares = ['a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8', 
							 'a7', 'b7', 'c7', 'd7','e7', 'f7', 'g7', 'h7'
							]

def Move_the_arm_Callback(data):
		if data:
			### NOTE: the look_up and look_down movements are TIAGo-specific
			grasper.look_up() # look up to avoid head-arm collisions 
			time.sleep(2)
			grasper.move_arm_joint(0.471)	### TODO: set a rest pose from launch file
			time.sleep(2)
			grasper.look_down() # look down at the chessboard



if ready:
	global last_squares
	global first_squares
	global color
	global squares_to_index
	with open(imported_configurations) as file:
		configurazioni = yaml.load(file)
		color = configurazioni.get('color')
	if color == 'white':
		squares_to_index = cfg.squares_to_index_white
		last_squares = ['a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8', 
						'a7', 'b7', 'c7', 'd7','e7', 'f7', 'g7', 'h7'
					   ]
		first_squares = ['a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1', 
						 'a2', 'b2', 'c2', 'd2','e2', 'f2', 'g2', 'h2'
						]
	elif color == 'black':
		squares_to_index = cfg.squares_to_index_black
		last_squares = ['a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1', 
						'a2', 'b2', 'c2', 'd2','e2', 'f2', 'g2', 'h2'
					   ]
		first_squares = ['a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8', 
						 'a7', 'b7', 'c7', 'd7','e7', 'f7', 'g7', 'h7'
						]

# Starts a new node
rospy.init_node('play_chess', anonymous = True)

# Publishers initialization
state_publisher = rospy.Publisher('/state', Int16, queue_size = 10)

# Subscribers initialization
rospy.Subscriber("/state", Int16, Callback_State)
rospy.Subscriber('/start_square', String, CallbackStartSquare)
rospy.Subscriber('/end_square', String, CallbackEndSquare)
rospy.Subscriber('/pushed', Bool, CallbackPushbutton)
rospy.Subscriber('/segment_again', Bool, CallbackSegmentAgain)
rospy.Subscriber('/search_again', Bool, CallbackSearchAgain)
rospy.Subscriber('/pc_saved', Bool, CallbackPcSaved)
rospy.Subscriber('/en_passant_square_to_move', String, CallbackEnPassantSquare)
rospy.Subscriber("/color", String, Color_Callback)
rospy.Subscriber("/move_the_arm", Bool, Move_the_arm_Callback)


# Messages initialization
state = Int16()
state.data = 0

en_passant_square = 'none'

# Flags initialization
segmentation_not_launched = True
aruco_detector_not_launched = True
CV_launcher_not_launched = True
chess_move_not_exectuted_yet = True
pc_saver_not_launched = True

# Parameters loading
z_coord_chessboard_mean_file = rospy.get_param('/tiago_playchess/z_coord_chessboard_mean')
clock_pose_file = rospy.get_param('/tiago_playchess/clock_pose_file')
box_pose_file = rospy.get_param('/tiago_playchess/box_pose_file')
simul_config = rospy.get_param('/tiago_playchess/simul_config')

class Playing:
# Class with functions to make TIAGo wait for the move, to tell him what to move and, finally, 
# to make him perform the move.

	def __init__(self):
		# Initialize a MoveIt commander
		moveit_commander.roscpp_initialize(sys.argv)
		# MoveIt utilities
		self.motions = Grasping()

		# Subscribers initialization
		#rospy.Subscriber("/move_the_arm", Bool, self.Move_the_arm_Callback)

		# Define the interface with the robot's representation of the surrounding environment 
		self.scene = moveit_commander.PlanningSceneInterface() 		

		# Import pieces characteristics
		self.pawn = cfg.pawn
		self.rook = cfg.rook
		self.bishop = cfg.bishop
		self.king = cfg.king
		self.queen = cfg.queen
		self.knight = cfg.knight

		# Load configurations
		self.config_file = rospy.get_param('/tiago_playchess/config')		
		with open(self.config_file) as file:
			self.config = yaml.load(file)

		# Maximum time to wait when looking for the object or for ArUco markers
		self.time_limit = self.config.get('time_limit', 3)
		# Wheter to clear the octomap or not at the end of the task (defaulted to False) ### TOFIX
		self.clear_all = self.config.get('clear_octomap', True) 

		# Load parameters regarding the chessboard
		self.pieces_coordinates = cfg.pieces_coordinates

		# Directory of the live chessboard situation
		self.dir_live_chessboard_situation = PACKAGE_DIR + "/scripts/live_chessboard_situation.yaml"

		self.columns = cfg.columns_explicit

		self.transformed_centroids = None
		self.box_marker_pose = None

	def chess_move(self, start_place, end_place, centroids, clock_pose, box_pose, z_coord_chessboard):
		#en_passant_square = 'none'
		global en_passant_square
		global fraction
		global translation
		exception_1_flag = False
		exception_2_flag = False
		exception_3_flag = False
		which_error = 1 # 1. if the error comes from straight_eef_movement (i.e. highering or 
						#    lowering the gripper) 
						# 2. if it comes from straight_movements_chessboard (i.e. movements to 
						#    reach squares)
		# Execution of a complete move: a simple one or taking a piece.
		# Load the live chessboard situation
		with open(self.dir_live_chessboard_situation, 'rb') as live_file:
			self.live_chessboard_situation = yaml.load(live_file.read())

		try:
			'''
			### TODO: create a separate file `armies.py`. Create two dictionaries (`white` and
					  `black`). For each army, define:
						- close_squares
						- far_squares
						- start/end squares for long and short castle
			'''
			if start_square == 'short' or start_square == 'long': #Castle happened.
				rospy.loginfo(str(start_square) + ' castle happening')
				offset_start_place = 0.10
				offset_end_place = 0.10
				self.tilt = False

				if color == 'white':
					king = army.white['king_' + start_square]
					rook = army.white['rook_' + start_square]

				elif color == 'black':
					king = army.black['king_' + start_square]
					rook = army.black['rook_' + start_square]

				piece = king['piece']
				self.motions.gripper_over_piece(king['from'],
												centroids,
												self.live_chessboard_situation,
												squares_to_index,
												z_coord_chessboard
												)
				self.motions.playmotion_movement('open_gripper', planning = False)
				which_error = 1
				# Lower the gripper
				fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 
																		   'z', 
																		   -offset_start_place, 
																		   avoid_collisions = False
																		   )
				# Close the gripper to pick the piece
				self.motions.pick_piece(piece, self.pieces_coordinates)
				# Higher the gripper
				fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 
																		   'z', 
																		   offset_start_place, 
																		   avoid_collisions = False
																		   )
				which_error = 2
				# Move the arm with straight movements over the chessboard
				self.motions.straight_movements_chessboard(self.motions.arm, 
														   centroids, 
														   king['to'], 
														   squares_to_index, 
														   self.tilt, 
														   clock_pose, 
														   box_pose, 
														   avoid_collisions = False
														   ) 
				which_error = 1
				# Lower the gripper
				fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 
																		   'z', 
																		   -offset_end_place, 
																		   avoid_collisions = False
																		   )
				# Place the piece in the end square
				self.motions.place_piece(king['to'])
				# Higher the gripper
				fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 
																		   'z', 
																		   offset_end_place, 
																		   avoid_collisions = False
																		   )
				# Move the rook
				piece = rook['piece']
				self.motions.gripper_over_piece(rook['from'], 
												centroids, 
												self.live_chessboard_situation, 
												squares_to_index, 
												z_coord_chessboard
												)
				self.motions.playmotion_movement('open_gripper', planning = False)
				# Lower the gripper
				fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 
																		   'z', 
																		   -offset_start_place, 
																		   avoid_collisions = False
																		   )		
				# Close the gripper to pick the piece.
				self.motions.pick_piece(piece, self.pieces_coordinates)
				# Higher the gripper
				fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 
																		   'z', 
																		   offset_start_place, 
																		   avoid_collisions = False
																		   ) 
				which_error = 3
				# Move the arm with straight movements over the chessboard.
				self.motions.straight_movements_chessboard(self.motions.arm, 
														   centroids, 
														   rook['to'], 
														   squares_to_index, 
														   self.tilt, 
														   clock_pose, 
														   box_pose, 
														   avoid_collisions = False
														   )
				which_error = 1
				# Lower the gripper
				fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 
																		   'z', 
																		   -offset_end_place, 
																		   avoid_collisions = False
																		   )					
				# Place the piece in the end square.
				self.motions.place_piece(rook['to'])
				# Higher the gripper
				fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 
																		   'z', 
																		   offset_end_place, 
																		   avoid_collisions = False
																		   )

				# Click the clock
				self.motions.gripper_over_clock(clock_pose, z_coord_chessboard)
				self.motions.click_clock(self.scene, clock_pose)
						
			else:
				piece = self.live_chessboard_situation[start_place][0] #Save the piece that is going to be moved.
				taken_piece = self.live_chessboard_situation[end_place][0] #Save the piece that is going to be taken (if any).

				if start_place in last_squares:
					offset_start_place = 0.12
				else:
					offset_start_place = 0.10

				if end_place in last_squares or end_place == 'box': #QUANDO DOVREBBE VERIFICARSI CHE END_PLACE SIA BOX?
					offset_end_place = 0.12
				else:
					offset_end_place = 0.10

				if start_place in first_squares and end_place in last_squares:
					offset_start_place = 0.10
					offset_end_place = 0.12
					exception_1_flag = True

				if start_place in last_squares and end_place in first_squares:
					offset_start_place = 0.12
					offset_end_place = 0.10
					exception_2_flag = True

				if end_place in last_squares and start_square not in first_squares:
					offset_start_place = 0.12
					offset_end_place = 0.12
					exception_3_flag = True

				if taken_piece == 'none': #If the end target place is free from pieces, it's a simple move or an en-passant move.
					#Implement the capture of the pawn if en-passant happened.
					if en_passant_square != 'none':
						self.tilt = 'none'
						taken_piece = self.live_chessboard_situation[en_passant_square][0] #Save the name of the pawn that is captured.
						self.motions.gripper_over_piece(en_passant_square, centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard)
						self.motions.playmotion_movement('open_gripper', planning = False)
						which_error = 1
						fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', -offset_start_place, avoid_collisions = False) #Lower the gripper		
						self.motions.pick_piece(taken_piece, self.pieces_coordinates) #Close the gripper to pick the piece.
						fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', offset_start_place, avoid_collisions = False) #Higher the gripper
						#self.motions.straight_movements_chessboard(self.motions.arm, centroids, 'box', squares_to_index, self.tilt, clock_pose, box_pose, avoid_collisions = False) #Move the arm with straight movements over the chessboard.
						#self.motions.execute_remembered_joint_values(self.motions.arm, 'arm_over_box')
						self.motions.gripper_over_box(box_pose, z_coord_chessboard)
						self.motions.place_piece('box') #Place the piece in the end square.
						taken_piece = 'none' #Go back to the old variable, to avoid going in the other loop.
						en_passant_square = 'none' #Change back the flag
					
					if start_place in last_squares:
						self.motions.gripper_over_piece_last_row(start_place, centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard, False)
						self.motions.playmotion_movement('open_gripper', planning = False)
						self.tilt = 'avanti'
					elif start_place in first_squares and end_place in last_squares:
						self.tilt = 'indietro'
						self.motions.gripper_over_piece_special(start_place, centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard)
						self.motions.playmotion_movement('open_gripper', planning = False)
					elif start_place not in first_squares and end_place in last_squares:
						self.tilt = 'avanti'
						self.motions.gripper_over_piece_last_row(start_place, centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard, False)
						self.motions.playmotion_movement('open_gripper', planning = False)
					else:
						self.tilt = 'none'
						self.motions.gripper_over_piece(start_place, centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard)
						self.motions.playmotion_movement('open_gripper', planning = False)
								
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', -offset_start_place, avoid_collisions = False) #Lower the gripper		
					self.motions.pick_piece(piece, self.pieces_coordinates) #Close the gripper to pick the piece.
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', offset_start_place, avoid_collisions = False) #Higher the gripper
					if not exception_2_flag and not exception_1_flag and not exception_3_flag:
						which_error = 10
						self.motions.straight_movements_chessboard(self.motions.arm, centroids, end_place, squares_to_index, self.tilt, clock_pose, box_pose, avoid_collisions = False) #Move the arm with straight movements over the chessboard.
					elif exception_1_flag:
						self.motions.gripper_over_piece_last_row(end_place, centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard, True) #Finish to move the arm.
					elif exception_2_flag:
						self.motions.gripper_over_piece_special(end_place, centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard) #Finish to move the arm.
					elif exception_3_flag:
						self.motions.gripper_over_piece_last_row(end_place, centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard, False) #Finish to move the arm.
					which_error = 1
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', -offset_end_place, avoid_collisions = False) #Lower the gripper					
					self.motions.place_piece(end_place) #Place the piece in the end square.
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', offset_end_place, avoid_collisions = False) #Higher the gripper
					self.motions.gripper_over_clock(clock_pose, z_coord_chessboard)
					self.motions.click_clock(self.scene, clock_pose)		

				else: #If the end target place is occupied by another piece, this means that that piece needs to be taken and put into the box.
					#Pick the captured piece and place it in the box.
					if end_place in last_squares:
						self.motions.gripper_over_piece_last_row(end_place, centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard, False)
						self.motions.playmotion_movement('open_gripper', planning = False)
						self.tilt = 'avanti'
					else:
						self.tilt = 'none'
						self.motions.gripper_over_piece(end_place, centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard)
						self.motions.playmotion_movement('open_gripper', planning = False)
					
					which_error = 1
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', -offset_end_place, avoid_collisions = False) #Lower the gripper
					self.motions.pick_piece(taken_piece, self.pieces_coordinates) #Close the gripper to pick the piece.
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', offset_end_place, avoid_collisions = False) #Higher the gripper
					#self.motions.straight_movements_chessboard(self.motions.arm, centroids, 'box', squares_to_index, self.tilt, clock_pose, box_pose, avoid_collisions = False) #CAMBIO:PIANIFICO IL MOVIMENTO SUL CLOCK
					self.motions.gripper_over_box(box_pose, z_coord_chessboard)
					#self.motions.execute_remembered_joint_values(self.motions.arm, 'arm_over_box')
					self.motions.place_piece('box') #Place the taken piece into the box.

					#Move the piece to the target square.
					if start_place in last_squares and end_place not in first_squares:
						self.motions.gripper_over_piece_last_row(start_place, centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard, False)
						self.motions.playmotion_movement('open_gripper', planning = False)
						self.tilt = 'avanti'
					elif start_place in last_squares and end_place in first_squares:
						self.motions.gripper_over_piece_last_row(start_place, centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard, True)
						self.motions.playmotion_movement('open_gripper', planning = False)
						self.tilt = 'avanti'
					elif start_place in first_squares and end_place in last_squares:
						self.tilt = 'indietro' #indietro
						self.motions.gripper_over_piece_special(start_place, centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard)
						self.motions.playmotion_movement('open_gripper', planning = False)
					elif start_place not in first_squares and end_place in last_squares:
						self.tilt = 'avanti'
						self.motions.gripper_over_piece_last_row(start_place, centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard, False)
						self.motions.playmotion_movement('open_gripper', planning = False)
					else:
						self.tilt = 'none'
						self.motions.gripper_over_piece(start_place, centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard)
						self.motions.playmotion_movement('open_gripper', planning = False)

					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', -offset_start_place, avoid_collisions = False) #Lower the gripper
					self.motions.pick_piece(piece, self.pieces_coordinates)
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', offset_start_place, avoid_collisions = False) #Higher the gripper
					if not exception_2_flag and not exception_1_flag and not exception_3_flag:
						which_error = 11
						self.motions.straight_movements_chessboard(self.motions.arm, centroids, end_place, squares_to_index, self.tilt, clock_pose, box_pose, avoid_collisions = False) #Move the arm with straight movements over the chessboard.
					elif exception_1_flag:
						self.motions.gripper_over_piece_last_row(end_place, centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard, True)
					elif exception_2_flag: 
						self.motions.gripper_over_piece_special(end_place, centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard) #Finish to move the arm.
					elif exception_3_flag:
						self.motions.gripper_over_piece_last_row(end_place, centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard, False)
					which_error = 1
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', -offset_end_place, avoid_collisions = False) #Lower the gripper			
					self.motions.place_piece(end_place) #Place the piece in the end square.
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', offset_end_place, avoid_collisions = False) #Higher the gripper
					self.motions.gripper_over_clock(clock_pose, z_coord_chessboard)		
					self.motions.click_clock(self.scene, clock_pose)

					

			print('Move execution time: '+str(time() - tempo_mossa)+'s')		
			rospy.sleep(5) #TOVERIFY IF THIS WAITING TIME IS ENOUGH.

			state_publisher.publish(14) #Change the status to the status of waiting for the pushbutton to be pressed, to start looking for the move made by the opponent.
		
		except StraightMovementError:
			rospy.loginfo('The straight movement startegy was not successful. Try planning the movement.')
			#If the error comes from a stright movement over the chessboard, try planning the movement
			if which_error != 1:	### TODO: error dictionary to display meaningful strings
				if which_error == 2: 
					self.motions.gripper_over_piece('g1', centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard)
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', -offset_end_place, avoid_collisions = False) #Lower the gripper					
					self.motions.place_piece('g1') #Place the piece in the end square.
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', offset_end_place, avoid_collisions = False) #Higher the gripper
					#Move the rook
					piece = 'rook_h1'
					self.motions.gripper_over_piece('h1', centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard)
					self.motions.playmotion_movement('open_gripper', planning = False)
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', -offset_start_place, avoid_collisions = False) #Lower the gripper		
					self.motions.pick_piece(piece, self.pieces_coordinates) #Close the gripper to pick the piece.
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', offset_start_place, avoid_collisions = False) #Higher the gripper
					which_error = 3
					self.motions.straight_movements_chessboard(self.motions.arm, centroids, 'f1', squares_to_index, self.tilt, clock_pose, box_pose, avoid_collisions = False) #Move the arm with straight movements over the chessboard.
					which_error = 1
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', -offset_end_place, avoid_collisions = False) #Lower the gripper					
					self.motions.place_piece('f1') #Place the piece in the end square.
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', offset_end_place, avoid_collisions = False) #Higher the gripper
					#click the clock
					self.motions.gripper_over_clock(clock_pose, z_coord_chessboard)
					self.motions.click_clock(self.scene, clock_pose)

				elif which_error == 3:
					self.motions.gripper_over_piece('f1', centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard)
					which_error = 1
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', -offset_end_place, avoid_collisions = False) #Lower the gripper					
					self.motions.place_piece('f1') #Place the piece in the end square.
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', offset_end_place, avoid_collisions = False) #Higher the gripper
					#click the clock
					self.motions.gripper_over_clock(clock_pose, z_coord_chessboard)
					self.motions.click_clock(self.scene, clock_pose)

				elif which_error == 4:
					self.motions.gripper_over_piece('c1', centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard)
					which_error = 1
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', -offset_end_place, avoid_collisions = False) #Lower the gripper					
					self.motions.place_piece('c1') #Place the piece in the end square.
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', offset_end_place, avoid_collisions = False) #Higher the gripper

					#Move the rook
					piece = 'rook_a1'
					self.motions.gripper_over_piece('a1', centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard)
					self.motions.playmotion_movement('open_gripper', planning = False)
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', -offset_start_place, avoid_collisions = False) #Lower the gripper		
					self.motions.pick_piece(piece, self.pieces_coordinates) #Close the gripper to pick the piece.
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', offset_start_place, avoid_collisions = False) #Higher the gripper
					which_error = 5
					self.motions.straight_movements_chessboard(self.motions.arm, centroids, 'd1', squares_to_index, self.tilt, clock_pose, box_pose, avoid_collisions = False) #Move the arm with straight movements over the chessboard.
					which_error = 1
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', -offset_end_place, avoid_collisions = False) #Lower the gripper					
					self.motions.place_piece('d1') #Place the piece in the end square.
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', offset_end_place, avoid_collisions = False) #Higher the gripper
					#Click the clock
					self.motions.gripper_over_clock(clock_pose, z_coord_chessboard)
					self.motions.click_clock(self.scene, clock_pose)

				elif which_error == 5:
					self.motions.gripper_over_piece('d1', centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard)
					which_error = 1
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', -offset_end_place, avoid_collisions = False) #Lower the gripper					
					self.motions.place_piece('d1') #Place the piece in the end square.
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', offset_end_place, avoid_collisions = False) #Higher the gripper
					#Click the clock
					self.motions.gripper_over_clock(clock_pose, z_coord_chessboard)
					self.motions.click_clock(self.scene, clock_pose)

				elif which_error == 6: 
					self.motions.gripper_over_piece('g8', centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard)
					which_error = 1
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', -offset_end_place, avoid_collisions = False) #Lower the gripper					
					self.motions.place_piece('g8') #Place the piece in the end square.
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', offset_end_place, avoid_collisions = False) #Higher the gripper
					#Move the rook
					piece = 'rook_h8'
					self.motions.gripper_over_piece('h8', centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard)
					self.motions.playmotion_movement('open_gripper', planning = False)
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', -offset_start_place, avoid_collisions = False) #Lower the gripper		
					self.motions.pick_piece(piece, self.pieces_coordinates) #Close the gripper to pick the piece.
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', offset_start_place, avoid_collisions = False) #Higher the gripper
					which_error = 7
					self.motions.straight_movements_chessboard(self.motions.arm, centroids, 'f8', squares_to_index, self.tilt, clock_pose, box_pose, avoid_collisions = False) #Move the arm with straight movements over the chessboard.
					which_error = 1
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', -offset_end_place, avoid_collisions = False) #Lower the gripper					
					self.motions.place_piece('f8') #Place the piece in the end square.
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', offset_end_place, avoid_collisions = False) #Higher the gripper
					#click the clock
					self.motions.gripper_over_clock(clock_pose, z_coord_chessboard)
					self.motions.click_clock(self.scene, clock_pose)

				elif which_error == 7:
					self.motions.gripper_over_piece('f8', centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard)
					which_error = 1
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', -offset_end_place, avoid_collisions = False) #Lower the gripper					
					self.motions.place_piece('f8') #Place the piece in the end square.
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', offset_end_place, avoid_collisions = False) #Higher the gripper
					#click the clock
					self.motions.gripper_over_clock(clock_pose, z_coord_chessboard)
					self.motions.click_clock(self.scene, clock_pose)

				elif which_error == 8:
					self.motions.gripper_over_piece('c8', centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard)
					which_error = 1
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', -offset_end_place, avoid_collisions = False) #Lower the gripper					
					self.motions.place_piece('c8') #Place the piece in the end square.
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', offset_end_place, avoid_collisions = False) #Higher the gripper
					#Move the rook
					piece = 'rook_a8'
					self.motions.gripper_over_piece('a8', centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard)
					self.motions.playmotion_movement('open_gripper', planning = False)
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', -offset_start_place, avoid_collisions = False) #Lower the gripper		
					self.motions.pick_piece(piece, self.pieces_coordinates) #Close the gripper to pick the piece.
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', offset_start_place, avoid_collisions = False) #Higher the gripper
					which_error = 9
					self.motions.straight_movements_chessboard(self.motions.arm, centroids, 'd8', squares_to_index, self.tilt, clock_pose, box_pose, avoid_collisions = False) #Move the arm with straight movements over the chessboard.
					which_error = 1
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', -offset_end_place, avoid_collisions = False) #Lower the gripper					
					self.motions.place_piece('d8') #Place the piece in the end square.
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', offset_end_place, avoid_collisions = False) #Higher the gripper
					#Click the clock
					self.motions.gripper_over_clock(clock_pose, z_coord_chessboard)
					self.motions.click_clock(self.scene, clock_pose)

				elif which_error == 9:
					self.motions.gripper_over_piece('d8', centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard)
					which_error = 1
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', -offset_end_place, avoid_collisions = False) #Lower the gripper					
					self.motions.place_piece('d8') #Place the piece in the end square.
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', offset_end_place, avoid_collisions = False) #Higher the gripper
					#Click the clock
					self.motions.gripper_over_clock(clock_pose, z_coord_chessboard)
					self.motions.click_clock(self.scene, clock_pose)

				elif which_error == 10:
					self.motions.gripper_over_piece(end_place, centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard)
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', -offset_end_place, avoid_collisions = False) #Lower the gripper					
					self.motions.place_piece(end_place) #Place the piece in the end square.
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', offset_end_place, avoid_collisions = False) #Higher the gripper
					self.motions.gripper_over_clock(clock_pose, z_coord_chessboard)
					self.motions.click_clock(self.scene, clock_pose)
				elif which_error == 11:
					self.motions.gripper_over_piece(end_place, centroids, self.live_chessboard_situation, squares_to_index, z_coord_chessboard)
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', -offset_end_place, avoid_collisions = False) #Lower the gripper			
					self.motions.place_piece(end_place) #Place the piece in the end square.
					fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', offset_end_place, avoid_collisions = False) #Higher the gripper
					self.motions.gripper_over_clock(clock_pose, z_coord_chessboard)		
					self.motions.click_clock(self.scene, clock_pose)

				rospy.sleep(5) #TOVERIFY IF THIS WAITING TIME IS ENOUGH.

				state_publisher.publish(14) #Change the status to the status of waiting for the pushbutton to be pressed, to start looking for the move made by the opponent.

			else:
				#If the error comes from highering or lowering the gripper, force the sdtate to 11 and ask the user to manually change the GUI chessboard and to perform the move again.
				
				rospy.loginfo('Go back to state 11, manually change the GUI and ask TIAGo to perform the move another time.')
				chess_move_not_exectuted_yet = True
				state_publisher.publish(11)
				'''
				rospy.loginfo('The staright movement was not completed, go back to the rest pose and try performing the move again')
				#Perform a straight movement opposite to the one that TIAGo was able to complete.
				fraction, translation = self.motions.straight_eef_movement(self.motions.arm, 'z', fraction * translation, avoid_collisions = False)
				self.motions.gripper_over_clock(clock_pose, z_coord_chessboard)
				chess_move_not_exectuted_yet = True
				state_publisher.publish(11)
				'''


	def reset(self, strategy): #TODO
		#Remove the attached object from the grasping frame, clear the planning scene from said object.
		#Then, based on user input, either tuck the arm or just move the end-effector some cm away.
		#Finally, remove the objects from the planning scene and clear the octomap if it is requested.
		#Go to a safe position
		self.scene.remove_attached_object(link = 'gripper_grasping_frame') #In case the object is still attached
		#self.motions.move_joints(arm_torso, joint_values = [0.30], joints_to_move = ['torso_lift_joint']) #NON SERVE PERCHE' TIAGO E' GIA' SU AL MASSIMO
		#Perform one of the possible exit strategies depending on the corresponding keyvalue in the configuration .yaml file
		if strategy == 'nearby':
			rospy.loginfo('Planning to reach the \'nearby\' homing position.')
			goal_pose = self.motions.arm.get_current_pose(end_effector_link = 'gripper_grasping_frame')
			if goal_pose.header.frame_id == 'base_footprint':
				#goal_pose.pose.position.y -= 0.3 #QUESTI VALORI VANNO BENE ANCHE PER LA MIA APPLICAZIONE??
				goal_pose.pose.position.z += 0.1 #QUESTI VALORI VANNO BENE ANCHE PER LA MIA APPLICAZIONE??
				success, __ = self.motions.eef_to_valid_pose(self.motions.arm, goal_pose.pose.position, goal_pose.pose.orientation, 'z', 0.05, 0.10)
				if success:
					rospy.loginfo('The \'nearby\' homing position has been successfully reached.')
				else:
					rospy.logerr('Impossible to reach the \'nearby\' homing position.')
			else:
				rospy.logerr('The current pose is in the ' + goal_pose.header.frame_id + '. The \'nearby\' homing is meant to work in the \'base_footprint\' frame, hence it won\'t be executed.')
		elif strategy == 'home':
			#Bring the arm far from the table
			self.motions.execute_remembered_joint_values(self.motions.arm, 'open_wide_right')
			#Bring the arm close to the robot body
			self.motions.playmotion_movement('home', timeout = 20) #Be careful: this is pre-recorded.
		else:
			rospy.logerr('No valid exit strategy was given, the robot is going to stay there.')

		if self.clear_all:
			rospy.loginfo('Clearing the octomap.')
			rospy.wait_for_service('/clear_octomap')                        # keeping the octomap for a certain number of iteration,
			clear_octomap = rospy.ServiceProxy('/clear_octomap', Empty)     # (in a controlled environment) would spare some time
			clear_octomap()





if __name__ == '__main__':
	global tempo_mossa
	try:
		while not rospy.is_shutdown():
			if state.data == 1: #State of TIAGO initialization.
				grasper = Grasping()
				grasper.initialize_TIAGo() #Extend torso, look down, inspect surroundings (if necessary).
				#NB: the state is changed in grasp_piece.py

			elif state.data == 4: #State of chessboard segmentation.
				if segmentation_not_launched:
					segmentation_launcher = roslauncher.roslaunch_from_file(PACKAGE_DIR + '/launch/segmentation.launch')
					segmentation_not_launched = False

			elif state.data == 5: #State of asking confirmation for the chessboard segmentation
				segmentation_launcher.shutdown() #Shutdown the segmentation node.
				segmentation_not_launched = True

				#Launch the CV depth images processing.
				if CV_launcher_not_launched:
					CV_launcher = roslauncher.roslaunch_from_file(PACKAGE_DIR + '/launch/CV_move_recognition.launch')
					print(PACKAGE_DIR + '/launch/CV_move_recognition.launch')
					CV_launcher_not_launched = False
			
			elif state.data == 7: #State of search for ARUCO markers.
				if aruco_detector_not_launched:
					aruco_launcher = roslauncher.roslaunch_from_file(PACKAGE_DIR + '/launch/multi_detector.launch')
					aruco_detector_not_launched = False
				if pc_saver_not_launched:
					pc_saver_launcher = roslauncher.roslaunch_from_file(PACKAGE_DIR + '/launch/pc_saver.launch')
					pc_saver_not_launched = False

			elif state.data == 40: #State of asking for the markers localization confirmation.
				aruco_launcher.shutdown() #Shutdown the markers searcher node.
				aruco_detector_not_launched = True

			elif state.data == 9: #State of TIAGo's arm preparation to grasp pieces.
				with open(z_coord_chessboard_mean_file) as file:
						z_coord_chessboard_mean = yaml.load(file)
				with open(clock_pose_file) as file:
						clock_marker_pose = yaml.load(file)
				with open(box_pose_file) as file:
						box_marker_pose = yaml.load(file)
				escape_strategy = grasper.prepare_TIAGo(clock_marker_pose, z_coord_chessboard_mean, box_marker_pose) #Prepare TIAGo to grasp pieces.

			elif state.data == 13: #State of move execution by TIAGo.
				if chess_move_not_exectuted_yet:
					tempo_mossa = time()
					player = Playing()

					#Load the previously computed parameters
					#if transformed_centroids is None:
					with open(simul_config) as file:
						transformed_centroids = yaml.load(file)
					
					#if self.box_marker_pose is None:
					with open(box_pose_file) as file:
						box_marker_pose = yaml.load(file)

					if ready:
						with open(clock_pose_file) as file:
							clock_marker_pose = yaml.load(file)
						with open(z_coord_chessboard_mean_file) as file:
							z_coord_chessboard_mean = yaml.load(file)

					player.chess_move(start_square, end_square, transformed_centroids, clock_marker_pose, box_marker_pose, z_coord_chessboard_mean) #Execute the complete move. #TOVERIFY: TIENE TRANSFORMED_CENTRIDS?
					chess_move_not_exectuted_yet = False

			elif state.data == 14: #State of waiting for the opponent move.
				chess_move_not_exectuted_yet = True

	except KeyboardInterrupt:
		rospy.logwarn('Shutting down the play chess node')


