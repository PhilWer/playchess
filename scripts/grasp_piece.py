#!/usr/bin/env python
#Script to make TIAGo grasp pieces on the chess board

#Import ROS libraries
import rospy, roslaunch, rosnode
import moveit_commander

#Import Python libraries
import sys
import yaml
import math
import time
import copy

#Import ROS messages
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal
from actionlib import SimpleActionClient
from geometry_msgs.msg import Quaternion, PointStamped, PoseStamped, Pose, Point, Twist
from std_msgs.msg import Bool, Int16, String
from moveit_commander import PlanningSceneInterface
from moveit_commander.exception import MoveItCommanderException
from sensor_msgs.msg import JointState

#My scripts
import config as cfg #real_config
import roslauncher as roslauncher
import quaternions_and_points as qp
#from save_pose import SavePose
#from manual_navigation import ManualNavigation
from errors import ArucoException, PlanningException, StraightMovementError

#ready = False

PLAYCHESS_PKG_DIR = '/home/luca/tiago_public_ws/src/playchess'

saved_joints_pose_file = PLAYCHESS_PKG_DIR + '/config/saved_joints_pose_file.yaml'
simul_config = rospy.get_param('/playchess/simul_config')

# Callbacks definition
def CallbackColor(data):
	color.data = data.data
	if color.data == 'white':
		squares_to_index = cfg.squares_to_index_white
	elif color.data == 'black':
		squares_to_index = cfg.squares_to_index_black
	pieces_coordinates = cfg.pieces_coordinates

#Publishers initialization
state_publisher = rospy.Publisher('/state', Int16, queue_size = 10)

#Subscribers initialization
rospy.Subscriber('/color', String, CallbackColor)

#Messages initialization
color = String()
color.data = 'none'

class Grasping:
# Class containing the different movements that TIAGo has to perform to grasp pieces
	def __init__(self):
		# Initialize a MoveIt commander
		moveit_commander.roscpp_initialize(sys.argv)
		# MoveIt utilities
		# Define TIAGo interfaces and commanders for each group of the robot 
		self.robot = moveit_commander.RobotCommander()
		self.scene = moveit_commander.PlanningSceneInterface() #The interface with the world surrounding the robot
		self.gripper = moveit_commander.MoveGroupCommander('gripper')
		self.arm = moveit_commander.MoveGroupCommander('arm')
		self.arm.set_goal_orientation_tolerance(0.05) #Set the tolerance for the orientation of the gripper to 0.1 radians
		self.arm.set_goal_position_tolerance(0.005) #Set the tolerance for the position of the gripper
		self.arm.set_max_velocity_scaling_factor(0.2) #Lower the velocity of the arm.
		self.arm_torso = moveit_commander.MoveGroupCommander('arm_torso')
		self.head_cmd = rospy.Publisher('/head_controller/command', JointTrajectory, queue_size = 1)
		self.torso_cmd = rospy.Publisher('/torso_controller/command', JointTrajectory, queue_size = 1)
		self.arm_cmd = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size = 1)

		# Create an action client for PlayMotionAction
		self.playmotion = SimpleActionClient("play_motion", PlayMotionAction)
		# Wait until the action sever has started
		if not self.playmotion.wait_for_server(rospy.Duration(30)):
			rospy.logerr("Could not connect to /play_motion")
			exit()
		rospy.sleep(2)

		# Remember some meaniningful joint values
		self.arm.remember_joint_values('prepare_chess_grasp', values = [1.40, 0.70, -0.50, 0.50, -2.00, 1.00, 0.0]) #Arm in resting pose
		self.arm.remember_joint_values('open_arm', values = [0.20, 0.70, -0.20, 0.50, -1.50, 0.0, 0.0]) #Open the arm
		self.arm.remember_joint_values('open_wide_right', values = [math.pi/40, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

		# Import pieces characteristics
		self.pawn = cfg.pawn
		self.rook = cfg.rook
		self.bishop = cfg.bishop
		self.king = cfg.king
		self.queen = cfg.queen
		self.knight = cfg.knight

		# Load configurations
		self.config_file = rospy.get_param('/playchess/config')
		with open(self.config_file) as file:
			self.config = yaml.load(file)
		self.simulation = self.config.get('simulation') #Is the code running in simulation mode?
		self.aruco = self.config.get('aruco', False) #Is the robot looking for ARUCO markers? In simulation TIAGo can also not look for them.
		self.color = self.config.get('color') #Is TIAGo playing with white or black?
		self.offset_last_row = 0.03
		self.success_clock = False
		self.verbose = False

		# Load build/clear scene and octomap configurations
		self.time_limit = self.config.get('time_limit', 3) #Maximum time to wait when looking for the object or for ArUco markers
		self.build_full_scene = self.config.get('full_setup', True)  #if False, the 'explore_surroundings' movement is not performed. Useful to save time in laboratory.
		self.clear_all = self.config.get('clear_octomap', False) #Clear the octomap at the end of the task (defaulted to False)

		# Load motion strategies
		self.desired_escape = self.config.get('escape_strategy', 'nearby')
		self.current_escape = None #The escape_strategy remains None (i.e. 'stay still') until the eef is placed close to the object
		self.pick_gripper_orientation = qp.list_to_Quaternion(self.config.get('GRIPPER_ORIENT', [0, 0.707, 0, 0.707]))
		self.pick_gripper_orientation_last_row = qp.list_to_Quaternion(self.config.get('GRIPPER_ORIENT_LAST_ROW', [0, 0.538, 0, 0.843])) #[-0.538, 0, 0.843, 0]
		self.pick_gripper_orientation_last_row_second_opt = qp.list_to_Quaternion(self.config.get('GRIPPER_ORIENT_LAST_ROW_SECOND_OPT', [0, 0.5, 0, 0.866])) #[0, 0.5, 0, 0.866]
		self.prova = qp.list_to_Quaternion(self.config.get('GRIPPER_ORIENT_LAST_ROW_SECOND_OPT', [0.843, 0, -0.537, 0])) #[0, 0.5, 0, 0.866]
		self.pick_gripper_orientation_first_row = qp.list_to_Quaternion(self.config.get('GRIPPER_ORIENT_LAST_ROW', [0, -0.844, 0, -0.537]))
		self.pick_gripper_orientation_first_row_prova = qp.list_to_Quaternion(self.config.get('GRIPPER_ORIENT_LAST_ROW', [0, -0.819, 0, -0.574])) #70 gradi
		self.pick_gripper_orientation_last_row_prova = qp.list_to_Quaternion(self.config.get('GRIPPER_ORIENT_LAST_ROW', [0, -0.574, 0, -0.819])) #70 gradi
		if self.desired_escape == 'home':
		    self.arm.remember_joint_values('tuck', values = [0.191986, -1.3439, -0.191986, 1.93732, -math.pi/2, 1.37881, 0.0])

	def back_to_initial_state(self):
		# Bring back TIAGo to its initial state
		rospy.sleep(5)
		rospy.loginfo("Back to initial state")
		goal = PlayMotionGoal()
		goal.motion_name = 'home'
		goal.skip_planning = False
		self.playmotion.send_goal_and_wait(goal)

	def click_clock(self, scene, clock_pose):
		'''
		Click the clock after a move is made. When this function is called, the gripper must be
		already positioned above the clock. 
		'''
		### TODO: remove unused arguments scene and clock_pose
		### TODO: expose the geometrical params
		rospy.loginfo("Closing the gripper")
		self.playmotion_movement('close_gripper', planning = False)		
		rospy.loginfo("Lowering the gripper to click the button")
		self.straight_eef_movement(self.arm, 'z', -0.095, avoid_collisions = False) #Lower the gripper
		rospy.loginfo("Highering back the gripper over the clock")
		self.straight_eef_movement(self.arm, 'z', 0.095, avoid_collisions = False) #Higher the gripper
		rospy.loginfo("Opening the gripper back")
		self.playmotion_movement('open_gripper', planning = False)
		# Move the arm away to see the chessboard
		self.move_arm_joint(0.471) #30 degrees
		rospy.sleep(4)
		self.look_down()

	def eef_to_valid_pose(self, group, eef_target_position, eef_target_orientation, exploration_direction, shift_amount, max_shift):
		'''
		Bring the end-effector to the given pose.
		eef_target_position: [Point] the point to which the hand_grasping_frame has to be positioned
		eef_target_orientation: [Quaternion] the orientation of the hand in the final point
		shift_amount: [float] if the eef_target_position cannot be reached, the target point is raised of shift_amount [m] until a valid position is reached
		max_shift: [float] a limit (in [m]) to the overall shift
		'''
		success = False
		valid_height_value = True
		total_shift = 0.0
		eef_target = Pose(eef_target_position, eef_target_orientation)
		while (total_shift <= max_shift) and not success:  #Try to grasp the bottle at the eef_target_position, if it is not possible, raise the grasping point until a reasonable limit
			group.set_pose_target(eef_target)
			#Plan and execute the action
			plan = group.plan()
			if plan.joint_trajectory.points:  #True if trajectory contains points
				rospy.loginfo("Bringing the end-effector to:\n" + str(eef_target))
				group.execute(plan)
				group.stop()  #Be sure that there is no residual movement
				success = True
			else:
				rospy.logwarn('Try planning again with different target. The grasping target is shifted ' + str(shift_amount) + ' above with respect to the original.')   
				#Shift the target point until a valid one its found
				if exploration_direction == 'z':
					eef_target.position.z += shift_amount
				if exploration_direction == 'y':
					eef_target.position.y += shift_amount
				if exploration_direction == 'x':
					eef_target.position.x += shift_amount
				#Track the total shift wrt the initial point
				total_shift += shift_amount
			return success, eef_target.position

	def execute_remembered_joint_values(self, group, name):
		'''
		Plan and execute a trajectory to bring the arm to a remembered joint value
		'''
		if not name in group.get_remembered_joint_values():
			print('\'' + name + '\' is not a remembered joint value.')
		else:
			#group.set_max_velocity_scaling_factor(SLOWDOWN_FACTOR)
			plan = group.plan(name)
		if plan.joint_trajectory.points: # True if trajectory contains points
			group.execute(plan)
		else:
			print('The \'' + name +'\' joint value cannot be safely reached now.')

	def extend_torso(self, elevation = 0.35):
		### TODO: make it a simpler message publisher (with control)
		rospy.loginfo("Extending the torso up")
		if elevation < 0.0:
			elevation = 0.0
		elif elevation > 0.35:
			elevation = 0.35
		# Get the joint values of the torso and the arm and lift the torso to the desired elevation
		target_joint_values = self.arm_torso.get_current_joint_values()
		target_joint_values[0] = elevation
		self.arm_torso.set_max_velocity_scaling_factor(1.0) # torso movements are considered to be 
															# safe, perform them at full speed
		try:
			self.arm_torso.set_joint_value_target(target_joint_values)
		except MoveItCommanderException as e:
			print(e)
		# Lift the torso
		self.arm_torso.go() 
		rospy.loginfo('Torso extension is ' + str(elevation) + 'm')
		self.arm_torso.stop()

	def gripper_over_box (self, box_pose, z_coord_chessboard):
		# Bring the end-effector to the given pose.
		# clock_pose: pose of the aruco marker positioned over the button of the clock.

		self.success_box = False
		self.arm.set_end_effector_link('gripper_grasping_frame')
		
		count = 0
		while not self.success_box:
			print('Planning')
			target = Pose(qp.list_to_Point([(box_pose.position.x + count) - 0.02 , (box_pose.position.y - count) + 0.02, z_coord_chessboard + 0.2 + count]), self.pick_gripper_orientation_last_row) #+0.02 in dir x
			self.arm.set_pose_target(target)
			plan = self.arm.plan()
			if plan.joint_trajectory.points:
				self.arm.execute(plan)
				self.arm.stop() #Make sure there's no residual movement
				self.success_box = True
			else:
				rospy.logerr('Cannot perform the movement')
			count += 0.07

	def gripper_over_clock (self, clock_pose, z_coord_chessboard):
		#Bring the end-effector to the given pose.
		#clock_pose: pose of the aruco marker positioned over the button of the clock.

		self.success_clock = False
		self.arm.set_end_effector_link('gripper_grasping_frame')
		target = Pose(qp.list_to_Point([clock_pose.position.x - 0.06, clock_pose.position.y, z_coord_chessboard + 0.2]), self.pick_gripper_orientation_last_row) #0.16 E' IL LIMITE PIU' BASSO #- 0.06
		while not self.success_clock:
			print('Planning')
			self.arm.set_pose_target(target)
			plan = self.arm.plan()
			if plan.joint_trajectory.points:
				self.arm.execute(plan)
				self.arm.stop() #Make sure there's no residual movement
				self.success_clock = True
			else:
				rospy.logerr('Cannot perform the movement')
		self.playmotion_movement('open_gripper', planning = False)

	def gripper_over_piece (self, casella_target, centroids, live_chessboard_situation, squares_to_index, z_coord_chessboard):
		#Bring the end-effector to the given pose.
		#casella_target: chessboard coordiantes where the gripper has to be positioned.

		#Make TIAGo look up to avoid collisions between head and arm during the move execution.
		self.look_up()
		rospy.sleep(1)

		piece = live_chessboard_situation[casella_target]
		index = squares_to_index[casella_target]
		#piece_pose = scene.get_object_poses([piece])
		self.success = False
		self.arm.set_end_effector_link('gripper_grasping_frame')
		end_position = [centroids[index].point.x, centroids[index].point.y, z_coord_chessboard + 0.2]
		#target = Pose(qp.list_to_Point([piece_pose[piece].position.x, piece_pose[piece].position.y, piece_pose[piece].position.z + 0.2]), self.pick_gripper_orientation) #0.16 E' IL LIMITE PIU' BASSO
		target = Pose(qp.list_to_Point(end_position), self.pick_gripper_orientation) #0.16 E' IL LIMITE PIU' BASSO   [piece_pose[piece].position.x, piece_pose[piece].position.y, z_coord_chessboard + 0.2]
		if self.verbose:
			print('Target piece pose: ' + str(target))
		while not self.success:
			print('Planning')
			self.arm.set_pose_target(target)
			plan = self.arm.plan()
			if plan.joint_trajectory.points:
				self.arm.execute(plan)
				self.arm.stop() #Make sure there's no residual movement
				self.success = True
			else:
				rospy.logwarn('Cannot perform the movement')
				state_publisher.publish(11)
				break

		#self.playmotion_movement('open_gripper', planning = False)

	def gripper_over_piece_last_row (self, casella_target, centroids, live_chessboard_situation, squares_to_index, z_coord_chessboard, inverted):
		#Bring the end-effector to the given pose.
		#casella_target: chessboard coordiantes where the gripper has to be positioned.
		#inverted: [Bool] False if the gripper arrives with normal orientation, True if it arrives inverted.

		#Make TIAGo look up to avoid collisions between head and arm during the move execution.
		self.look_up()
		rospy.sleep(1)

		piece = live_chessboard_situation[casella_target]
		self.success = False
		index = squares_to_index[casella_target]
		#end_position = [centroids[index].point.x - 0.015, centroids[index].point.y, centroids[index].point.z + 0.2]
		end_position = [centroids[index].point.x - self.offset_last_row, centroids[index].point.y, z_coord_chessboard + 0.2]
		self.arm.set_end_effector_link('gripper_grasping_frame')
		if not inverted:
			target = Pose(qp.list_to_Point(end_position), self.pick_gripper_orientation_last_row)
			if self.verbose:
				print('Target piece pose: ' + str(target))		
			while not self.success:
				print('Planning')
				target = Pose(qp.list_to_Point(end_position), self.pick_gripper_orientation_last_row_second_opt)
				self.arm.set_pose_target(target)
				plan = self.arm.plan()
				if plan.joint_trajectory.points:
					self.arm.execute(plan)
					self.arm.stop() #Make sure there's no residual movement
					self.success = True
				else:
					#rospy.logerr('Cannot perform the movement')
					rospy.logwarn('Cannot perform the movement')
					state_publisher.publish(11)
					break

		elif inverted:
			target = Pose(qp.list_to_Point(end_position), self.prova)
			if self.verbose:
				print('Target piece pose: ' + str(target))		
			while not self.success:
				print('Planning')
				target = Pose(qp.list_to_Point(end_position), self.prova) #pick_gripper_orientation_last_row
				self.arm.set_pose_target(target)
				plan = self.arm.plan()
				if plan.joint_trajectory.points:
					self.arm.execute(plan)
					self.arm.stop() #Make sure there's no residual movement
					self.success = True
				else:
					#rospy.logerr('Cannot perform the movement')
					rospy.logwarn('Cannot perform the movement')
					state_publisher.publish(11)
					break

		#self.playmotion_movement('open_gripper', planning = False)

	def gripper_over_piece_special (self, casella_target, centroids, live_chessboard_situation, squares_to_index, z_coord_chessboard):
		#Bring the end-effector to the given pose.
		#It's a special grasp because it happens only when a piece in the first rows needs to go to the last rows.
		#casella_target: chessboard coordiantes where the gripper has to be positioned.

		#Make TIAGo look up to avoid collisions between head and arm during the move execution.
		self.look_up()
		rospy.sleep(1)

		piece = live_chessboard_situation[casella_target]
		self.success = False
		index = squares_to_index[casella_target]
		#end_position = [centroids[index].point.x - 0.015, centroids[index].point.y, centroids[index].point.z + 0.2]
		end_position = [centroids[index].point.x + self.offset_last_row, centroids[index].point.y, z_coord_chessboard + 0.2]
		self.arm.set_end_effector_link('gripper_grasping_frame')
				
		while not self.success:
			print('Planning')
			target = Pose(qp.list_to_Point(end_position), self.pick_gripper_orientation_first_row)
			if self.verbose:
				print('Target piece pose: ' + str(target))
			self.arm.set_pose_target(target)
			plan = self.arm.plan()
			if plan.joint_trajectory.points:
				self.arm.execute(plan)
				self.arm.stop() #Make sure there's no residual movement
				self.success = True
			else:
				rospy.logwarn('Cannot perform the movement')
				target = Pose(qp.list_to_Point(end_position), self.pick_gripper_orientation_first_row_prova)
				if self.verbose:
					print('Target piece pose: ' + str(target))
				self.arm.set_pose_target(target)
				plan = self.arm.plan()
				if plan.joint_trajectory.points:
					self.arm.execute(plan)
					self.arm.stop() #Make sure there's no residual movement
					self.success = True
				else:
					rospy.logwarn('Cannot perform the movement')
					state_publisher.publish(11)
					break #DA VERIFICARE CHE FUNZIONI: SE NON PUO' FARE IL MOVIMENTO, RITORNA ALLO STATO IN CUI ASPETTA LA MOSSA DA FARE.
		#self.playmotion_movement('open_gripper', planning = False)

	def initialize_TIAGo(self):
		#Initialize TIAGo making him extend the torso, look down and look around to inspect surroundings.
		#self.extend_torso(0.35) ###DA COMMENTARE
		print('TIAGO INITIALIZATION')
		self.look_down()
		#if self.build_full_scene: ###DA COMMENTARE
			#self.look_around() ###DA COMMENTARE

		#rospy.sleep(2)
		
		state_publisher.publish(60) #Pass to the state of publishing the current chessboard setup.

		time.sleep(3)


	def look_around(self):
		#Move the head of TIAGo down exploiting a pre-defined PlayMotion
		rospy.loginfo("Moving the head around to inspect surroundings")
       		goal = PlayMotionGoal()
        	goal.motion_name = 'inspect_surroundings'
		goal.skip_planning = True
                self.playmotion.send_goal_and_wait(goal)

	def look_down(self):
		#Move the head of TIAGo down exploiting a pre-defined PlayMotion
		rospy.loginfo("Moving the head down")
		joint_trajectory = JointTrajectory()
		joint_trajectory.joint_names = ['head_1_joint', 'head_2_joint']
		joint_trajectory_point = JointTrajectoryPoint()
		joint_trajectory_point.positions = [0.0, -0.98]
		joint_trajectory_point.time_from_start = rospy.Duration(1.0)
		joint_trajectory.points.append(joint_trajectory_point)
		self.head_cmd.publish(joint_trajectory)

	def look_right(self): ###TODO: POTREBBE NON SERVIRE
		#Move the head of TIAGo right exploiting a pre-defined PlayMotion
		rospy.loginfo("Turning the head right")
		joint_trajectory = JointTrajectory()
		joint_trajectory.joint_names = ['head_1_joint', 'head_2_joint']
		joint_trajectory_point = JointTrajectoryPoint()
		joint_trajectory_point.positions = [-0.80, -0.98]
		joint_trajectory_point.time_from_start = rospy.Duration(1.0)
		joint_trajectory.points.append(joint_trajectory_point)
		self.head_cmd.publish(joint_trajectory)

	def look_up(self):
		#Move the head of TIAGo up exploiting a pre-defined PlayMotion
		rospy.loginfo("Moving the head up")
		joint_trajectory = JointTrajectory()
		joint_trajectory.joint_names = ['head_1_joint', 'head_2_joint']
		joint_trajectory_point = JointTrajectoryPoint()
		joint_trajectory_point.positions = [0.0, 0.0]
		joint_trajectory_point.time_from_start = rospy.Duration(1.0)
		joint_trajectory.points.append(joint_trajectory_point)
		self.head_cmd.publish(joint_trajectory)

	def move_joints(self, group, joint_values = [], joints_to_move = []):
		#Bring the hand to the desired configuration. A sequence of configuration can be given to make the hand perform a sequence of movements.
		#postures: [[float]] joint values [rad]
		#positions: [[float]] joint positions. 
		#joint_names: [[string]] joint names

		joint_names = group.get_active_joints()
		target_joint_values = group.get_current_joint_values()
		for idx, joint_name in enumerate(joint_names):
			if joint_name in joints_to_move:
				target_joint_values[idx] = joint_values[joints_to_move.index(joint_name)]
		group.set_joint_value_target(target_joint_values) 
		group.go()
		group.stop()
		#Check if the movement has been performed as expected
		current_joint_values = group.get_current_joint_values()
		#Compute the error (absolute and relative) for each joint
		absolute_errors = [current_joint_value - target_joint_value for current_joint_value, target_joint_value in zip(current_joint_values, target_joint_values)]
		relative_errors = [(current_joint_value - target_joint_value) / target_joint_value for current_joint_value, target_joint_value in zip(current_joint_values, target_joint_values)]
		return absolute_errors, relative_errors

	def move_arm_joint(self, target):
		#Move joint 1 of TIAGo's arm.
		#Target: target degree of TIAGo's joint 1 of the arm.
		rospy.loginfo("Moving joint 1 of the arm")
		#Get the current state of the arm joints
		current_values = self.arm.get_current_joint_values()
		target_values = current_values
		target_values[0] = target

		joint_trajectory = JointTrajectory()
		joint_trajectory.joint_names = ['arm_1_joint', 'arm_2_joint', 'arm_3_joint', 'arm_4_joint', 'arm_5_joint', 'arm_6_joint', 'arm_7_joint']
		joint_trajectory_point = JointTrajectoryPoint()
		joint_trajectory_point.positions = target_values
		joint_trajectory_point.time_from_start = rospy.Duration(5.0)
		joint_trajectory.points.append(joint_trajectory_point)
		self.arm_cmd.publish(joint_trajectory)

	def move_wrist_joint(self, target_rotation):
		#Move joint 1 of TIAGo's arm.
		#target_rotation: target rotation of TIAGo's wrist.
		rospy.loginfo("Moving joint 7 of the arm")
		#Get the current state of the arm joints
		current_values = self.arm.get_current_joint_values()
		target_values = current_values
		if (target_values[6] + target_rotation) < 2: #119
			target_values[6] = target_values[6] + target_rotation
		elif (target_values[6] - target_rotation) > -2:
			target_values[6] = target_values[6] - target_rotation
		else:
			rospy.logwarn('The complete rotation of the wrist is not possible. Proceding with the maximum reachable rotation.')
			segm1 = (target_values[6] + target_rotation) - 2
			segm2 = -2 - (target_values[6] - target_rotation)
			if segm1 > segm2:
				target_values[6] = 2
			elif segm2 > segm1:
				target_values[6] = -2

		joint_trajectory = JointTrajectory()
		joint_trajectory.joint_names = ['arm_1_joint', 'arm_2_joint', 'arm_3_joint', 'arm_4_joint', 'arm_5_joint', 'arm_6_joint', 'arm_7_joint']
		joint_trajectory_point = JointTrajectoryPoint()
		joint_trajectory_point.positions = target_values
		joint_trajectory_point.time_from_start = rospy.Duration(5.0)
		joint_trajectory.points.append(joint_trajectory_point)
		self.arm_cmd.publish(joint_trajectory)
		

	def pick_piece(self, piece, pieces_coordinates):
		#Pick a piece knowing its coordinates on the chess board.
		rospy.loginfo('Pick the piece')
		#Save the piece type.
		piece_type = pieces_coordinates[piece][1]
		#Allow collisions between object and links of the gripper.
		#self.gripper.attach_object(piece, link_name = 'gripper_grasping_frame', touch_links = self.gripper.get_joints()) ############MODIFICATO
		self.scene.remove_attached_object(link = 'gripper_grasping_frame') ############MODIFICATO
		#Close the gripper according to the diameter of the piece.
		movement = 'close_gripper_' + piece_type['name'] ####piece_type['name']
		self.playmotion_movement(movement, planning = False)

	def place_piece(self, end_place):
		#Lower the gripper to place the piece.
		#end_place: chessboard coordinates of the end place.

		if end_place == 'box':
			self.straight_eef_movement(self.arm, 'z', -0.08, avoid_collisions = False) #Lower the gripper			
			self.playmotion_movement('open_gripper', planning = False)
			#self.scene.remove_attached_object(link = 'gripper_grasping_frame') ############MODIFICATO
			self.straight_eef_movement(self.arm, 'z', 0.08, avoid_collisions = False) #Higher the gripper			
		else:
			rospy.loginfo("Opening the gripper")
			self.playmotion_movement('open_gripper', planning = False) #Open the gripper
			#self.scene.remove_attached_object(link = 'gripper_grasping_frame') ############MODIFICATO

	def playmotion_movement(self, movement, planning = True, timeout = 10.0):
		#Play a pre-recorded movement exploiting the playmotion package.
		#movement: [string] the name of a playmotion movement. 
		#planning: [bool] wheter to plan or not the movement toward the first point of the pre-recorded trajectory.
		#timeout: [float] maximum time in [s] for the action execution.

		rospy.loginfo('Preparing to play the \'' + movement + '\' movement.')
		#Exploit a pre-defined PlayMotion
		goal = PlayMotionGoal()
		goal.motion_name = movement
		goal.skip_planning = not planning
		#Execute the movement
		self.playmotion.send_goal(goal)
		success = self.playmotion.wait_for_result(rospy.Duration(timeout))
		if success:
		    rospy.loginfo('The \'' + movement + '\' movement has been executed.')
		else:
		    rospy.logwarn('Timeout (' + str(timeout) + 's) elapsed. The \'' + movement + '\' movement has not been completed.')

	def prepare_TIAGo(self, clock_pose, z_coord_chessboard, box_pose):
		#global saved_arm_joints
		
		#Prepare TIAGo to grasp the pieces: unfold the arm and prepare the grasping motion.
		rospy.loginfo("Unfold the arm and prepare the grasp")
		#self.execute_remembered_joint_values(self.arm, 'open_arm') ###DA COMMENTARE
		#self.execute_remembered_joint_values(self.arm, 'prepare_chess_grasp')
		self.look_up()
		time.sleep(1)
		if clock_pose != 'none':
			self.gripper_over_clock(clock_pose, z_coord_chessboard)
		self.current_escape =  self.desired_escape #Change the current escape strategy.
		'''
		with open(simul_config) as file:
				transformed_centroids = yaml.load(file)

		squares_to_index = cfg.squares_to_index_white
		#Move the arm over the box one time
		self.gripper_over_box(box_pose, z_coord_chessboard)
		
		#Save the arm joints values to reach the same position in different phases of the game (save them also in yaml to use them if the configuration has already been done).
		self.sub = rospy.Subscriber('joint_states', JointState, self.joint_states_callback)
		time.sleep(1)
		saved_arm_joints = self.saved_position[:7]
		print('SAVED ARM JOINTS: ' + str(saved_arm_joints))
		with open(saved_joints_pose_file, "w") as t_p:
			yaml.dump(saved_arm_joints, t_p)
		rospy.loginfo("TIAGo's arm joints configuration over the box saved'")
		#Remember the joint arm
		self.arm.remember_joint_values('arm_over_box', values = [saved_arm_joints[0], saved_arm_joints[1], saved_arm_joints[2], saved_arm_joints[3], saved_arm_joints[4], saved_arm_joints[5], saved_arm_joints[6]])
		print(self.arm.get_remembered_joint_values())
        #Bring back the arm over the clock
		if clock_pose != 'none':
			self.gripper_over_clock(clock_pose, z_coord_chessboard)

		'''
		self.playmotion_movement('open_gripper', planning = False)

		rospy.loginfo("TIAGo is ready to play")
		rospy.sleep(2)

		state_publisher.publish(10) #State of waiting to start the game.
		time.sleep(1)
		
		return self.current_escape

	def joint_states_callback(self, msg):
		self.saved_position = msg.position
		self.sub.unregister()

	def rest_pose(self):
		#Bring TIAGo's arm back to the "ready to play" position.
		rospy.loginfo("Getting back to the resting pose")
		self.execute_remembered_joint_values(self.arm, 'prepare_chess_grasp')

	def straight_eef_movement(self, group, direction, translation, eef_step = 0.005, jump_threshold = 0.0, avoid_collisions = True):
	    #Move the end-effector along a straight line in the specified direction
	    #direction: [string] must be 'x, 'y' or 'z'
	    #translation: [float] distance in [m] of the end-effector translation
	    #avoid_collisions: [bool] if set to True, moves the end-effector even if a collision is detected
	    waypoints = []
	    #Add a waypoint
	    waypoint_pose = group.get_current_pose().pose
	    if direction == 'x':
	        waypoint_pose.position.x += translation 
	    elif direction == 'y':
	        waypoint_pose.position.y += translation
	    elif direction == 'z':
	        waypoint_pose.position.z += translation
	    waypoints.append(copy.deepcopy(waypoint_pose))       
	    (plan, fraction) = group.compute_cartesian_path(waypoints, eef_step, jump_threshold, avoid_collisions = avoid_collisions) # waypoints to follow, interpolation resolution in [m], jump_threshold (0.0 means it is disabled.)
	    #Add velocity control/scaling.      
	    if plan.joint_trajectory.points:
	        group.execute(plan)  #Move the robot
	        group.stop() #Make sure there is no residual movement
	    rospy.loginfo(str(int(fraction * 100)) + '% of the end-effector translation of ' + str(translation) + 'm along the ' + direction + ' direction has been completed.')

	    if fraction < 1.0:
	    	rospy.logwarn('The straight movement has not been totally completed.')
	    	raise StraightMovementError

	    return fraction, translation

	def straight_movements_chessboard(self, group, centroids, end_place, squares_to_index, tilt, clock_pose, box_pose, eef_step = 0.005, jump_threshold = 0.0, avoid_collisions = True):
	    #Move the end-effector along a straight line in the specified direction
	    #centroids: [list] list of centroids of the chessboard.
	    #end_place: [string] square to which the end-effector is going.
	    #avoid_collisions: [bool] if set to False, moves the end-effector even if a collision is detected.

	    #Set the x offset looking at the eventual tilt of the gripper.
		if tilt == 'avanti': #If the gripper is tilted because he reached the last row
			offset = self.offset_last_row
		elif tilt == 'indietro':
			offset = self.offset_last_row
		else:
			offset = 0.0

		#Get the current arm pose
		current_pose = group.get_current_pose().pose
		#Get the coordinate of the end square
		if end_place == 'clock':
			end_position = [clock_pose.position.x, clock_pose.position.y, current_pose.position.z]				
		elif end_place == 'box':
			end_position = [box_pose.position.x - 0.08, box_pose.position.y, current_pose.position.z] #DA VERIFICARE IL -0.06		
		else:
			index = squares_to_index[end_place]
			end_position = [centroids[index].point.x, centroids[index].point.y, current_pose.position.z]

		#Compute the needed translation in the x and y directions
		x_translation = (end_position[0] - current_pose.position.x)
		y_translation = (end_position[1] - current_pose.position.y)
		
		waypoints = []
		waypoint_pose = current_pose
		#Add waypoints
		if abs(x_translation) > 0.01:
			if (abs(abs(x_translation) - abs(y_translation)) < 0.03): #If the movement is diagonal, execute a diagonal movement of the piece.
				waypoint_pose.position.x += x_translation
				waypoint_pose.position.x -= offset
				waypoint_pose.position.y += y_translation
				waypoints.append(copy.deepcopy(waypoint_pose))
			else:
				waypoint_pose.position.x += x_translation
				waypoint_pose.position.x -= offset
				waypoints.append(copy.deepcopy(waypoint_pose))       

		if abs(y_translation) > 0.01 and (abs(abs(x_translation) - abs(y_translation)) >= 0.03):
			waypoint_pose.position.y += y_translation
			waypoints.append(copy.deepcopy(waypoint_pose))

		(plan, fraction) = group.compute_cartesian_path(waypoints, eef_step, jump_threshold, avoid_collisions = avoid_collisions) #Waypoints to follow, interpolation resolution in [m], jump_threshold (0.0 means it is disabled.)

		#Save the plan in a txt file so to take track of the joints planning
		with open('planning.txt', 'w') as planning_file:
				planning_file.write(str(plan))
	    
		if plan.joint_trajectory.points:
			group.execute(plan)  #Move the robot
			group.stop() #Make sure there is no residual movement
		rospy.loginfo(str(int(fraction * 100)) + '% of the end-effector translation over the chessboard.')

		if fraction < 1.0:
			rospy.logwarn('The straight movement over the chessboard has not been totally completed.')
			raise StraightMovementError


def main():
	rospy.init_node('grasp_piece')
	#grasper = Grasping()

	try:
		rospy.spin()
		#grasper.move_arm_joint(0.471)   
	except KeyboardInterrupt:
		print("Shutting down the grasper module.")





if __name__ == '__main__':
     main()