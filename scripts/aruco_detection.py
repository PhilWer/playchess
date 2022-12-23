#!/usr/bin/env python
"""This script implements the ROS node to find the ArUco markers needed to locate the box for the captured pieces and the clock.
"""

# Python libraries
import copy
from math import pi
import cv2
import os
import time
import yaml

# ROS libraries
import rospy
import rospkg
import moveit_commander

# ROS packages
import ros_numpy

# ROS messages
from geometry_msgs.msg import Point, Quaternion, Pose, PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Int16

# My scripts
import config as cfg

''' TODO
Merge a global publisher, a function and a class into single class. A reasonable name could be ArucoSetup or SceneSetup (as it also adds the "walls" to the scene, a feature that should probably moved to a different place, e.g. where the pieces are added).

Behavior could be: setup two subscribers (for clock and box marker). Upon request, activate the callbacks, receive detections and average them (for a fixed time, i.e. few seconds). Check if both have been found at least once, otherwise warn the user/ask for intervention/re-run.

In a different script, there should be a SetupManager (opposed to a GameManager) that handles the node start and stop to avoid having unuseful nodes running continuously.

Isolate the SceneBuilder, just use the `new` class to pass the clock and box Pose to the SceneBuilder
'''

# Publishers initialization
state_publisher = rospy.Publisher('/state', Int16, queue_size = 10)

def save_image(data):
	# Convert the ROS Image message into a numpy ndarray.
	img = ros_numpy.msgify(msg)
	# Save the image to the proper folder to open it in the GUI.
	gui_img_dir = os.path.join(rospkg.get_path('playchess'), 'gui', 'images')
	cv2.imwrite(os.path.join(gui_img_dir, 'markers_localization.png'), img) 
	rospy.loginfo('ARUCO markers localization result image saved in the ' + gui_img_dir + ' folder.')
	# Publish a state message to be read by the GUI. Once received, the GUI will enable the button to confirm the markers localization.
	state_publisher.publish(40)


class Aruco:	# probably should be something like "SceneBuilder or SceneSetup"
	def __init__(self):
		# ArUco marker ID associated with each object
		self.clock = cfg.aruco_clock 	# 100
		self.box = cfg.aruco_box 		# 300

		#Define TIAGo's interface
		self.scene = moveit_commander.PlanningSceneInterface() #The interface with the world surrounding the robot

		self.clock_pose_file = rospy.get_param('/playchess/clock_pose_file')
		self.box_pose_file = rospy.get_param('/playchess/box_pose_file')
		# TODO: use a namespace instead of hardcoding the package name


	def aruco_detection(self, target, time_limit = 3):
		#Look for ArUco markers, if at least one marker with the correct ID (set at robot startup), return its pose or check the displacement and misalignment if an initial pose is set in input.
		#target: [string] what is TIAGo trying to identify (chessboard --> 100, clock --> 200 or box --> 300)
		#time_limit: [float] the maximum time (in [s]) to wait when looking for a marker.

		if target == 'clock':
			name = 'aruco_single{}'.format(self.clock)
		elif target == 'box':
			name = 'aruco_single{}'.format(self.box)

		rospy.loginfo('Looking for ArUco marker...')
		averaged_samples = 1
		elapsed_time = 0
		while elapsed_time < time_limit: #Set also a minimum (and maximum) number of averaged samples?
			t = rospy.get_time()
			try:
				pose_msg = rospy.wait_for_message('/{}/pose'.format(name), PoseStamped, timeout = time_limit)
				got_aruco = True
			except rospy.exceptions.ROSException: #Handle the exception thrown by wait_for_message if timeout occurs
				rospy.logwarn('Timeout exceeded the time limit of {time_limit:.0f}s.'.format(time_limit = time_limit))
				got_aruco = False

			if got_aruco: # and pose_msg.header.frame_id == '/base_footprint'
				if averaged_samples == 1:
					pose_f = pose_msg.pose
				elif averaged_samples > 1:
					pose_f.position = pts.average_point(new_point = pose_msg.pose.position, num_samples = averaged_samples, avg_point = pose_f.position)
					pose_f.orientation = quat.average_Quaternions(new_q = pose_msg.pose.orientation, num_samples = averaged_samples, avg_q = pose_f.orientation)
				averaged_samples += 1
				rospy.loginfo('ArUco marker for {} found:'.format(target))
				rospy.loginfo(pose_f)
				return pose_f
			else:
				rospy.loginfo('No marker corresponding to {} found, try again'.format(target))

			elapsed_time += rospy.get_time() - t

	'''TODO
	def add_box(self, name, pose, size, savefile = None)
	'''

	def populate_clock(self, clock_pose):
		#Add collision box corresponding to the clock.
		pose = PoseStamped()
		pose.header.frame_id = 'base_footprint'
		pose.header.stamp = rospy.Time.now()
		pose.pose = Pose(Point(clock_pose.position.x, clock_pose.position.y, clock_pose.position.z), Quaternion(0, 0, 0, 1))
		self.scene.add_box('clock', pose, size = (0.20, 0.11, 0.06))
		with open(self.clock_pose_file, "w") as t_coord:
			yaml.dump(clock_pose, t_coord) #Save the pose of the clock in the yaml file

	def populate_box(self, box_pose):
		#Add collision box corresponding to the pieces box.
		pose = PoseStamped()
		pose.header.frame_id = 'base_footprint'
		pose.header.stamp = rospy.Time.now()
		pose.pose = Pose(Point(box_pose.position.x, box_pose.position.y, box_pose.position.z), Quaternion(0, 0, 0, 1))
		self.scene.add_box('box', pose, size = (0.22, 0.15, 0.09)) #era (0.22, 0.15, 0.06)
		with open(self.box_pose_file, "w") as t_coord:
			yaml.dump(box_pose, t_coord) #Save the pose of the box in the yaml file

	def populate_for_safety(self, clock_pose, box_pose):
		#Add collision boxes to limit TIAGo's movements.
		'''
		#Box to the left of the box.
		pose = PoseStamped()
		pose.header.frame_id = 'base_footprint'
		pose.header.stamp = rospy.Time.now()
		pose.pose = Pose(Point(box_pose.position.x, box_pose.position.y + 0.30, box_pose.position.z), Quaternion(0, 0, 0, 1))
		self.scene.add_box('wall_1', pose, size = (1.2, 0.02, 1.2))

		#Box to the right of the clock.
		pose = PoseStamped()
		pose.header.frame_id = 'base_footprint'
		pose.header.stamp = rospy.Time.now()
		pose.pose = Pose(Point(clock_pose.position.x, clock_pose.position.y - 0.30, clock_pose.position.z), Quaternion(0, 0, 0, 1))
		self.scene.add_box('wall_2', pose, size = (1.2, 0.02, 1.2))
		'''
		self.scene.remove_world_object('wall_1')
		self.scene.remove_world_object('wall_2')
		#Box after the end of the chessboard.
		pose = PoseStamped()
		pose.header.frame_id = 'base_footprint'
		pose.header.stamp = rospy.Time.now()
		pose.pose = Pose(Point(box_pose.position.x + 0.40, box_pose.position.y - 0.25, box_pose.position.z), Quaternion(0, 0, 0, 1))
		self.scene.add_box('wall_3', pose, size = (0.02, 1.2, 1.2))
		

if __name__ == '__main__':
	rospy.init_node('aruco_finder', anonymous = True)
	aruco_identifier = Aruco()

	clock_marker_pose = aruco_identifier.aruco_detection(target = 'clock', time_limit = 3) #Look for the marker identifying the clock and save its pose in a variable.
	box_marker_pose = aruco_identifier.aruco_detection(target = 'box', time_limit = 3) #Look for the marker identifying the box and save its pose in a variable.
	
	#Populate the planning scene with the jus located clock and box for collision management.
	if clock_marker_pose:
		aruco_identifier.populate_clock(clock_marker_pose)
	if box_marker_pose:
		aruco_identifier.populate_box(box_marker_pose)

	if clock_marker_pose and box_marker_pose:
		aruco_identifier.populate_for_safety(clock_marker_pose, box_marker_pose) #Populate the planning scene with boxes to avoid big movements of TIAGo's arm.

	time.sleep(1)

	#Initalize a Subscriber to get the image of the found markers.
	rospy.Subscriber("/aruco_single100/result", Image, save_image)


	try:
		rospy.spin()    
	except KeyboardInterrupt:
		print("Shutting down the ARUCO markers finder module.")			














