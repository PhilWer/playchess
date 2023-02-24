#!/usr/bin/env python
"""This script implements the ROS node to find the ArUco markers needed to locate the box for the captured pieces and the clock. According to the detection, the MoveIt planning scene is populated with obstacles representing the clock, the box, and a 'virtual wall' to prevent the robot arm to move too close to the opponent.
"""

# Python libraries
import sys
import os
import argparse
import time

from math import pi
import cv2
import yaml

# ROS libraries
import rospy
import moveit_commander
import ros_numpy

# ROS messages
from geometry_msgs.msg import Point, Quaternion, Pose, PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Int16


class SceneSetup:	# probably should be something like "SceneBuilder or SceneSetup"
	def __init__(self, aruco_clock_id, aruco_box_id):
		#Define TIAGo's interface with the surrounding environment
		self.scene = moveit_commander.PlanningSceneInterface() 

	#############
	### ARUCO ###
	#############
	def get_aruco_pose(self, marker_name, timeout = 5.0):
		"""Wait for a message published from a node of type `aruco_single` (see [aruco_ros](http://wiki.ros.org/aruco_ros)) and return the detected pose of the marker. 

		Args:
			marker_name (string): The name of the marker. The topic on which the pose is published is assumed to be `aruco_<marker_name>/pose`.
			timeout (float, optional): The time (in [s]) to wait for the . Defaults to 5.0.

		Raises:
			NotImplementedError: If no marker is found on the selected topic within the timeout. This is meant as a placeholder for the future implementation of backup plans when markers are not available.

		Returns:
			geometry_msgs/PoseStamped: The pose of the detected marker.
		"""
		rospy.loginfo('Looking for {} ArUco marker...'.format(marker_name))
		try:
			topic_name = 'aruco_{}/pose'.format(marker_name)
			pose_msg = rospy.wait_for_message(topic_name, PoseStamped, timeout = timeout)
			rospy.loginfo('ArUco marker for {obj} found:\n{pose}'.format(obj = marker_name,
																		pose = pose_msg)
						)
			return pose_msg
		except rospy.exceptions.ROSException: 
			# Handle the exception thrown by wait_for_message if timeout occurs
			rospy.logwarn('Timeout exceeded the time limit of {t:.0f}s.'.format(t = timeout))
			raise NotImplementedError('ArUCo marker for {obj} not found. This error cannot be handled, kill and restart the application.')

	######################
	### PLANNING SCENE ###
	######################
	def add_box(self, name, pos, size, savedir = None):
		"""Add a virtual obstacle of given pose and size to the MoveIt planning scene.

		Args:
			name (string): Name of the obstacle.
			pos (geometry_msgs/Point): The center of the box representing the obstacle. The orientation is fixed, parallel to the axis of the `base_footprint` reference frame.
			size (tuple(float, float, float)): The size of the box along the XYZ directions.
			savedir (string, optional): If specified, the pose of the object will be save into a `.yaml` file in the specified folder. Defaults to None.
		"""
		# Add collision box corresponding to the `name` object.
		pose = PoseStamped()
		pose.header.frame_id = 'base_footprint'
		pose.header.stamp = rospy.Time.now()
		pose.pose = Pose(Point(pos.x, pos.y, pos.z), 
		   				Quaternion(0, 0, 0, 1)	# ASSUMPTION. The object tilt wrt to base RF is negligible.
						)
		
		self.scene.add_box(name, pose, size = size)
		if savedir is not None:
			savepath = os.path.join(savedir, '{}_pose.yaml'.format(name))
			with open(savepath, "w") as f:
				yaml.dump(pose, f) # Save the pose of the object in the yaml file

	#####################
	### VISUALIZATION ###
	#####################
	def get_result_img(self, marker_name, savedir):
		"""Get an image in which the detected markers are highlighted and save it to a file.

		Args:
			marker_name (string): The name of the marker. The topic on which the pose is published is assumed to be `aruco_<marker_name>/result`. Assumed that such topic exists, the image will contain all the detected markers, not just `<marker_name>`.
			savedir (string): The image will be save into a `.png` file in the specified folder.
		"""
		# Get an image of the detected markers
		topic_name = 'aruco_{}/result'.format(marker_name)
		msg = rospy.wait_for_message(topic_name, Image)
		# Convert the ROS Image message into a numpy ndarray.
		img = ros_numpy.numpify(msg)
		# Save the image to the proper folder to open it in the GUI.
		cv2.imwrite(os.path.join(savedir, 'markers_localization.png') , 
	      			cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
					) 
		rospy.loginfo('ARUCO markers localization result image saved in the',
					  '{} file.'.format(os.path.join(savedir, 'markers_localization.png'))
					  )
		

if __name__ == '__main__':
	# Initialize the node
	rospy.init_node('aruco_finder', anonymous = True)
	# Create an instance of the setup manager
	sc_setup = SceneSetup()
	
	try:	
		# Get the package root dir to make the filepaths relative
		PLAYCHESS_PKG_DIR = rospy.get_param('/playchess/root')
		TMP_DIR 		  = os.path.join(PLAYCHESS_PKG_DIR, 'tmp')
		GUI_IMG_DIR       = os.path.join(PLAYCHESS_PKG_DIR, 'scripts', 'gui', 'image')

		# Define a Publisher to change the state at the end of the process
		state_publisher = rospy.Publisher('/state', Int16, queue_size = 10)
			
		# Populate the planning scene:
		# 1. Add clock
		marker = rospy.get_param('~clock_marker_name')
		pose = sc_setup.get_aruco_pose(marker)
		size = rospy.get_param('~clock_size')
		sc_setup.add_box('clock', pose.position, size, savedir = TMP_DIR)
		# 2. Add box
		marker = rospy.get_param('~box_marker_name')
		pose = sc_setup.get_aruco_pose(marker)
		size = rospy.get_param('~box_size')
		sc_setup.add_box('box', pose.position, size, savedir = TMP_DIR)
		# 3. Add virtual wall
		pos = rospy.get_param('~wall_position')
		size = rospy.get_param('~wall_size')
		sc_setup.add_box('wall', pos, size)

		rospy.sleep(1) 	# Wait for the planning scene to be updated.
						# TODO. Implement an actual check instead of simply waiting.

		# Get an image of the result and store it to update the GUI accordingly
		sc_setup.get_result_img(marker_name = 'clock',	# all the detected markers are highlighted
			  											# regardless of their ID
			  					savedir = GUI_IMG_DIR)

		# Publish a state message to be read by the GUI. Once received, the GUI will enable the button to confirm the markers localization.
		state_publisher.publish(40)
		
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down the ARUCO markers finder module.")			














