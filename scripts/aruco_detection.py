#!/usr/bin/env python
#Class to detect aruco markers and locate table and chessboard

#ROS libraries
import rospy, rosnode
import moveit_commander

#ROS messages
from moveit_msgs.srv import GetPlanningScene, ApplyPlanningScene
from moveit_msgs.msg import PlanningSceneComponents, PlanningScene, AllowedCollisionEntry, AllowedCollisionMatrix, Constraints, PositionConstraint, BoundingVolume, VisibilityConstraint, JointConstraint, OrientationConstraint
from geometry_msgs.msg import Point, Quaternion, Pose, PoseStamped, Vector3
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int16, String

#Python libraries
import copy
from math import pi
import cv2
import os
import time
import yaml

# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError

#My scripts
import config as cfg

PLAYCHESS_PKG_DIR = '/home/luca/tiago_public_ws/src/tiago_playchess'
GUI_PKG_DIR       = '/home/luca/tiago_public_ws/src/chess_gui'

#Instantiate CvBridge
bridge = CvBridge()

#Publishers initialization
state_publisher = rospy.Publisher('/state', Int16, queue_size = 10)


#Callback to save the image
def SaveImage(data):
	#Save the image of the ARUCO markers that have been identified.
	try:
		# Convert your ROS Image message to OpenCV2
		cv2_img = bridge.imgmsg_to_cv2(data, "bgr8")
	except CvBridgeError, e:
		print(e)
	else:
		cv2.imwrite(os.path.join(GUI_PKG_DIR + '/images', 'markers_localization.png'), cv2_img) #save the OpenCv2 image as png

	rospy.loginfo('ARUCO markers localization result image saved in the chess_gui/images folder')

	state_publisher.publish(40) #State of enabling the GUI pushbutton to confirm the markers localization.


class Aruco:
	def __init__(self):
		#ARUCO numbers associated with each object
		self.clock = cfg.aruco_clock #100
		self.box = cfg.aruco_box #300

		#Define TIAGo's interface
		self.scene = moveit_commander.PlanningSceneInterface() #The interface with the world surrounding the robot

		self.clock_pose_file = rospy.get_param('/tiago_playchess/clock_pose_file')
		self.box_pose_file = rospy.get_param('/tiago_playchess/box_pose_file')


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
	rospy.Subscriber("/aruco_single100/result", Image, SaveImage)


	try:
		rospy.spin()    
	except KeyboardInterrupt:
		print("Shutting down the ARUCO markers finder module.")			














