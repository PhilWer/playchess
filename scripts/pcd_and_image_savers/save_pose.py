#!/usr/bin/env python

# ROS libs
import time
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, PoseWithCovarianceStamped

class SavePose:
	def __init__(self):
		#Save the pose of TIAGo
		self._pose = Pose()
		self._pose_sub = rospy.Subscriber('/robot_pose', PoseWithCovarianceStamped, self.sub_callback)

	def sub_callback(self, msg):
		self._pose = msg.pose.pose

		#rospy.loginfo(self._pose)
