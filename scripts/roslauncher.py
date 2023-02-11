#!/usr/bin/env python
# simple functions to programatically launch nodes, eploiting the roslaunch API
# http://wiki.ros.org/roslaunch/API%20Usage

# ROS libs
import rospy, roslaunch

def roslaunch_from_file(path):
	# programatically launch the nodes for vision
	uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
	roslaunch.configure_logging(uuid)
	# launch the nodes
	parent = roslaunch.parent.ROSLaunchParent(uuid, [path])
	parent.start()
	return parent #returned in order to be available to shutdown the node
