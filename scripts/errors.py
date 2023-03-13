#!/usr/bin/env python

class ArucoException(Exception):
	''' raised when markers are required, but it is not possible to spot any of them '''
	pass 

class PlanningException(Exception):
	''' raised if MoveIt cannot find a plan to bring the eef to the desired position '''
	pass

class StraightMovementError(Exception):
	''' raised if the straight movement has not been completed '''
	pass
