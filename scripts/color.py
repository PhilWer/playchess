#!/usr/bin/env python

#Script to ask the user which color TIAGo is playing with and save the parameter in standard_config.yaml

# ROS libs
import rospy
import yaml
import sys

PLAYCHESS_PKG_DIR = '/home/luca/tiago_public_ws/src/tiago_playchess'
#Defines
simulation_congif_root = PLAYCHESS_PKG_DIR + '/scripts/config/simulation_config.yaml' #'/root/tiago_public_ws/src/tiago_playchess/scripts/config/simulation_config.yaml'

class Color:
	def __init__(self):
		pass

	def query_color(self, question, default = "white"):
	    #Ask a question via raw_input() and return their answer.
	    #question: string that is presented to the user.
	    #default: presumed answer if the user just hits <Enter>. It must be "white" (the default), "black" or None (meaning an answer is required of the user).
	    
	    valid = {"white": 'white', "w": 'white', "W": 'white', "black": 'black', "b": 'black', "B": 'black'}
	    if default is None:
	        prompt = " [white/black] "
	    elif default == "white":
	        prompt = " [White/black] "
	    elif default == "black":
	        prompt = " [white/Black] "
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
	            sys.stdout.write("Please respond with 'white' or 'black' " "(or 'w' or 'b').\n")

	def which_color(self):
		#TODO: ASK THE USER THE COLOR OF TIAGO AND SAVE IT IN A STRING VARIABLE
		color = self.query_color('Which color is TIAGo playing with?')
		#color = 'white' #PER ORA USO QUESTO PER LE SIMULAZIONI, POI TROVO IL MODO DI INTERFACCIARMI CON L'USER
		#Change the yaml parameter 'color' to the color of TIAGo (chosen by the user) to import the corret configuration parameters.
		#file = rospy.get_param('/tiago_playchess/config') #Opening of the yaml file with a launch parameter
		with open(simulation_config_root) as t_p: 
    			params = yaml.load(t_p)
		params['color'] = color #color e' il risultato dell'operazione prima che chiedera' all'user di inserire il colore di tiago.
		with open(simulation_config_root, "w") as t_p:
    			yaml.dump(params, t_p)
		return color


if __name__ == '__main__':
	try:
		rospy.init_node('ask_color')

		colors = Color() #Save TIAGo's color
		color = colors.which_color()
		print('TIAGo is playing with: ' + color)


        #print(live_situation)

	except KeyboardInterrupt:
		rospy.logwarn('Shutting down the grasping node')
