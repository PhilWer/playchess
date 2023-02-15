#!/usr/bin/env python3

# Rospy for the subscriber
import numpy as np
import math
import sys
# ROS Image messages
# from sensor_msgs.msg import CompressedImage, Image
# ROS Image message --> Open CV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# Open CV2 for saving an image
import cv2
import os
# import imutils
import open3d as o3d
import pickle
import matplotlib.path as mpath
import yaml
# from skimage.metrics import compare_mse
'''change it for p2.7 or older version of skimages in useful_functions.chess_move_detection_ssim  '''


from useful_functions.chess_move_detection_ssim import MoveDetection
from useful_functions.depth_to_point_cloud_count import DepthImageProcessing
#from useful_functions.homographic_transformation import HOMO_TRANSFOR as ht
from useful_functions import config as cfg

# '/root/tiago_public_ws/src/tiago_playchess/scripts/config/simulation_config.yaml'
#imported_configurations = r'/home/vaishakh/tiago_public_ws/src/playchess/scripts/config/simulation_config.yaml'
imported_configurations = (os.path.join(os.path.realpath(os.path.dirname(__file__)),'..','..','config', 'simulation_config.yaml'))


class OccupancyChecke:
    def __init__(self):
        self.debug = True
        # Load configurations
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
            self.castle_squares = ['a8', 'e8', 'h8',
                                   'king_e8', 'rook_a8', 'rook_h8']
            self.castle_ending_squares = ['d8', 'c8', 'g8', 'f8']
            self.seventh_row = 'row7'
            self.second_row = 'row2'
        else:
            self.squares_to_index = cfg.squares_to_index_black
            self.rows_indexes = cfg.rows_black
            self.opposite_color = 'white'
            self.castle_squares = ['a1', 'e1', 'h1',
                                   'king_e1', 'rook_a1', 'rook_h1']
            self.castle_ending_squares = ['d1', 'c1', 'g1', 'f1']
            self.seventh_row = 'row2'
            self.second_row = 'row7'
    
    
    def opponent_move(self, state1_image, state2_image,depth_image_state2,debug=True):
        tiago_color=self.color
        with open(os.path.join(os.path.realpath(os.path.dirname(__file__)),'yaml','store_chess_board_edges.yaml'), 'r') as file:
         chessboardedges_with_out_borders = yaml.load(file,Loader=yaml.Loader)
        self.potential_altered_squares = MoveDetection().possible_squares_start_end(chessboardedges_with_out_borders, state1_image, state2_image, tiago_color,debug)
        self.possible_start_square=[]
        self.possible_end_square=[]
        self.possible_capture=[]
        self.possible_castle=[]
        self.possible_enpasent=[]
        for i in self.potential_altered_squares:
            current_square = i['name'] 
            if self.live_chessboard_situation_complete[current_square][1]==self.opposite_color:
                self.possible_start_square.append(current_square)
 
            elif self.live_chessboard_situation_complete[current_square][1]=='none':
                self.possible_end_square.append(current_square)
            elif self.live_chessboard_situation_complete[current_square][1]==self.color:
                self.possible_capture.append(current_square)
        if self.debug:
            print('befor pcl processing:start',self.possible_start_square)
            print('end',self.possible_end_square)
            print('capture',self.possible_capture)


        with open(os.path.join(os.path.realpath(os.path.dirname(__file__)),'yaml','corners_names_empty_square.yaml', ),'r') as file:
            chessboard_square_corners = yaml.load(file,Loader=yaml.Loader)
        dip=DepthImageProcessing()
        for i in self.possible_start_square:
            square_name=i
            for j in chessboard_square_corners:
                if j[5] == square_name:
                    filter_points =np.array([j[0],j[1],j[2],j[3]])
                    pcl_count,mean_rgb_value,occupancy=dip.depth_pcl_counting(depth_image_state2,state2_image,filter_points,quat=[-0.672, 0.680, -0.202, 0.214]
,trans=[0.232, 0.015, 1.315])
                    print('pcl-count',pcl_count)
                    if occupancy==True:
                        self.possible_start_square.remove(i)
        for i in self.possible_end_square:
            square_name=i
            for j in chessboard_square_corners:
                if j[5] == square_name:
                    filter_points =np.array([j[0],j[1],j[2],j[3]])
                    pcl_count,mean_rgb_value,occupancy=dip.depth_pcl_counting(depth_image_state2,state2_image,filter_points,quat=[-0.672, 0.680, -0.202, 0.214]
,trans=[0.232, 0.015, 1.315])
                    if occupancy!=True:
                        self.possible_end_square.remove(i)
                    print(pcl_count)
                    

        for i in self.possible_capture:
            square_name=i
            for j in chessboard_square_corners:
                if j[5] == square_name:
                    filter_points =np.array([j[0],j[1],j[2],j[3]])
                    pcl_count,mean_rgb_value,occupancy=dip.depth_pcl_counting(depth_image_state2,state2_image,filter_points,quat=[-0.672, 0.680, -0.202, 0.214]
,trans=[0.232, 0.015, 1.315])
                    if occupancy==True :
                        if self.opposite_color =='black' and mean_rgb_value[0] >=0.5:
                            self.possible_capture.remove(i)
                        elif self.opposite_color =='white' and mean_rgb_value[0]<0.5:
                            self.possible_capture.remove(i)
                    print(mean_rgb_value[0])

                    

                

        return self.possible_capture,self.possible_start_square,self.possible_end_square
    

    
    
    


if __name__ == '__main__':
    oc = OccupancyChecke()

    depth_image_state2=np.genfromtxt(os.path.join(os.path.realpath(os.path.dirname(__file__)),'move_detection/csv/move_7','empty_depth_image.csv'), delimiter=',')
    img1 = cv2.imread(os.path.join(os.path.realpath(os.path.dirname(__file__)),'move_detection/csv/move_6','empty_rgb_image.png'))
    img2 = cv2.imread(os.path.join(os.path.realpath(os.path.dirname(__file__)), 'move_detection/csv/move_7','empty_rgb_image.png'))
    tiago_colour = 'white'
    debug = True
    potential_capture,potential_start,potential_end = oc.opponent_move( img1, img2,depth_image_state2)
    print('after pclpotential_start_squares:',potential_start)
    print('potential_end_square:',potential_end)
    print('potential_capture:',potential_capture)

    
       # print(chessboard_square_corners[60])