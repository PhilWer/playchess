#!/usr/bin/env python3

# Rospy for the subscriber
import sys
import numpy as np
import math
import sys
# ROS Image messages
# from sensor_msgs.msg import CompressedImage, Image
# ROS Image message --> Open CV2 image converter
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
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

from useful_functions.chess_move_detection_ssim import MoveDetection
from useful_functions.depth_to_point_cloud_count import DepthImageProcessing
# from useful_functions.homographic_transformation import HOMO_TRANSFOR as ht
from useful_functions import config as cfg
if __name__ =='__main__':
    move=16
PLAYCHESS_PKG_DIR = os.path.normpath(os.path.join(os.path.realpath(os.path.dirname(__file__)), '..', '..'))

class OccupancyChecker:
    def __init__(self):
        self.debug =False
        
        # Load configurations
        self.root = os.path.normpath(os.path.join(os.path.realpath(os.path.dirname(__file__)), '..', '..'))
        imported_configurations = os.path.join(self.root, 'scripts', 'config', 
                                               'simulation_config.yaml')
        # Uncomment for fast debugging
        # imported_configurations = r'/home/vaishakh/tiago_public_ws/src/playchess/scripts/config/simulation_config.yaml'
        with open(imported_configurations) as file:
            self.config = yaml.load(file, Loader=yaml.Loader)
        self.color = self.config.get('color')
        self.dir_live_chessboard_situation = PLAYCHESS_PKG_DIR + \
            "/scripts/live_chessboard_situation.yaml"
        with open(self.dir_live_chessboard_situation, 'rb') as live_file:
            self.live_chessboard_situation_complete= yaml.load(
                live_file.read(), Loader=yaml.Loader)
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

    def opponent_move(self, state1_image, state2_image, depth_image_state2,move, debug=0):
        tiago_color = self.color

        tmp_yaml_dir = os.path.join(self.root, 'config', 'tmp', 'yaml')
        with open(os.path.join(tmp_yaml_dir, 'store_chess_board_edges.yaml'), 'r') as file:
            chessboardedges_with_out_borders = yaml.load(
                file, Loader=yaml.Loader)

        dip = DepthImageProcessing()
        self.potential_altered_squares = MoveDetection().possible_squares_start_end(
            chessboardedges_with_out_borders, state1_image, state2_image, tiago_color, debug)
        temp_list = []
        # for only getting square names with changes
        for i in self.potential_altered_squares:
            temp_list.append(i['name'])

        self.potential_altered_squares = temp_list.copy()

        self.possible_start_square = []
        self.possible_end_square = []
        self.possible_capture = []
        self.possible_castle = []
        self.possible_enpasent = []
        self.possible_promotion = []
        castle_done = False
        enpasent = False
        promotion = False

        if self.debug:
            print('potential_altered_square',self.potential_altered_squares)

        # TODO. Get quat and trans as input (use tf in the node calling this script)
        with open(os.path.join(tmp_yaml_dir, 'corners_names_empty_square.yaml'), 'r') as file:
            chessboard_square_corners = yaml.load(file, Loader=yaml.Loader)

        # load for debuging 
        if __name__ == '__main__':
            with open(os.path.join(PLAYCHESS_PKG_DIR,'data','moves','move_{}'.format(move),'live_chessboard_situation.yaml'), 'rb') as live_file:
                self.live_chessboard_situation_complete= yaml.load(live_file.read(), Loader=yaml.Loader)
            

        
        if self.opposite_color == 'white' and all(cell in self.potential_altered_squares for cell in ['e1', 'f1', 'g1', 'h1']):
            for j in chessboard_square_corners:
                if j[5] == 'e1':
                    filter_points = np.array([j[0], j[1], j[2], j[3]])
                    pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                        depth_image_state2, state2_image, filter_points,move, quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315])
                    #print('pcl-count', pcl_count)
                    
                    if occupancy != True:
                        for k in chessboard_square_corners:
                            if k[5] == 'h1':
                                
                                filter_points = np.array(
                                    [k[0], k[1], k[2], k[3]])
                                pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                                    depth_image_state2, state2_image, filter_points,move,quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315])
                                #print('pcl-count', pcl_count)
                                if occupancy != True:
                                    for l in chessboard_square_corners:
                                        if l[5] == 'f1':
                                            filter_points = np.array(
                                                [l[0], l[1], l[2], l[3]])
                                            pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                                                depth_image_state2, state2_image, filter_points,move, quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315])
                                            #print('pcl-count', pcl_count)
                                            if occupancy == True and mean_rgb_value > 0.5:
                                                for m in chessboard_square_corners:
                                                    if m[5] == 'g1':
                                                        filter_points = np.array(
                                                            [m[0], m[1], m[2], m[3]])
                                                        pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                                                            depth_image_state2, state2_image, filter_points,move, quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315])
                                                        print(
                                                            'pcl-count', pcl_count)
                                                        if occupancy == True and mean_rgb_value > 0.5:
                                                            self.possible_castle.append(
                                                                'e1')
                                                            self.possible_castle.append(
                                                                'h1')
                                                            self.possible_castle.append(
                                                                'f1')
                                                            self.possible_castle.append(
                                                                'g1')
                                                            castle_done = True

        elif self.opposite_color == 'white' and  all(cell in self.potential_altered_squares for cell in ['a1', 'c1', 'd1', 'e1']): 
            for j in chessboard_square_corners:
                if j[5] == 'e1':
                    filter_points = np.array([j[0], j[1], j[2], j[3]])
                    pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                        depth_image_state2, state2_image, filter_points,move, quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315])
                    #print('pcl-count', pcl_count)
                    if occupancy != True:
                        for k in chessboard_square_corners:
                            if k[5] == 'a1':
                                filter_points = np.array(
                                    [k[0], k[1], k[2], k[3]])
                                pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                                    depth_image_state2, state2_image, filter_points,move,quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315])
                                #print('pcl-count', pcl_count)
                                if occupancy != True:
                                    for l in chessboard_square_corners:
                                        if l[5] == 'c1':
                                            filter_points = np.array(
                                                [l[0], l[1], l[2], l[3]])
                                            pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                                                depth_image_state2, state2_image, filter_points,move, quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315])
                                            #print('pcl-count', pcl_count)
                                            if occupancy == True:
                                                for m in chessboard_square_corners:
                                                    if m[5] == 'd1':
                                                        filter_points = np.array(
                                                            [m[0], m[1], m[2], m[3]])
                                                        pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                                                            depth_image_state2, state2_image, filter_points,move, quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315])
                                                        print(
                                                            'pcl-count', pcl_count)
                                                        if occupancy == True:
                                                            self.possible_castle.append(
                                                                'e1')
                                                            self.possible_castle.append(
                                                                'a1')
                                                            self.possible_castle.append(
                                                                'c1')
                                                            self.possible_castle.append(
                                                                'd1')
                                                            castle_done = True

        elif self.opposite_color == 'black' and all(cell in self.potential_altered_squares for cell in ['e8', 'f8', 'g8', 'h8']):
            
            for j in chessboard_square_corners:
                if j[5] == 'e8':
                    filter_points = np.array([j[0], j[1], j[2], j[3]])
                    pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                        depth_image_state2, state2_image, filter_points,move, quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315])
                    # print('occupancy',occupancy)
                    if occupancy != True:
                        for k in chessboard_square_corners:
                            if k[5] == 'h8':
                                filter_points = np.array(
                                    [k[0], k[1], k[2], k[3]])
                                pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                                    depth_image_state2, state2_image, filter_points,move,quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315])

                                if occupancy != True:
                                    for l in chessboard_square_corners:
                                        if l[5] == 'f8':
                                            filter_points = np.array(
                                                [l[0], l[1], l[2], l[3]])
                                            pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                                                depth_image_state2, state2_image, filter_points,move, quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315])
                                            if occupancy == True:
                                                for m in chessboard_square_corners:
                                                    if m[5] == 'g8':
                                                        filter_points = np.array(
                                                            [m[0], m[1], m[2], m[3]])
                                                        pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                                                            depth_image_state2, state2_image, filter_points,move, quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315])
                                                        if occupancy == True:
                                                            self.possible_castle.append(
                                                                'e8')
                                                            self.possible_castle.append(
                                                                'h8')
                                                            self.possible_castle.append(
                                                                'f8')
                                                            self.possible_castle.append(
                                                                'g8')
                                                            castle_done = True

        elif self.opposite_color == 'black' and  all(cell in self.potential_altered_squares for cell in ['a8', 'c8', 'd8', 'e8']):
            
            for j in chessboard_square_corners:
                if j[5] == 'e8':
                    filter_points = np.array([j[0], j[1], j[2], j[3]])
                    pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                        depth_image_state2, state2_image, filter_points,move, quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315])
                    #print('pcl-count', pcl_count)
                    if occupancy != True:
                        for k in chessboard_square_corners:
                            if k[5] == 'a8':
                                filter_points = np.array(
                                    [k[0], k[1], k[2], k[3]])
                                pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                                    depth_image_state2, state2_image, filter_points,move,quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315])
                                #print('pcl-count', pcl_count)
                                if occupancy != True:
                                    for l in chessboard_square_corners:
                                        if l[5] == 'c8':
                                            filter_points = np.array(
                                                [l[0], l[1], l[2], l[3]])
                                            pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                                                depth_image_state2, state2_image, filter_points,move, quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315])
                                            #print('pcl-count', pcl_count)
                                            if occupancy == True:
                                                for m in chessboard_square_corners:
                                                    if m[5] == 'd8':
                                                        filter_points = np.array(
                                                            [m[0], m[1], m[2], m[3]])
                                                        pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                                                            depth_image_state2, state2_image, filter_points,move, quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315])
                                                        print(
                                                            'pcl-count', pcl_count)
                                                        if occupancy == True:
                                                            self.possible_castle.append(
                                                                'e8')
                                                            self.possible_castle.append(
                                                                'a8')
                                                            self.possible_castle.append(
                                                                'c8')
                                                            self.possible_castle.append(
                                                                'd8')
                                                            castle_done = True

        elif (self.opposite_color == 'white' and castle_done != True and any(cell[1] in ['5'] for cell in self.potential_altered_squares)
              and any(cell[1] in ['6'] for cell in self.potential_altered_squares)):  # checking for empassment
            for j in self.potential_altered_squares:
                if j[1] == '5' and self.live_chessboard_situation_complete[j][0][0:4] == 'pawn' and self.live_chessboard_situation_complete[j][1] == self.opposite_color:
                    for m in chessboard_square_corners:
                        if m[5] == j:
                            filter_points = np.array(
                                [m[0], m[1], m[2], m[3]])
                            pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                                depth_image_state2, state2_image, filter_points,move, quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315])
                            #print('pcl-count', pcl_count)
                            if occupancy != True:
                                for k in self.potential_altered_squares:
                                    if k[1] == '5' and self.live_chessboard_situation_complete[k][0][0:4] == 'pawn' and self.live_chessboard_situation_complete[k][1] == self.color:
                                        for m in chessboard_square_corners:
                                            if m[5] == k:
                                                filter_points = np.array(
                                                    [m[0], m[1], m[2], m[3]])
                                                pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                                                    depth_image_state2, state2_image, filter_points,move, quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315])
                                                #print('pcl-count', pcl_count)
                                                if occupancy != True:
                                                    for l in self.potential_altered_squares:
                                                        if l[1] == '6' and l != j and self.live_chessboard_situation_complete[l][1] == 'none':
                                                            for m in chessboard_square_corners:
                                                                if m[5] == l:
                                                                    filter_points = np.array(
                                                                        [m[0], m[1], m[2], m[3]])
                                                                    pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                                                                        depth_image_state2, state2_image, filter_points,move, quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315])
                                                                    print(
                                                                        'pcl-count', pcl_count)
                                                                    if occupancy == True:
                                                                        self.possible_enpasent.append(
                                                                            j)
                                                                        self.possible_enpasent.append(
                                                                            k)
                                                                        self.possible_enpasent.append(
                                                                            l)
                                                                        enpasent = True

        elif (self.opposite_color == 'black' and castle_done != True and any(cell[1] in ['3'] for cell in self.potential_altered_squares)
              and any(cell[1] in ['4'] for cell in self.potential_altered_squares)):
            for j in self.potential_altered_squares:
                if j[1] == '4' and self.live_chessboard_situation_complete[j][0][0:4] == 'pawn' and self.live_chessboard_situation_complete[j][1] == self.opposite_color:
                    for m in chessboard_square_corners:
                        if m[5] == j:
                            filter_points = np.array(
                                [m[0], m[1], m[2], m[3]])
                            pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                                depth_image_state2, state2_image, filter_points,move, quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315])

                            if occupancy != True:
                                for k in self.potential_altered_squares:
                                    if k[1] == '4' and self.live_chessboard_situation_complete[k][0][0:4] == 'pawn' and self.live_chessboard_situation_complete[k][1] == self.color:
                                        for m in chessboard_square_corners:
                                            if m[5] == k:
                                                filter_points = np.array(
                                                    [m[0], m[1], m[2], m[3]])
                                                pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                                                    depth_image_state2, state2_image, filter_points,move, quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315])
                                                if occupancy != True:
                                                    for l in self.potential_altered_squares:
                                                        if l[1] == '3' and l != j and self.live_chessboard_situation_complete[l][1] == 'none':
                                                            for m in chessboard_square_corners:
                                                                if m[5] == l:
                                                                    filter_points = np.array(
                                                                        [m[0], m[1], m[2], m[3]])
                                                                    pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                                                                        depth_image_state2, state2_image, filter_points,move, quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315])

                                                                    if occupancy == True:
                                                                        self.possible_enpasent.append(
                                                                            j)
                                                                        self.possible_enpasent.append(
                                                                            k)
                                                                        self.possible_enpasent.append(
                                                                            l)
                                                                        enpasent = True
        
        elif self.opposite_color == 'black' and castle_done != True and enpasent != True:  # checking for promotion is opponent is  black
            #print('yes'); exit()
            for j in self.potential_altered_squares:
                
                if j[1] == '2' and self.live_chessboard_situation_complete[j][0][0:4] == 'pawn' and self.live_chessboard_situation_complete[j][1] == self.opposite_color:
                    for k in self.potential_altered_squares:
                        # if row 1 was empty befor opponent move
                        if k[1] == '1' and k != j and self.live_chessboard_situation_complete[k][0] == 'none':
                            for m in chessboard_square_corners:
                                if m[5] == k:
                                    filter_points = np.array(
                                        [m[0], m[1], m[2], m[3]])
                                    pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                                        depth_image_state2, state2_image, filter_points,move, quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315])
                                    print('occupancy', occupancy)
                                    if occupancy == True:
                                        self.possible_promotion.append(j)
                                        self.possible_promotion.append(k)
                                        promotion = True
                        # if row 1  is not empty before opponent move
                        elif k[1] == '1' and k != j and self.live_chessboard_situation_complete[k][1] == self.color:
                            for m in chessboard_square_corners:
                                if m[5] == k:
                                    filter_points = np.array(
                                        [m[0], m[1], m[2], m[3]])
                                    pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                                        depth_image_state2, state2_image, filter_points, move, quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315])
                                    #print('pcl-count', pcl_count)
                                    if occupancy == True and mean_rgb_value < 0.5:
                                        self.possible_promotion.append(j)
                                        self.possible_promotion.append(k)
                                        promotion = True

        elif self.opposite_color == 'white' and castle_done != True and enpasent != True:  # checking for promotion if opponent is white
            for j in self.potential_altered_squares:
                if j[1] == '7' and self.live_chessboard_situation_complete[j][0][0:4] == 'pawn' and self.live_chessboard_situation_complete[j][1] == self.opposite_color:
                    
                    for k in self.potential_altered_squares:
                        # if row 8  is  empty before opponent move
                        if k[1] == '8' and k != j and self.live_chessboard_situation_complete[k][0] == 'none':
                            for m in chessboard_square_corners:
                                if m[5] == k:
                                    filter_points = np.array(
                                        [m[0], m[1], m[2], m[3]])
                                    pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                                        depth_image_state2, state2_image, filter_points,move, quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315])
                                    if occupancy == True:
                                        self.possible_promotion.append(j)
                                        self.possible_promotion.append(k)
                                        promotion = True
                        # if row 8  is not empty before opponent move
                        elif k[1] == '8' and k != j and self.live_chessboard_situation_complete[k][1] == self.color:
                            
                            for m in chessboard_square_corners:
                                if m[5] == k:
                                    filter_points = np.array(
                                        [m[0], m[1], m[2], m[3]])
                                    pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                                        depth_image_state2, state2_image, filter_points,move,quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315])
                                    #print('pcl-count', pcl_count)
                                    if occupancy == True and mean_rgb_value > 0.5:
                                        self.possible_promotion.append(j)
                                        self.possible_promotion.append(k)
                                        promotion = True

        
        # if len(self.possible_castle) != 0:
        #     castle_done = True
        # if len(self.possible_enpasent) != 0:
        #     enpasent = True
        # if len(self.possible_promotion) != 0:
        #     promotion = True

        if castle_done != True and enpasent != True and promotion != True:
            for i in self.potential_altered_squares:
                current_square = i
                if self.live_chessboard_situation_complete[current_square][1] == self.opposite_color:
                    self.possible_start_square.append(current_square)

                elif self.live_chessboard_situation_complete[current_square][1] == 'none':
                    self.possible_end_square.append(current_square)
                elif self.live_chessboard_situation_complete[current_square][1] == self.color:
                    self.possible_capture.append(current_square)
            print('self.possible_start_square:',self.possible_start_square)
            print('self.possible_capture_square:',self.possible_capture)
            print('self.possible_end_square:',self.possible_end_square)

            temp_start_square = self.possible_start_square.copy()
            if len(self.possible_start_square) >1:
                for i in self.possible_start_square:
                    square_name = i
                    for j in chessboard_square_corners:
                        if j[5] == square_name:
                            #print(j)
                            filter_points = np.array([j[0], j[1], j[2], j[3]])
                            pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                            depth_image_state2, state2_image, filter_points,move, quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315])
                            #print('pcl-count', pcl_count)
                            if occupancy == True:
                                #print(j,occupancy)
                                temp_start_square.remove(i)
                self.possible_start_square = temp_start_square.copy()

                # to filter out start squre due to oclusion by comparing total no of points in empyt square as a whole
                if len(self.possible_start_square) > 1:
                    max_v=0 
                    for i in self.possible_start_square:
                        square_name = i
                        for j in chessboard_square_corners:
                            if j[5] == square_name:
                                #print(j)
                                filter_points = np.array([j[0], j[1], j[2], j[3]])
                                pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                                depth_image_state2, state2_image, filter_points,move, quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315],min_dist=-7)
                                if max_v<(pcl_count):
                                    max_v = pcl_count
                                    new_cell = square_name
                                print(i,pcl_count)
                    self.possible_start_square = [new_cell]
            
                    
            


            tem_end_square = self.possible_end_square.copy()
            if len(self.possible_end_square)>1:
                for i in self.possible_end_square:
                    #print("possible_end_square:",self.possible_end_square)
                    square_name = i
                    #print("square_ name:",i)
                    for j in chessboard_square_corners:
                        if j[5] == square_name:
                            filter_points = np.array([j[0], j[1], j[2], j[3]])
                            pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                            depth_image_state2, state2_image, filter_points,move, quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315])
                            #print(self.possible_end_square); 
                            if occupancy != True:
                                tem_end_square.remove(square_name)
                            #print(i,occupancy);
                            
                            #print(pcl_count)
                
                self.possible_end_square = tem_end_square.copy()
                #print(self.possible_end_square)
                #exit()

            
            tem_capture = self.possible_capture.copy()
            if len(self.possible_capture)>1:
                for i in self.possible_capture:
                    square_name = i
                    for j in chessboard_square_corners:
                        if j[5] == square_name:
                            filter_points = np.array([j[0], j[1], j[2], j[3]])
                            pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                            depth_image_state2, state2_image, filter_points,move, quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315])
                            if occupancy == True:
                                if self.opposite_color == 'black' and mean_rgb_value >= 0.5:
                                    tem_capture.remove(i)
                                elif self.opposite_color == 'white' and mean_rgb_value < 0.5:
                                    tem_capture.remove(i)
                            #print(mean_rgb_value)
                self.possible_capture = tem_capture.copy()
            
                

            if len(self.possible_start_square) == 0:
                self.possible_start_square.append(self.potential_altered_squares[0])
            
            if len(self.possible_capture) == 0 and len(self.possible_end_square) == 0:
                end = False
                for a in self.potential_altered_squares:
                    if end == True:
                        pass
                    elif a != self.possible_start_square[0]:
                        self.possible_end_square.append(a)
                        end = True
            
            print('self.possible_capture_square:',self.possible_capture)

            if len(self.possible_capture) !=0:
                tem_capture = self.possible_capture.copy()
            if len(self.possible_capture)!=0:
                for i in self.possible_capture:
                    square_name = i
                    for j in chessboard_square_corners:
                        if j[5] == square_name:
                            filter_points = np.array([j[0], j[1], j[2], j[3]])
                            pcl_count, mean_rgb_value, occupancy = dip.depth_pcl_counting(
                            depth_image_state2, state2_image, filter_points,move, quat=[-0.672, 0.680, -0.202, 0.214], trans=[0.232, 0.015, 1.315])
                            if occupancy == True:
                                if self.opposite_color == 'black' and mean_rgb_value >= 0.5:
                                    tem_capture.remove(i)
                                elif self.opposite_color == 'white' and mean_rgb_value < 0.5:
                                    tem_capture.remove(i)
                            #print(mean_rgb_value)
                self.possible_capture = tem_capture.copy()
                if len(self.possible_capture)!=0:
                    self.possible_end_square =[]

            #print('potential_squares are:',self.potential_altered_squares)
            
            if __name__ =='__main__':
                print('after pcl processing:\nstart',
                      self.possible_start_square)
                print('end', self.possible_end_square)
                print('capture', self.possible_capture)

        return self.possible_capture, self.possible_start_square, self.possible_end_square, self.possible_castle, self.possible_enpasent, self.possible_promotion


if __name__ == '__main__':
    oc = OccupancyChecker()
    # to test different moves individually ofline
    
    # depth_image_state2 = np.genfromtxt(os.path.join(os.path.realpath(os.path.dirname(
    #     __file__)), 'move_detection/csv/move_7', 'empty_depth_image.csv'), delimiter=',')
    # img1 = cv2.imread(os.path.join(os.path.realpath(os.path.dirname(
    #     __file__)), 'move_detection/csv/move_6', 'empty_rgb_image.png'))
    # img2 = cv2.imread(os.path.join(os.path.realpath(os.path.dirname(
    #     __file__)), 'move_detection/csv/move_7', 'empty_rgb_image.png'))
    root = os.path.normpath(os.path.join(os.path.realpath(os.path.dirname(__file__)), '..','..'))
    img_path = os.path.join(root, 'data', 'moves', 'move_{}'.format(move))
    depth_image_state2 = np.genfromtxt(os.path.join(img_path,'depth_after.csv'), delimiter=',')
    img1 = cv2.imread(os.path.join(img_path,'rgb_before.png'))
    img2 = cv2.imread(os.path.join(img_path,'rgb_after.png'))
     
    # cv2.imshow('img_1', img1)
    # cv2.imshow("img_2", img2)
    # cv2.waitKey(0)
    tiago_colour = 'white'
    debug = False
    
    potential_capture, potential_start, potential_end, potential_castle, potential_enpasent,potential_promotion = oc.opponent_move(
        img1, img2, depth_image_state2,move)
    print('after pclpotential_start_squares:', potential_start)
    print('potential_end_square:', potential_end)
    print('potential_capture:', potential_capture)
    print('potential_castle:', potential_castle)
    print('potential_enpasent:', potential_enpasent)
    print('potential_promotion:', potential_promotion)

    # print(chessboard_square_corners[60])
    '''
    for de bug change move near line 20-40
    '''