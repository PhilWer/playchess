#!/usr/bin/env python3
# Processing of the pointcloud over the chessboard

# Rospy for the subscriber
import rospy
from sensor_msgs.msg import PointCloud2, PointField, Image, CompressedImage
from std_msgs.msg import Bool, Int16, String
import numpy as np
import open3d as o3d
import os
import sensor_msgs.point_cloud2 as pc2
import ctypes
import struct
import yaml
import pickle
import copy
import cv2
import csv
import message_filters
from time import time
from decoder import decode_CompressedImage_depth


# My scripts
import config as cfg
from useful_functions.move_detection_ssim_and_pcl import OccupancyChecker
ready = False

PLAYCHESS_PKG_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__),'..'))
MOVES_DATA_DIR = PLAYCHESS_PKG_DIR + '/data/moves'

# Publishers initialization
state_publisher = rospy.Publisher('/state', Int16, queue_size=10)
promotion_happened_publisher = rospy.Publisher(
    '/promotion_happened', String, queue_size=10)
en_passant_square_publisher = rospy.Publisher(
    '/en_passant_square', String, queue_size=10)
castle_square_publisher = rospy.Publisher(
    '/castle_square', String, queue_size=10)
opponent_move_start_square_publisher = rospy.Publisher(
    '/opponent_move_start_square', String, queue_size=10)
opponent_move_end_square_publisher = rospy.Publisher(
    '/opponent_move_end_square', String, queue_size=10)

# Defines
imported_chessboard_squares = PLAYCHESS_PKG_DIR + \
    '/config/simul_config_not_transformed.yaml'
imported_chessboard_vertices = PLAYCHESS_PKG_DIR + \
    '/config/chessboard_vertices_not_transformed.yaml'
imported_chessboard_vertices_pickle = PLAYCHESS_PKG_DIR + \
    '/config/vertices_not_transformed.pickle'
imported_configurations = PLAYCHESS_PKG_DIR + \
    '/scripts/config/simulation_config.yaml'
imported_chessboard_squares_pickle = PLAYCHESS_PKG_DIR + \
    '/config/simul_config_not_transformed.pickle'

# Flags initialization
analysis_not_done_yet = True


class DepthProcessing:
    # Class containing functions to process depth and rgb images to detect the opponent move.
    def __init__(self):
        # Initialize the move counter (used to name data according to the move)
        self.move = 0
        self.rgb_img_before = None
        self.rgb_img_after = None
        self.depth_arr_after = None

        # Message initialiation
        self.state = Int16()

        self.verbose = False

        with open(imported_chessboard_squares_pickle, 'rb') as fin:
            self.squares_centers = pickle.load(fin, encoding='latin1')

        # Import the chessboard vertices as computed with CV
        with open(imported_chessboard_vertices_pickle, 'rb') as file:
            self.vertices = pickle.load(file, encoding='latin1')

        # Load configurations
        with open(imported_configurations) as file:
            self.config = yaml.load(file, Loader=yaml.Loader)
        self.color = self.config.get('color')
        self.dir_live_chessboard_situation = PLAYCHESS_PKG_DIR + \
            "/scripts/live_chessboard_situation.yaml"
        self.starting_chessboard_situation_complete = cfg.starting_chessboard_situation_complete
        self.pieces_coordinates = cfg.pieces_coordinates
        if self.color == 'white':
            self.squares_to_index = cfg.squares_to_index_white
            self.rows_indexes = cfg.rows_white
            self.opposite_color = 'black'
            self.castle_squares = ['a8', 'e8', 'h8',
                                   'king_e8', 'rook_a8', 'rook_h8']
            self.castle_ending_squares = ['d8', 'c8', 'g8', 'f8']
            self.seventh_row = 'row7'
            self.second_row = 'row2'
            self.columns_indexes = cfg.columns
        else:
            self.squares_to_index = cfg.squares_to_index_black
            self.rows_indexes = cfg.rows_black
            self.opposite_color = 'white'
            self.castle_squares = ['a1', 'e1', 'h1',
                                   'king_e1', 'rook_a1', 'rook_h1']
            self.castle_ending_squares = ['d1', 'c1', 'g1', 'f1']
            self.seventh_row = 'row2'
            self.second_row = 'row7'
            self.columns_indexes = cfg.columns_black

    def move_identification(self,move):
        # Function to look for the presence of pieces over the chessboard squares.
        # chessboard: PointCloud of the isolated chessboard.
        # squares_boundboxes: a list containing the 64 bounding boxes to isolate each square and the things inside them.
        # threshold: threshold of points to consider a square occupied or not

        castle = False
        en_passant = False
        en_passant_square = 'none'
        ending_square = 'none'
        promotion = False
        self.castle_squares_found = 'none'
        # Save the live chessboard situation
        with open(self.dir_live_chessboard_situation, 'rb') as live_file:
            self.live_chessboard_situation = yaml.load(
                live_file.read(), Loader=yaml.Loader)

        opponent_move = OccupancyChecker()
        self.possible_capture, self.possible_start_square, self.possible_end_square, self.possible_castle, self.possible_enpasent, self.possible_promotion = opponent_move.opponent_move(
            self.rgb_img_before, self.rgb_img_after, self.depth_arr_after,move, False)
        print('self.possible_capture, self.possible_start_square, self.possible_end_square, self.possible_castle, self.possible_enpasent, self.possible_promotion:\n',self.possible_capture, self.possible_start_square, self.possible_end_square, self.possible_castle, self.possible_enpasent, self.possible_promotion)
        if len(self.possible_castle) != 0:
            castle = True
            if 'h8' in self.possible_castle:
                self.castle_squares_found = ['g8', 'f8']
            elif 'a8' in self.possible_castle:
                self.castle_squares_found = ['c8', 'd8']
            elif 'h1' in self.possible_castle:
                self.castle_squares_found = ['g1', 'f1']
            elif 'a1' in self.possible_castle:
                self.castle_squares_found = ['c1', 'd1']
            # self.castle_squares_found = cast_sq
            checking_flag = False
            en_passant = False
            en_passant_square = 'none'
            promotion = False
            promoted_piece = 'none'
            ending_square = 'castle'
            move_square = 'castle'
            moved_piece = 'castle'
            ending_square = 'castle'
        
        elif len(self.possible_enpasent) != 0:
            en_passant = True
            for i in self.possible_enpasent:
                if self.opposite_color == 'white':
                    if i[1] == '5' and self.live_chessboard_situation[i][1] == self.opposite_color:
                        move_square = i
                        moved_piece = self.live_chessboard_situation[i][0]
                    elif i[1] == '6':
                        ending_square = i
                    elif i[1] =='5' and self.live_chessboard_situation[i][1] == self.color:
                        en_passant_square = i
                elif self.opposite_color == 'black':
                    if i[1] == '4' and self.live_chessboard_situation[i][1] == self.opposite_color:
                        move_square = i
                        moved_piece = self.live_chessboard_situation[i][0]
                    elif i[1] == '3':
                        ending_square = i
                    elif i[1] =='4' and self.live_chessboard_situation[i][1] == self.color:
                        en_passant_square = i

        elif len(self.possible_promotion) != 0:
            promotion = True
            for i in self.possible_promotion:
                if self.opposite_color == 'white':
                    if i[1] == '7':
                        move_square = i
                        moved_piece = self.live_chessboard_situation[i][0]
                    elif i[1] == '8':
                        ending_square = i

                elif self.opposite_color == 'black':
                    if i[1] == '2':
                        move_square = i
                        moved_piece = self.live_chessboard_situation[i][0]
                    elif i[1] == '1':
                        ending_square = i

        elif len(self.possible_start_square) != 0:

            move_square = self.possible_start_square[0]
            moved_piece = self.live_chessboard_situation[move_square][0]
            if len(self.possible_capture) != 0:
                ending_square = self.possible_capture[0]
            elif len(self.possible_end_square) != 0:
                ending_square = self.possible_end_square[0]

        # else:
        #     if len(self.possible_start_square) != 0:
        #         if len(self.possible_start_square)==1:
        #             move_square= self.possible_start_square[0]
        #             moved_piece = self.live_chessboard_situation[move_square][0]
        #         elif len(self.possible_start_square) >1:
        #             print('start squares option :',self.possible_start_square)
        #             move_square = input('I am confused!\nPls enter start square :\n')
        #             moved_piece = self.live_chessboard_situation[move_square][0]
            
        #     else:
        #         move_square = input('I am confused!\nPls enter start square :\n')
        #         moved_piece = self.live_chessboard_situation[move_square][0]
            
        #     if len(self.possible_end_square)!=0:
        #         if len(self.possible_end_square)==1:
        #             ending_square = self.possible_end_square[0]
        #         else:
        #             print('end square option:',self.possible_end_square)
        #             ending_square = input('I am confused!\nPls enter end square :\n')
            
        #     elif len(self.possible_end_square)==0 and len(self.possible_capture)==0:
        #         ending_square = input('I am confused!\nPls enter end square :\n')
            
        #     elif len(self.possible_end_square)!=0 and len(self.possible_capture)!=0:
        #         ending_square = input('I am confused!\nPls enter end square :\n')

        #     if len(self.possible_capture)!=0:
        #         if len(self.possible_capture)==1:
        #             ending_square = self.possible_capture[0]
                
        #         else:
        #             print('end square option:',self.possible_capture)
        #             ending_square = input('I am confused!\nPls enter end square :\n')

        #     else:
        #         ending_square = input('I am confused!\nPls enter end square :\n')

        #     move_square = self.possible_start_square[0]
        #     moved_piece = self.live_chessboard_situation[move_square][0]
        #     if len(self.possible_capture) != 0:
        #         ending_square = self.possible_capture[0]
        #     elif len(self.possible_end_square) != 0:
        #         ending_square = self.possible_end_square[0]
        
        if en_passant:
            self.live_chessboard_situation[ending_square][0] = self.live_chessboard_situation[move_square][0]
            self.live_chessboard_situation[ending_square][1] = self.live_chessboard_situation[move_square][1]
            self.live_chessboard_situation[move_square][0] = 'none'
            self.live_chessboard_situation[move_square][1] = 'none'
            #change enpasent square to none
            print('self.live_chessboard_situation[en_pasent_square]',self.possible_enpasent,)
            for i in self.possible_enpasent:
                if i != ending_square and i != move_square and i[1]=='4' and self.live_chessboard_situation[i][1] == self.color:
                
                    en_pasent_square = i

                    # self.live_chessboard_situation[en_pasent_square][0] = 'none'
                    # self.live_chessboard_situation[en_pasent_square][1] = 'none'

            self.found = True



        else:
            # updating the live_chessboard_situation
            # self.live_chessboard_situation[ending_square][0] = self.live_chessboard_situation[move_square][0]
            # self.live_chessboard_situation[ending_square][1] = self.live_chessboard_situation[move_square][1]
            # self.live_chessboard_situation[move_square][0] = 'none'
            # self.live_chessboard_situation[move_square][1] = 'none'
            self.found = True


        if castle:
            # If castle happened, send a message to correctly update the GUI.
            # Publish one square of the castle.
            castle_square_publisher.publish(self.castle_squares_found[0])

        if promotion:
            # Change status to the status of asking the opponent promoted piece.
            state_publisher.publish(16)
            promotion_happened_publisher.publish(self.opposite_color)
            rospy.sleep(2)

        if ending_square == 'castle':
            # If castle happened, send a message to correctly update the GUI.
            # Publish one square of the castle
            castle_square_publisher.publish(self.castle_squares_found[0])

        if en_passant:
            # If en-passant happened, send a message to correctly update the GUI.
            en_passant_square_publisher.publish(en_passant_square)

        return moved_piece, move_square, ending_square

    def identify_moved_piece(self,move):
        global analysis_not_done_yet

        # Understand which piece has been moved
        start = time()
        moved_piece, start_square, end_square = self.move_identification(move
        )
        end = time()
        elapsed_time = end- start

        print('MOVED PIECE: ' + str(moved_piece))
        print('STARTING SQUARE: ' + str(start_square))
        print('ENDING SQUARE: ' + str(end_square))


        mv_folder = MOVES_DATA_DIR + '/move_{}'.format(self.move)
        # saving the recognized move
        with open(os.path.join(mv_folder,'recognized_move.txt'),"w") as file:
            file.write("moved piece:{}".format(str(moved_piece)))
            file.write("sratring square:{}".format(str(start_square)))
            file.write("moved piece: {}".format(str(end_square)))
            file.write("time_taken_for_move_recognition:{}".format(str(elapsed_time)))
            
        # Send messages regarding the move executed by the opponent to change the GUI.
        rospy.loginfo('Publishing the opponent move...')
        # print('Time for move recognition: ' +
        #       str(time() - self.t_recognition)+' s')
        opponent_move_start_square_publisher.publish(start_square)
        opponent_move_end_square_publisher.publish(end_square)

        analysis_not_done_yet = False

    def CallbackState(self, data):
        self.state.data = data.data
        global analysis_not_done_yet
        global calibration_not_done_yet
        global occupied_threshold
        global color_threshold
        global is_color_over
        print(data.data)
        save = True  # TODO. Expose this param in launch
        if data.data == 14:
            # Get RGB (+ depth, for reproducibility with v1.0)

            self.rgb_img_before, depth_img_before = self._get_rgb_and_depth()
            rospy.loginfo('image after tiago move acquired')
            self.move += 1
            print('move:',self.move)
            if save:
                mv_folder = MOVES_DATA_DIR + '/move_{}'.format(self.move)
                try:
                    os.mkdir(mv_folder)
                except FileExistsError:
                    rospy.logwarn(
                        'The content of the ' + 'move_{}'.format(self.move) + ' will be overwritten.')
                # Save the RGB and depth data BEFORE the execution of the opponent's move
                cv2.imwrite(mv_folder + '/rgb_before.png', self.rgb_img_before)
                cv2.imwrite(mv_folder + '/depth_before.png', depth_img_before)
                np.savetxt(mv_folder + '/depth_before.csv',
                           depth_img_before, delimiter=",")
            
            
                # NOTE. Retrieve the depth data with depth_image = np.genfromtxt(mv_folder + '/depth_before.csv', delimiter=',')

        if data.data == 15:
            # Get RGB and depth
            print(data.data)
            self.rgb_img_after, self.depth_arr_after = self._get_rgb_and_depth()
            if save:
                mv_folder = MOVES_DATA_DIR + '/move_{}'.format(self.move)
                
                #read yaml file to dump it later
                with open(os.path.join(PLAYCHESS_PKG_DIR,'scripts/live_chessboard_situation.yaml'), 'rb') as live_file:
                    save_for_debug = yaml.load(live_file.read(), Loader=yaml.Loader)
                # dump yaml fro debugin
                with open(os.path.join(mv_folder,'live_chessboard_situation.yaml'), "w") as t_p:
                            yaml.dump(save_for_debug, t_p)
                # Save the RGB and depth data AFTER the execution of the opponent's move
                cv2.imwrite(mv_folder + '/rgb_after.png', self.rgb_img_after)
                cv2.imwrite(mv_folder + '/depth_after.png',
                            self.depth_arr_after)
                np.savetxt(mv_folder + '/depth_after.csv',
                           self.depth_arr_after, delimiter=",")
                # Retrieve the depth data with depth_image = np.genfromtxt(mv_folder + '/depth_after.csv', delimiter=',')
            # Identify the move and broadcast messages accordingly
            self.identify_moved_piece(self.move)


        elif data.data == 50:
            # TODO. Verify if needed, then remove.
            state_publisher.publish(9)

        elif data.data == 17:
            analysis_not_done_yet = True  # Change back the flag

    #######################
    ### PRIVATE METHODS ###
    #######################
    def _get_rgb_and_depth(self):
        # Get RGB and depth data
        img_msg = rospy.wait_for_message(
            '/xtion/rgb/image_raw', Image, timeout=30)
        rgb_img = self._convert_image_msg(img_msg, rgb=True)
        depth_msg = rospy.wait_for_message(
            '/xtion/depth/image_raw', Image, timeout=30)
        depth_img = self._convert_image_msg(depth_msg, rgb=False)
        return rgb_img, depth_img

    def _convert_image_msg(self, data, rgb=True):

        try:
            if rgb:
                img = np.frombuffer(data.data, dtype=np.uint8).reshape(
                    data.height, data.width, -1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            else:   # if depth
                depth_image_norm_factor = 1
                img = np.frombuffer(data.data, dtype=np.uint16).reshape(
                    data.height, data.width)
                img = img.astype(np.float32) / depth_image_norm_factor

        except Exception as e:
            print(e)

        return img


def main():
    rospy.init_node('pcl_processor')
    depth_processor = DepthProcessing()

    # Initialize a subscriber to monitor the state.
    rospy.Subscriber("/state", Int16, depth_processor.CallbackState)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down the CV module.")


if __name__ == '__main__':
    main()
