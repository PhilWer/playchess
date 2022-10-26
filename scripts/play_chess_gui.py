#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, QEvent, pyqtSignal, QSize
from PyQt5.QtWidgets import QApplication
from cv_bridge import CvBridge, CvBridgeError
import rospy
import sys
import time
import yaml
import cv2
import os
from std_msgs.msg import Bool, Int16, String
from geometry_msgs.msg import Point
from sensor_msgs.msg import CompressedImage
import moveit_commander
from geometry_msgs.msg import Quaternion, PointStamped, Pose, Point, PoseStamped, WrenchStamped

import config_pyqt_white as cfg_w
import config_pyqt_black as cfg_b
import config as cfg

from InitializationWindow import Ui_InitializationWindow #Main window
from ColorAskingWindow import Ui_ColorAskingWindow
from StartSegmentationWindow import Ui_StartSegmentation
from ChessboardSegmentationWindow import Ui_SegmentationExecution
from SegmentationConfirmationAskingWindow import Ui_SegmentationConfirmationAskingWindow
from StartMarkersSearchWindow import Ui_StartMarkersSearch
from MarkersSearchWindow import Ui_SearchExecution
from SearchConfirmationAskingWindow import Ui_SearchConfirmationAskingWindow
from TiagoPreparationWindow import Ui_TiagoPreparation
from Chessboard_White import Ui_Chessboard_White
from Chessboard_Black import Ui_Chessboard_Black
from WhitePromotionWindow import Ui_WhitePromotion
from BlackPromotionWindow import Ui_BlackPromotion
from ManualModeWindow import Ui_ManualModeWindow
from WizardWindow import Ui_WizardWindow


PLAYCHESS_PKG_DIR = '/home/luca/tiago_public_ws/src/tiago_playchess'
GUI_PKG_DIR       = '/home/luca/tiago_public_ws/src/chess_gui'
#Starts a new node
rospy.init_node('GUI_commander', anonymous = True)

#Global variables initialization
update_flag = True
current_window = 1
square_object = 'none'
square_captured_piece = 'none'
simul_config = rospy.get_param('/tiago_playchess/simul_config')
imported_configurations = PLAYCHESS_PKG_DIR + '/scripts/config/simulation_config.yaml'

#Defines
pawns_w = ['pawn_a2', 'pawn_b2', 'pawn_c2', 'pawn_d2', 'pawn_e2', 'pawn_f2', 'pawn_g2', 'pawn_h2']
pawns_b = ['pawn_a7', 'pawn_b7', 'pawn_c7', 'pawn_d7', 'pawn_e7', 'pawn_f7', 'pawn_g7', 'pawn_h7']

#Publishers initialization
state_publisher = rospy.Publisher('/state', Int16, queue_size = 10)
en_passant_square_to_move_publisher = rospy.Publisher('/en_passant_square_to_move', String, queue_size = 10)
messaggio_publisher = rospy.Publisher('/move_the_arm', Bool, queue_size = 10)

class MainWindow(QtWidgets.QMainWindow, Ui_InitializationWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        #Subscribers initialization.
        rospy.Subscriber("/state", Int16, self.ChangeState_Callback)
        rospy.Subscriber("/color", String, self.Color_Callback)
        rospy.Subscriber("/button_one_point_one", Bool, self.SelectButton)

        self.bridge = CvBridge()

        #Connect signals and slots.
        self.GoPushButton.clicked.connect(self.GoToColorChoice)
        self.RepositionPushButton.clicked.connect(self.Reposition)

        #Create instances of the children.
        self.ColorWindow = ColorAskingWindow()
        
        #Connects signals and slots.
        self.ColorWindow.clicked.connect(self.GoToStartSegmentation)

        #Variables initialization
        self.current_focus = 'GoPushButton'
        self.reposition = False

    def MoveHighlight(self, data):
        #If a movement in the x direction, perform the corresponding movement on the GUI.
        if data.x != 0: # and self.state == 2:
            #Change the highlighted button
            if self.current_focus == 'GoPushButton':
                self.current_focus = 'RepositionPushButton'
                self.RepositionPushButton.setFocus()
                self.RepositionPushButton.setStyleSheet("background-color: rgb(176, 202, 220);")
                self.GoPushButton.setStyleSheet("")                
            elif self.current_focus == 'RepositionPushButton':
                self.current_focus = 'GoPushButton'
                self.GoPushButton.setFocus()
                self.GoPushButton.setStyleSheet("background-color: rgb(176, 202, 220);")
                self.RepositionPushButton.setStyleSheet("")

            print('CURRENT FOCUS: '+str(self.current_focus))

    def SelectButton(self, data):
        if data:
            if self.current_focus == 'GoPushButton':
                self.GoToColorChoice()
            elif self.current_focus == 'RepositionPushButton':
                self.Reposition()

    def GoToColorChoice(self):
        self.ColorWindow.show()
        self.hide()
        #Change operational state.
        state_publisher.publish(2)

    def Reposition(self):
        #Change operational state.
        state_publisher.publish(60)
        self.reposition = True

    def GoToStartSegmentation(self, color):
        self.StartSegmentationWindow = StartSegmentationWindow()
        self.StartSegmentationWindow.clicked.connect(self.GoToSegmentation)
        self.StartSegmentationWindow.show()
        self.ColorWindow.hide()

    def GoToSegmentation(self, btn):
        if btn == 1: #Make the user choose the color again
            state_publisher.publish(2)
            self.ColorWindow.show()
            self.StartSegmentationWindow.hide()
        elif btn == 2: #Go to segmentation
            self.SegmentationWindow = SegmentationWindow()
            self.SegmentationWindow.show()
            self.SegmentationWindow.clicked.connect(self.GoToCheckSegmentation)
            self.StartSegmentationWindow.hide()

    def GoToCheckSegmentation(self):
        self.CheckSegmentationWindow = CheckSegmentationWindow()
        self.CheckSegmentationWindow.show()
        self.CheckSegmentationWindow.clicked.connect(self.GoToStartMarkersSearch)
        self.SegmentationWindow.hide()

    def GoToStartMarkersSearch(self, btn):
        if btn == 1: #Perform segmentation again.
            self.StartSegmentationWindow = StartSegmentationWindow()
            self.StartSegmentationWindow.show()
            self.StartSegmentationWindow.clicked.connect(self.GoToSegmentation)
            self.CheckSegmentationWindow.hide()
        elif btn == 2: #Go to start markers search window.
            self.StartMarkersSearchWindow = StartMarkersSearchWindow()
            self.StartMarkersSearchWindow.show()
            self.StartMarkersSearchWindow.clicked.connect(self.GoToMarkersSearch)
            self.CheckSegmentationWindow.hide()

    def GoToMarkersSearch(self):
        self.MarkersSearchWindow = MarkersSearchWindow()
        self.MarkersSearchWindow.show()
        self.MarkersSearchWindow.clicked.connect(self.GoToCheckSearch)
        self.StartMarkersSearchWindow.hide()

    def GoToCheckSearch(self):
        self.CheckSearchWindow = CheckMarkersSearchWindow()
        self.CheckSearchWindow.show()
        self.CheckSearchWindow.clicked.connect(self.GoToTiagoPreparation)
        self.MarkersSearchWindow.hide()

    def GoToTiagoPreparation(self, btn):
        if btn == 1: #Perform markers search again.
            self.StartMarkersSearchWindow = StartMarkersSearchWindow()
            self.StartMarkersSearchWindow.show()
            self.StartMarkersSearchWindow.clicked.connect(self.GoToMarkersSearch)
            self.CheckSearchWindow.hide()
        elif btn == 2: #Go to TIAGo's preparation window.
            self.TiagoPreparationWindow = TiagoPreparationWindow()
            self.TiagoPreparationWindow.show()
            self.TiagoPreparationWindow.clicked.connect(lambda: self.GoToChessboard())
            self.CheckSearchWindow.hide()

    def GoToChessboard(self):
        #Change window to the chessboard window       
        if color == 'white':
            self.ChessboardWindow = ChessboardWhiteWindow()
        elif color == 'black':
            #Move the arm away to see the chessboard and look down
            rospy.loginfo('Moving the arm away and looking down')
            messaggio_publisher.publish(True)
            self.ChessboardWindow = ChessboardBlackWindow()
        self.ChessboardWindow.show()
        self.ChessboardWindow.promotion.connect(self.GoToPromotion)
        self.ChessboardWindow.close_promotion.connect(self.ClosePromotion)
        self.ChessboardWindow.manual_mode.connect(self.ManualMode)
        self.ChessboardWindow.back_to_game.connect(self.BackToGame)

        self.WizardWindow = WizardWindow()
        self.WizardWindow.show()
        self.WizardWindow.close.connect(self.CloseWizard)

        self.TiagoPreparationWindow.hide()

    def GoToPromotion(self, btn):
        #Open the promotion widget.
        if btn == 1:
            self.WhitePromotionWindow = WhitePromotionWindow()
            self.WhitePromotionWindow.show()
        elif btn == 2:
            self.BlackPromotionWindow = BlackPromotionWindow()
            self.BlackPromotionWindow.show()

    def ClosePromotion(self, btn):
        #Close the promotion widget.
        if btn == 1:
            self.WhitePromotionWindow.close()
        elif btn == 2:
            self.BlackPromotionWindow.close()
        if state == 12:
            state_publisher.publish(13)
        elif state == 16:
            state_publisher.publish(17)

    def ManualMode(self):
        #Switch to manual mode: make a widget appear to change the GUI manually. NB we are in state 70.
        self.ManualModeWindow = ManualMode()
        self.ManualModeWindow.show()
        self.ManualModeWindow.clicked.connect(self.ManuallyChange)

    def BackToGame(self):
        #Go back to the "TIAGo play game" mode. Close the manual mode widget.
        self.ManualModeWindow.hide()

    def ManuallyChange(self, btn):
        #Save the action that the user wants to do.
        global action
        if btn == 1: #White pawn
            action = 'w_pawn'
        elif btn == 2: #Black pawn
            action = 'b_pawn'
        elif btn == 3: #White knight
            action = 'w_knight'
        elif btn == 4: #Black knight
            action = 'b_knight'
        elif btn == 5: #White bishop
            action = 'w_bishop'
        elif btn == 6: #Black bishop
            action = 'b_bishop'
        elif btn == 7: #White rook
            action = 'w_rook'
        elif btn == 8: #Black rook
            action = 'b_rook'
        elif btn == 9: #White queen
            action = 'w_queen'
        elif btn == 10: #Black queen
            action = 'b_queen'
        elif btn ==11: #White king
            action = 'w_king'
        elif btn == 12: #Black king
            action = 'b_king'
        elif btn == 13: #Delete piece
            action = 'delete'

    def ChangeState_Callback(self, data):
        global state
        state = data.data
        ###############
        if state == 11:
            self.ChessboardWindow.indicationLabel.setStyleSheet("background-color: rgb(228, 244, 107);")
            self.ChessboardWindow.indicationLabel.setText("Your turn")

        if state == 13:
            self.ChessboardWindow.indicationLabel.setStyleSheet("background-color: rgb(176, 202, 220);")
            self.ChessboardWindow.indicationLabel.setText("Move received!")

        if state == 14:
            self.ChessboardWindow.indicationLabel.setStyleSheet("background-color: rgb(186, 189, 182);")
            self.ChessboardWindow.indicationLabel.setText("Waiting")

        if state == 15:
            self.ChessboardWindow.indicationLabel.setStyleSheet("background-color: rgb(247, 185, 185);")
            self.ChessboardWindow.indicationLabel.setText("Pointcloud processing")

        if state == 17:
            self.ChessboardWindow.indicationLabel.setStyleSheet("background-color: rgb(141, 231, 144);")
            self.ChessboardWindow.indicationLabel.setText("Move recognized!")
        '''
        if state == 1:

            if ready == True:
                

                state_publisher.publish(11)

                self.ChessboardWindow = ChessboardWhiteWindow()
                self.ChessboardWindow.show()
                self.ChessboardWindow.promotion.connect(self.GoToPromotion)
                self.ChessboardWindow.close_promotion.connect(self.ClosePromotion)
                self.ChessboardWindow.manual_mode.connect(self.ManualMode)
                self.ChessboardWindow.back_to_game.connect(self.BackToGame)

                self.hide()

                import config as cfg
                import moveit_commander
                from geometry_msgs.msg import Quaternion, PointStamped, Pose, Point, PoseStamped, WrenchStamped


                self.pieces_coordinates = cfg.pieces_coordinates
                self.squares_to_index = cfg.squares_to_index_white

                self.scene = moveit_commander.PlanningSceneInterface() #The interface with the world surrounding the robot

                for name in self.pieces_coordinates:
                    self.scene.remove_world_object(name)

                with open(simul_config) as file:
                    square_centers = yaml.load(file)

                #Add cylinders corresponding to the pieces to the planning scene in the starting position
                for name in self.pieces_coordinates:
                    pose = PoseStamped()
                    pose.header.frame_id = 'base_footprint'
                    pose.header.stamp = rospy.Time.now()
                    pose.pose = Pose(Point(square_centers[self.squares_to_index[self.pieces_coordinates[name][0]]].point.x, square_centers[self.squares_to_index[self.pieces_coordinates[name][0]]].point.y, square_centers[self.squares_to_index[self.pieces_coordinates[name][0]]].point.z + self.pieces_coordinates[name][1]['height']/2), Quaternion(0, 0, 0, 1))
                    self.scene.add_cylinder(name, pose, height = self.pieces_coordinates[name][1]['height'], radius = self.pieces_coordinates[name][1]['diameter']/2)
    
        ######################
        '''

        if state == 60: #If the status is 60, acquire and show the image of the current chessboard setup.
            self.done = False
            #Initialize the RGB subscriber.
            while not self.done:
                self.sub = rospy.Subscriber('/xtion/rgb/image_rect_color/compressed', CompressedImage, self.save_rgb)
            time.sleep(1)
            #Show the image of the current setup.
            sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            sizePolicy.setHorizontalStretch(0)
            sizePolicy.setVerticalStretch(0)
            sizePolicy.setHeightForWidth(self.CurrentChessboardImageLabel.sizePolicy().hasHeightForWidth())
            self.CurrentChessboardImageLabel.setSizePolicy(sizePolicy)
            self.CurrentChessboardImageLabel.setMinimumSize(QtCore.QSize(200, 250))
            self.CurrentChessboardImageLabel.setMaximumSize(QtCore.QSize(320, 250))
            self.CurrentChessboardImageLabel.setPixmap(QtGui.QPixmap(GUI_PKG_DIR + "/images/current_setup.png"))

            time.sleep(1)
            state_publisher.publish(20)

        if state == 20: #If the status is 2, enable the button to be pushed to procees to the color choice phase.
            if not self.reposition:
                self.GoPushButton.setEnabled(True)
                self.GoPushButton.setStyleSheet("background-color: rgb(176, 202, 220);")
                self.RepositionPushButton.setEnabled(True)

    def Color_Callback(self, data):
        global color
        color = data.data
        ################
        with open(imported_configurations) as file:
            params = yaml.load(file, Loader = yaml.Loader)
        params['color'] = color
        with open(imported_configurations, "w") as t_p:
            yaml.dump(params, t_p)
        ###################

    def save_rgb(self, rgb_msg):
        if not self.done:
            try:
                #Try to decode the CompressedImage message from the RGB camera stream to a CV2 image.
                cv2_image = self.bridge.compressed_imgmsg_to_cv2(rgb_msg, desired_encoding = 'passthrough')
            except CvBridgeError as e:
                rospy.logerr(e)
            else:
                #Save the image to a .png file
                cv2.imwrite(os.path.join(GUI_PKG_DIR + '/images', 'current_setup.png'), cv2_image)
                time.sleep(1)
                self.sub.unregister()
                self.done = True

    def CloseWizard(self):
        self.WizardWindow.close()


###################################################################


class ColorAskingWindow(QtWidgets.QWidget, Ui_ColorAskingWindow):
    clicked = QtCore.pyqtSignal(int)
    def __init__(self):
        super(ColorAskingWindow, self).__init__()
        self.setupUi(self)

        #Variables initialization
        self.current_focus_color = 'WhiteBtn'

        #Subscribers initialization
        rospy.Subscriber("/cursor_step_two", Point, self.MoveHighlight)
        rospy.Subscriber("/button_two", Bool, self.SelectButton)

        #Publishers initialization
        self.color_publisher = rospy.Publisher('/color', String, queue_size = 10)

        #Connect signals and slots.
        self.WhiteBtn.clicked.connect(self.white_choice)
        self.BlackBtn.clicked.connect(self.black_choice)

    def white_choice(self):
        #Save the choice of color, change GUI window and change operation status.

        #Publish the color with which TIAGo will play.
        self.color_publisher.publish('white')
        time.sleep(1)

        #Change status.
        state_publisher.publish(3)

        self.clicked.emit(1) #If 1 is emitted, white has been chosen.

    def black_choice(self):
        #Save the choice of color, change GUI window and change operation status

        #Publish the color with which TIAGo will play.
        self.color_publisher.publish('black')
        time.sleep(1)

        #Change status
        state_publisher.publish(3)
        
        self.clicked.emit(2) #If 2 is emitted, black has been chosen

    def MoveHighlight(self, data):
        #If a movement in the x direction, perform the corresponding movement on the GUI.
        if data.x != 0: # and self.state == 2:
            #Change the highlighted button
            if self.current_focus_color == 'WhiteBtn':
                self.current_focus_color = 'BlackBtn'
                self.BlackBtn.setFocus()
                self.BlackBtn.setStyleSheet("background-color: rgb(176, 202, 220);")
                self.WhiteBtn.setStyleSheet("")                
            elif self.current_focus_color == 'BlackBtn':
                self.current_focus_color = 'WhiteBtn'
                self.WhiteBtn.setFocus()
                self.WhiteBtn.setStyleSheet("background-color: rgb(176, 202, 220);")
                self.BlackBtn.setStyleSheet("")

            print('CURRENT FOCUS: '+str(self.current_focus_color))

    def SelectButton(self, data):
        if data:
            if self.current_focus_color == 'WhiteBtn':
                self.white_choice()
            elif self.current_focus_color == 'BlackBtn':
                self.black_choice()


###################################################################


class StartSegmentationWindow(QtWidgets.QWidget, Ui_StartSegmentation):
    clicked = QtCore.pyqtSignal(int)
    def __init__(self):
        super(StartSegmentationWindow, self).__init__()
        self.setupUi(self, color)

        #Variables initialization
        self.current_focus_start_segment = 'pushButton'

        #Subscribers initialization.
        rospy.Subscriber("/cursor_step_three", Point, self.MoveHighlight)
        rospy.Subscriber("/button_three", Bool, self.SelectButton)

        #Connect signals and slots.
        self.GoBackButton.clicked.connect(self.go_back)
        self.pushButton.clicked.connect(lambda: self.start_segmentation())

    def start_segmentation(self):
        #Change operation status to status 4 and change GUI window.
        state_publisher.publish(4)

        self.clicked.emit(2) #If 2 is emitted, go to the segmentation phase.  

    def go_back(self):
        self.clicked.emit(1) #If 1 is emitted, the user need to chose TIAGo's color again.

    def MoveHighlight(self, data):
        #If a movement in the x or in the y direction is detected, perform the corresponding movement on the GUI.
        if (data.x != 0 or data.y != 0):
            #Change the highlighted button
            if self.current_focus_start_segment == 'pushButton':
                self.current_focus_start_segment = 'GoBackButton'
                self.GoBackButton.setFocus()
                self.GoBackButton.setStyleSheet("background-color: rgb(176, 202, 220);")
                self.pushButton.setStyleSheet("")
            elif self.current_focus_start_segment == 'GoBackButton':
                self.current_focus_start_segment = 'pushButton'
                self.pushButton.setFocus()
                self.pushButton.setStyleSheet("background-color: rgb(176, 202, 220);")
                self.GoBackButton.setStyleSheet("")

            print('CURRENT HIGHLIGHT: ' + str(self.current_focus_start_segment))

    def SelectButton(self, data):
        if data:
            if self.current_focus_start_segment == 'pushButton':
                self.start_segmentation()
            elif self.current_focus_start_segment == 'GoBackButton':
                self.go_back()


###################################################################


class SegmentationWindow(QtWidgets.QWidget, Ui_SegmentationExecution):
    clicked = QtCore.pyqtSignal()
    def __init__(self):
        super(SegmentationWindow, self).__init__()
        self.setupUi(self)

        #Subscribers initialization
        rospy.Subscriber("/button_end_segmentation", Bool, self.SelectButton)
        rospy.Subscriber("/state", Int16, self.FinishSegmentation)

        #Connect signals and slots.
        self.CheckSegmentationPushButton.clicked.connect(lambda: self.ChangeWindow())

    def FinishSegmentation(self, data):
        if data.data == 30: #If State has changed to State 30 (chessboard segmentation ended), enable the pushbutton, show the "operation complete" text label and hide the other one.
            self.OperationCompletedLabel.setEnabled(True)
            self.SegmentationLabel.setEnabled(False)
            self.movie.stop()
            self.ProgressGifLabel.setEnabled(False)
            self.CheckSegmentationPushButton.setEnabled(True)
            self.CheckSegmentationPushButton.setStyleSheet("background-color: rgb(176, 202, 220);")

    def ChangeWindow(self):
        #Change state.
        state_publisher.publish(5)

        #Change window to the segmentation verification window
        self.clicked.emit()

    def SelectButton(self, data):
        if data:
            if self.CheckSegmentationPushButton.isEnabled():
                self.ChangeWindow()
            else:
                rospy.loginfo('Chessboard segmentation is not finished yet.')


###################################################################


class CheckSegmentationWindow(QtWidgets.QWidget, Ui_SegmentationConfirmationAskingWindow):
    clicked = QtCore.pyqtSignal(int)
    def __init__(self):
        super(CheckSegmentationWindow, self).__init__()
        self.setupUi(self)

        #Variables initialization
        self.current_focus_segm_confirm = 'CorrectPushButton'

        #Subscribers initialization
        rospy.Subscriber("/cursor_step_five", Point, self.MoveHighlight)
        rospy.Subscriber("/button_five", Bool, self.SelectButton)

        #Connect signals and slots
        self.SegmentAgainPushButton.clicked.connect(self.segmentation_again)
        self.CorrectPushButton.clicked.connect(lambda: self.start_markers_search())

    def segmentation_again(self):
        #Go back to the segmentation window and start segmentation again
        #Close Segmentation Confirmation Window
        self.clicked.emit(1) #If 1 is emitted, segmentation need to be performed again.

        #Change operation status.
        state_publisher.publish(3)

    def start_markers_search(self):    
        #Go to the start markers search window
        self.clicked.emit(2) #if 2 is emitted, go to the start markers search window.

        #Change operation status to status 6 (waiting to start the markers search).
        state_publisher.publish(6)

    def MoveHighlight(self, data):
        #If a movement in the x direction is detected, perform the corresponding movement on the GUI.
        if data.x != 0:
            #Change the highlighted button
            if self.current_focus_segm_confirm == 'CorrectPushButton':
                self.current_focus_segm_confirm = 'SegmentAgainPushButton'
                self.SegmentAgainPushButton.setFocus()
                self.SegmentAgainPushButton.setStyleSheet("background-color: rgb(176, 202, 220);")
                self.CorrectPushButton.setStyleSheet("")
            elif self.current_focus_segm_confirm == 'SegmentAgainPushButton':
                self.current_focus_segm_confirm = 'CorrectPushButton'
                self.CorrectPushButton.setFocus()
                self.CorrectPushButton.setStyleSheet("background-color: rgb(176, 202, 220);")
                self.SegmentAgainPushButton.setStyleSheet("")

            print('CURRENT HIGHLIGHT: ' + str(self.current_focus_segm_confirm))


    def SelectButton(self, data):
        if data:
            if self.current_focus_segm_confirm == 'CorrectPushButton':
                self.start_markers_search()
            elif self.current_focus_segm_confirm == 'SegmentAgainPushButton':
                self.segmentation_again()


###################################################################


class StartMarkersSearchWindow(QtWidgets.QWidget, Ui_StartMarkersSearch):
    clicked = QtCore.pyqtSignal()
    def __init__(self):
        super(StartMarkersSearchWindow, self).__init__()
        self.setupUi(self)

        #Subscribers initialization
        rospy.Subscriber("/button_six", Bool, self.SelectButton)

        #Connect signals and slots
        self.StartSearchPushButton.clicked.connect(lambda: self.start_search())

    def start_search(self):
        self.clicked.emit()

        #Change operation status to status 7 and change GUI window.
        state_publisher.publish(7)

    def SelectButton(self, data):
        if data:
            self.start_search()


###################################################################


class MarkersSearchWindow(QtWidgets.QWidget, Ui_SearchExecution):
    clicked = QtCore.pyqtSignal()
    def __init__(self):
        super(MarkersSearchWindow, self).__init__()
        self.setupUi(self)

        #Subscribers initialization
        rospy.Subscriber("/button_seven", Bool, self.SelectButton)
        rospy.Subscriber("/button_end_search", Bool, self.SelectButton)
        rospy.Subscriber("/state", Int16, self.FinishSearch)

        #Connect signals and slots
        self.CheckResultsPushButton.clicked.connect(lambda: self.ChangeWindow())

    def FinishSearch(self, data):
        if data.data == 40: #If State has changed to State 40 (markers search ended), enable the pushbutton, show the "operation complete" text label and hide the other one.
            self.OperationCompletedLabel.setEnabled(True)
            self.SearchLabel.setEnabled(False)
            self.movie.stop()
            self.ProgressGifLabel.setEnabled(False)
            self.CheckResultsPushButton.setEnabled(True)
            self.CheckResultsPushButton.setStyleSheet("background-color: rgb(176, 202, 220);")

    def ChangeWindow(self):
        #Change state.
        state_publisher.publish(8)

        #Change window to the search verification window
        self.clicked.emit()

    def SelectButton(self, data):
        if data:
            if self.CheckResultsPushButton.isEnabled():
                self.ChangeWindow()
            else:
                rospy.loginfo('Markers search is not finished yet.')

    def CloseSearchConfirmWindow(self, data):
        if data:
            self.SearchConfirmationAskingWindow.hide()



###################################################################


class CheckMarkersSearchWindow(QtWidgets.QWidget, Ui_SearchConfirmationAskingWindow):
    clicked = QtCore.pyqtSignal(int)
    def __init__(self):
        super(CheckMarkersSearchWindow, self).__init__()
        self.setupUi(self)

        #Subscribers initialization
        rospy.Subscriber("/cursor_step_eight", Point, self.MoveHighlight)
        rospy.Subscriber("/button_eight", Bool, self.SelectButton)

        #Variables initialization
        self.current_focus_search_confirm = 'CorrectPushButton'

        #Connect signals and slots
        self.SearchAgainPushButton.clicked.connect(self.search_again)
        self.CorrectPushButton.clicked.connect(lambda: self.start_tiago_preparation())

    def search_again(self):
        #Go back to the search window and start markers search again
        self.clicked.emit(1) #If 1 is emitted, the ARUCO markers search is performed again.

        #Change operational status.
        state_publisher.publish(6)

    def start_tiago_preparation(self):
        #Change operation status to status 50.
        state_publisher.publish(50)
        
        #Close the open windows and go to the start markers search window
        self.clicked.emit(2) #If 2 is emitted, pass to the preparation of TIAGo's arm to play the game.

    def MoveHighlight(self, data):
        #If a movement in the x direction is detected, perform the corresponding movement on the GUI.
        if data.x != 0:
            #Change the highlighted button
            if self.current_focus_search_confirm == 'CorrectPushButton':
                self.current_focus_search_confirm = 'SearchAgainPushButton'
                self.SearchAgainPushButton.setFocus()
                self.SearchAgainPushButton.setStyleSheet("background-color: rgb(176, 202, 220);")
                self.CorrectPushButton.setStyleSheet("")
            elif self.current_focus_search_confirm == 'SearchAgainPushButton':
                self.current_focus_search_confirm = 'CorrectPushButton'
                self.CorrectPushButton.setFocus()
                self.CorrectPushButton.setStyleSheet("background-color: rgb(176, 202, 220);")
                self.SearchAgainPushButton.setStyleSheet("")

            print('CURRENT HIGHLIGHT: ' + str(self.current_focus_search_confirm))

    def SelectButton(self, data):
        if data:
            if self.current_focus_search_confirm == 'CorrectPushButton':
                self.start_tiago_preparation()
            elif self.current_focus_search_confirm == 'SearchAgainPushButton':
                self.search_again()


###################################################################


class TiagoPreparationWindow(QtWidgets.QWidget, Ui_TiagoPreparation):
    clicked = QtCore.pyqtSignal()
    def __init__(self):
        super(TiagoPreparationWindow, self).__init__()
        self.setupUi(self)

        #Subscribers initialization
        rospy.Subscriber("/state", Int16, self.FinishPreparation)
        rospy.Subscriber("/button_ten", Bool, self.SelectButton)

        #Connect signals and slots
        self.StartGamePushButton.clicked.connect(lambda: self.ChessboardWindow())

    def FinishPreparation(self, data):
        if data.data == 10: #If State has changed to State 10 (TIAGo preparation ended), enable the pushbutton, show the "preparation complete" text label and hide the other one.
            self.OperationCompletedLabel.setEnabled(True)
            self.SearchLabel.setEnabled(False)
            self.ExplanationLabel.setEnabled(False)
            self.WarningLabel.setEnabled(False)
            self.WarningSignLabel.setEnabled(False)
            self.SuggestionLabel.setEnabled(False)
            self.ImageLabel.setEnabled(False)
            self.movie.stop()
            self.ProgressGifLabel.setEnabled(False)
            #self.TiagoPreparationProgressBar.setProperty("value", 100)
            self.StartGamePushButton.setEnabled(True)
            self.StartGamePushButton.setStyleSheet("background-color: rgb(176, 202, 220);")

    def ChessboardWindow(self):
        self.clicked.emit()
        global update_flag
        #Change operational state: 11 if TIAGo is playing with white, 14 if TIAGo is black (cause he will start from witing for the pushbutton and recognizing the opponent move).
        if color == 'white':
            update_flag = True
            state_publisher.publish(11)
        elif color == 'black':
            state_publisher.publish(14)
        
    def CloseChessboardWindow(self, data):
        if data:
            self.ChessboardWindow.close()

    def SelectButton(self, data):
        if data:
            if self.StartGamePushButton.isEnabled():
                self.ChessboardWindow()
            else:
                rospy.loginfo('TIAGo preparation has not been completed yet.')


###################################################################


class ChessboardWhiteWindow(QtWidgets.QWidget, Ui_Chessboard_White):
    #Signals initialization
    clicked = pyqtSignal(object)
    promotion = QtCore.pyqtSignal(int)
    close_promotion = QtCore.pyqtSignal(int)
    manual_mode = QtCore.pyqtSignal(int)
    back_to_game = QtCore.pyqtSignal(int)
    global update_flag
    
    def __init__(self):
        super(ChessboardWhiteWindow, self).__init__()
        self.app = MyApplication(sys.argv)
        self.app_focus = self.app.focusWidget() 
        self.setupUi(self, self.app_focus)

        #Subscribers initialization
        rospy.Subscriber("/cursor_step_eleven", Point, self.MoveHighlight)
        rospy.Subscriber("/button_eleven", Bool, self.SelectPiece)
        rospy.Subscriber("/promotion_happened", String, self.PromotionHappened)
        rospy.Subscriber("/promoted_piece", String, self.Promotion)
        rospy.Subscriber("/opponent_move_start_square", String, self.OpponentMoveStartSquare)
        rospy.Subscriber("/opponent_move_end_square", String, self.OpponentMoveEndSquare)
        rospy.Subscriber("/en_passant_square", String, self.EnPassant)
        rospy.Subscriber("/castle_square", String, self.Castle)
        rospy.Subscriber('/state', Int16, self.UpdateLiveSituation)

        #Publishers initialization
        self.start_square_move_publisher = rospy.Publisher('/start_square', String, queue_size = 10)
        self.end_square_move_publisher = rospy.Publisher('/end_square', String, queue_size = 10)

        #Variables initializiation
        self.promoted_piece_msg = 'none'
        self.opponent_promotion_happened = False
        self.promotion_happened = False

        #Import configurations to map the chessboard squares
        self.dict_squares = cfg_w.chessboard_numbers
        self.pieces_coordinates = cfg_w.pieces_coordinates
        self.chessboard_colors = cfg_w.chessboard_colors
        self.chessboard_matrix = cfg_w.matrix
        self.array = [self.a1, self.b1, self.c1, self.d1, self.e1, self.f1, self.g1, self.h1, self.a2, self.b2, self.c2, self.d2,
        self.e2, self.f2, self.g2, self.h2, self.a3, self.b3, self.c3, self.d3, self.e3, self.f3, self.g3, self.h3, self.a4, self.b4,
        self.c4, self.d4, self.e4, self.f4, self.g4, self.h4, self.a5, self.b5, self.c5, self.d5, self.e5, self.f5, self.g5, self.h5,
        self.a6, self.b6, self.c6, self.d6, self.e6, self.f6, self.g6, self.h6, self.a7, self.b7, self.c7, self.d7, self.e7, self.f7,
        self.g7, self.h7, self.a8, self.b8, self.c8, self.d8, self.e8, self.f8, self.g8, self.h8]

        self.icons_array_black = [self.icon_Wpawn_b, self.icon_Bpawn_b, self.icon_Wrook_b, self.icon_Brook_b, self.icon_Wknight_b,
        self.icon_Bknight_b, self.icon_Wbishop_b, self.icon_Bbishop_b, self.icon_Wqueen_b, self.icon_Bqueen_b, self.icon_Wking_b, self.icon_Bking_b]
        self.icons_array_white = [self.icon_Wpawn_w, self.icon_Bpawn_w, self.icon_Wrook_w, self.icon_Brook_w, self.icon_Wknight_w,
        self.icon_Bknight_w, self.icon_Wbishop_w, self.icon_Bbishop_w, self.icon_Wqueen_w, self.icon_Bqueen_w, self.icon_Wking_w, self.icon_Bking_w]
        self.icons_array_yellow = [self.icon_Wpawn_y, self.icon_Bpawn_y, self.icon_Wrook_y, self.icon_Brook_y, self.icon_Wknight_y,
        self.icon_Bknight_y, self.icon_Wbishop_y, self.icon_Bbishop_y, self.icon_Wqueen_y, self.icon_Bqueen_y, self.icon_Wking_y, self.icon_Bking_y]

        #Selection flag initializiation
        self.selection_flag = False
        self.yellow_selection_flag = False

        #Focus changed flag initialization
        self.already_changed = False

        #Opponent en-passant happened flag
        self.en_passant_happened = False

        #Save the live chessboard situation yaml file directory
        self.dir_live_chessboard_situation = PLAYCHESS_PKG_DIR + "/scripts/live_chessboard_situation.yaml"

        self.pieces_coordinates = cfg.pieces_coordinates
        self.squares_to_index = cfg.squares_to_index_white
        self.scene = moveit_commander.PlanningSceneInterface() #The interface with the world surrounding the robot

        #Connect signals and slots.
        self.a1.clicked.connect(lambda: self.click_and_focus('a1'))
        self.b1.clicked.connect(lambda: self.click_and_focus('b1'))
        self.c1.clicked.connect(lambda: self.click_and_focus('c1'))
        self.d1.clicked.connect(lambda: self.click_and_focus('d1'))
        self.e1.clicked.connect(lambda: self.click_and_focus('e1'))
        self.f1.clicked.connect(lambda: self.click_and_focus('f1'))
        self.g1.clicked.connect(lambda: self.click_and_focus('g1'))
        self.h1.clicked.connect(lambda: self.click_and_focus('h1'))
        self.a2.clicked.connect(lambda: self.click_and_focus('a2'))
        self.b2.clicked.connect(lambda: self.click_and_focus('b2'))
        self.c2.clicked.connect(lambda: self.click_and_focus('c2'))
        self.d2.clicked.connect(lambda: self.click_and_focus('d2'))
        self.e2.clicked.connect(lambda: self.click_and_focus('e2'))
        self.f2.clicked.connect(lambda: self.click_and_focus('f2'))
        self.g2.clicked.connect(lambda: self.click_and_focus('g2'))
        self.h2.clicked.connect(lambda: self.click_and_focus('h2'))
        self.a3.clicked.connect(lambda: self.click_and_focus('a3'))
        self.b3.clicked.connect(lambda: self.click_and_focus('b3'))
        self.c3.clicked.connect(lambda: self.click_and_focus('c3'))
        self.d3.clicked.connect(lambda: self.click_and_focus('d3'))
        self.e3.clicked.connect(lambda: self.click_and_focus('e3'))
        self.f3.clicked.connect(lambda: self.click_and_focus('f3'))
        self.g3.clicked.connect(lambda: self.click_and_focus('g3'))
        self.h3.clicked.connect(lambda: self.click_and_focus('h3'))
        self.a4.clicked.connect(lambda: self.click_and_focus('a4'))
        self.b4.clicked.connect(lambda: self.click_and_focus('b4'))
        self.c4.clicked.connect(lambda: self.click_and_focus('c4'))
        self.d4.clicked.connect(lambda: self.click_and_focus('d4'))
        self.e4.clicked.connect(lambda: self.click_and_focus('e4'))
        self.f4.clicked.connect(lambda: self.click_and_focus('f4'))
        self.g4.clicked.connect(lambda: self.click_and_focus('g4'))
        self.h4.clicked.connect(lambda: self.click_and_focus('h4'))
        self.a5.clicked.connect(lambda: self.click_and_focus('a5'))
        self.b5.clicked.connect(lambda: self.click_and_focus('b5'))
        self.c5.clicked.connect(lambda: self.click_and_focus('c5'))
        self.d5.clicked.connect(lambda: self.click_and_focus('d5'))
        self.e5.clicked.connect(lambda: self.click_and_focus('e5'))
        self.f5.clicked.connect(lambda: self.click_and_focus('f5'))
        self.g5.clicked.connect(lambda: self.click_and_focus('g5'))
        self.h5.clicked.connect(lambda: self.click_and_focus('h5'))
        self.a6.clicked.connect(lambda: self.click_and_focus('a6'))
        self.b6.clicked.connect(lambda: self.click_and_focus('b6'))
        self.c6.clicked.connect(lambda: self.click_and_focus('c6'))
        self.d6.clicked.connect(lambda: self.click_and_focus('d6'))
        self.e6.clicked.connect(lambda: self.click_and_focus('e6'))
        self.f6.clicked.connect(lambda: self.click_and_focus('f6'))
        self.g6.clicked.connect(lambda: self.click_and_focus('g6'))
        self.h6.clicked.connect(lambda: self.click_and_focus('h6'))
        self.a7.clicked.connect(lambda: self.click_and_focus('a7'))
        self.b7.clicked.connect(lambda: self.click_and_focus('b7'))
        self.c7.clicked.connect(lambda: self.click_and_focus('c7'))
        self.d7.clicked.connect(lambda: self.click_and_focus('d7'))
        self.e7.clicked.connect(lambda: self.click_and_focus('e7'))
        self.f7.clicked.connect(lambda: self.click_and_focus('f7'))
        self.g7.clicked.connect(lambda: self.click_and_focus('g7'))
        self.h7.clicked.connect(lambda: self.click_and_focus('h7'))
        self.a8.clicked.connect(lambda: self.click_and_focus('a8'))
        self.b8.clicked.connect(lambda: self.click_and_focus('b8'))
        self.c8.clicked.connect(lambda: self.click_and_focus('c8'))
        self.d8.clicked.connect(lambda: self.click_and_focus('d8'))
        self.e8.clicked.connect(lambda: self.click_and_focus('e8'))
        self.f8.clicked.connect(lambda: self.click_and_focus('f8'))
        self.g8.clicked.connect(lambda: self.click_and_focus('g8'))
        self.h8.clicked.connect(lambda: self.click_and_focus('h8'))

        self.ManualModePushbutton.clicked.connect(self.ManualMode)
        self.PlayGameModePushbutton.clicked.connect(self.BackToGame)

    def MoveHighlight(self, data):
        #Whenever a joystick movement is detected, a motion of the highlights pushbutton on the GUI is performed
        #If a movement in both the directions is captured, ignore it.
        if data.x != 0 and data.y != 0:
                pass

        #If a movement in the x direction, understand its direction and perform the corresponding movement on the GUI.
        if data.x != 0 and data.y == 0:
                print('MOVE HIGHLIGHT')
                for row in self.chessboard_matrix:
                        if self.current_focus in row: #str(focus) in row:
                                row_index = self.chessboard_matrix.index(row)
                                col_index = row.index(self.current_focus)

                if data.x == 1:
                        #Highlight the pushbutton to the right
                        col_index += 1
                        if col_index == 8:
                                if row_index == 7:
                                        rospy.logwarn('The square does not exist')
                                else:
                                        col_index = 0
                                        row_index += 1 
                elif data.x == 2:
                        #Highlight the pushbutton to the left
                        col_index -= 1
                        if col_index == -1:
                                if row_index == 0:
                                        rospy.logwarn('The square does not exist')
                                else:
                                        col_index = 7
                                        row_index -= 1

                #Remove the highlight from the previous focused square, if the focus has already been changed at least one time and if there is not a selected square.
                if (self.already_changed and not self.selection_flag) or (self.already_changed and self.selection_flag and self.current_focus != self.current_red_icon):
                    button_index = self.dict_squares[self.current_focus]
                    button = self.array[button_index]
                    
                    #Read if the focused square is occupied by one piece or if it's empty
                    with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                            live_situation = yaml.load(live_file.read())

                    if live_situation[self.current_focus][0] == 'none':
                        square_object = 'none'
                    else:
                        square_object = self.pieces_coordinates[live_situation[self.current_focus][0]][1]['name']
                        square_object_color = live_situation[self.current_focus][1]

                    if self.yellow_selection_flag: #If the square is yellow:
                        if self.self.piece_to_move == 'pawn':
                            if self.color_to_move == 'white':
                                button.setIcon(self.icon_Wpawn_y)
                            elif self.color_to_move == 'black':
                                button.setIcon(self.icon_Bpawn_y)
                        elif self.self.piece_to_move == 'rook':
                            if self.color_to_move == 'white':
                                button.setIcon(self.icon_Wrook_y)
                            elif self.color_to_move == 'black':
                                button.setIcon(self.icon_Brook_y)
                        elif self.self.piece_to_move == 'knight':
                            if self.color_to_move == 'white':
                                button.setIcon(self.icon_Wknight_y)
                            elif self.color_to_move == 'black':
                                button.setIcon(self.icon_Bknight_y)
                        elif self.self.piece_to_move == 'bishop':
                            if self.color_to_move == 'white':
                                button.setIcon(self.icon_Wbishop_y)
                            elif self.color_to_move == 'black':
                                button.setIcon(self.icon_Bbishop_y)
                        elif self.self.piece_to_move == 'queen':
                            if self.color_to_move == 'white':
                                button.setIcon(self.icon_Wqueen_y)
                            elif self.color_to_move == 'black':
                                button.setIcon(self.icon_Bqueen_y)
                        elif self.self.piece_to_move == 'king':
                            if self.color_to_move == 'white':
                                button.setIcon(self.icon_Wking_y)
                            elif self.color_to_move == 'black':
                                button.setIcon(self.icon_Bking_y)

                    elif self.chessboard_colors[self.current_focus] == 'white':
                        if square_object == 'pawn':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wpawn_w)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bpawn_w)
                        elif square_object == 'rook':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wrook_w)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Brook_w)
                        elif square_object == 'knight':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wknight_w)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bknight_w)
                        elif square_object == 'bishop':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wbishop_w)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bbishop_w)
                        elif square_object == 'queen':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wqueen_w)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bqueen_w)
                        elif square_object == 'king':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wking_w)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bking_w)
                        elif square_object == 'none':
                            button.setIcon(QtGui.QIcon())
                    elif self.chessboard_colors[self.current_focus] == 'black':
                        if square_object == 'pawn':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wpawn_b)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bpawn_b)
                        elif square_object == 'rook':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wrook_b)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Brook_b)
                        elif square_object == 'knight':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wknight_b)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bknight_b)
                        elif square_object == 'bishop':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wbishop_b)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bbishop_b)
                        elif square_object == 'queen':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wqueen_b)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bqueen_b)
                        elif square_object == 'king':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wking_b)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bking_b)
                        elif square_object == 'none':
                            button.setIcon(self.icon_grey)

                #Highlight the new pushbutton
                if row_index >= 0 and row_index <= 7 and col_index >= 0 and col_index <= 7:
                        self.current_focus = self.chessboard_matrix[row_index][col_index]
                        new_button_index = self.dict_squares[self.current_focus]
                        new_button = self.array[new_button_index]
                        new_button.setFocus()

                        #Read if the focused square is occupied by one piece or if it's empty
                        with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                                live_situation = yaml.load(live_file.read())

                        if live_situation[self.current_focus][0] == 'none':
                            square_object = 'none'
                        else:
                            square_object = self.pieces_coordinates[live_situation[self.current_focus][0]][1]['name']
                            square_object_color = live_situation[self.current_focus][1]

                        if self.chessboard_colors[self.current_focus] == 'white':
                            if square_object == 'pawn':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wpawn_w_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bpawn_w_frame)
                            elif square_object == 'rook':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wrook_w_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Brook_w_frame)
                            elif square_object == 'knight':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wknight_w_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bknight_w_frame)
                            elif square_object == 'bishop':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wbishop_w_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bbishop_w_frame)
                            elif square_object == 'queen':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wqueen_w_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bqueen_w_frame)
                            elif square_object == 'king':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wking_w_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bking_w_frame)
                            elif square_object == 'none':
                                new_button.setIcon(self.icon_white_frame)
                        elif self.chessboard_colors[self.current_focus] == 'black':
                            if square_object == 'pawn':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wpawn_b_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bpawn_b_frame)
                            elif square_object == 'rook':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wrook_b_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Brook_b_frame)
                            elif square_object == 'knight':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wknight_b_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bknight_b_frame)
                            elif square_object == 'bishop':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wbishop_b_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bbishop_b_frame)
                            elif square_object == 'queen':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wqueen_b_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bqueen_b_frame)
                            elif square_object == 'king':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wking_b_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bking_b_frame)
                            elif square_object == 'none':
                                new_button.setIcon(self.icon_grey_frame)

                        print('NEW FOCUS: ' + str(self.current_focus))
                        self.already_changed = True
                else:
                        rospy.logwarn('The square does not exist')

        #If a movement in the y direction, understand its direction and perform the corresponding movement on the GUI.
        if data.y != 0 and data.x == 0:
                #Save the row and the column index of the widget that currently has focus
                for row in self.chessboard_matrix:
                        if self.current_focus in row:
                                row_index = self.chessboard_matrix.index(row)
                                col_index = row.index(self.current_focus)
                if data.y == 1:
                        #Highlight the upper pushbutton
                        row_index += 1
                        if row_index == 8:
                                if col_index == 7:
                                        rospy.logwarn('The square does not exist')
                                else:
                                        row_index = 0
                                        col_index += 1
                elif data.y == 2:
                        #Highlight the lower pushbutton
                        row_index -= 1
                        if row_index == -1:
                                if col_index == 0:
                                        rospy.logwarn('The square does not exist')
                                else:
                                        row_index = 7
                                        col_index -= 1

                #Remove the highlight from the previous focused square, if the focus has already been changed at least one time.
                if (self.already_changed and not self.selection_flag) or (self.already_changed and self.selection_flag and self.current_focus != self.current_red_icon):
                    button_index = self.dict_squares[self.current_focus]
                    button_index = self.dict_squares[self.current_focus]
                    button = self.array[button_index]
                    
                    #Read if the focused square is occupied by one piece or if it's empty
                    with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                            live_situation = yaml.load(live_file.read())

                    if live_situation[self.current_focus][0] == 'none':
                        square_object = 'none'
                    else:
                        square_object = self.pieces_coordinates[live_situation[self.current_focus][0]][1]['name']
                        square_object_color = live_situation[self.current_focus][1]

                    if self.chessboard_colors[self.current_focus] == 'white':
                        if square_object == 'pawn':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wpawn_w)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bpawn_w)
                        elif square_object == 'rook':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wrook_w)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Brook_w)
                        elif square_object == 'knight':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wknight_w)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bknight_w)
                        elif square_object == 'bishop':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wbishop_w)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bbishop_w)
                        elif square_object == 'queen':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wqueen_w)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bqueen_w)
                        elif square_object == 'king':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wking_w)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bking_w)
                        elif square_object == 'none':
                            button.setIcon(QtGui.QIcon())
                    elif self.chessboard_colors[self.current_focus] == 'black':
                        if square_object == 'pawn':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wpawn_b)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bpawn_b)
                        elif square_object == 'rook':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wrook_b)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Brook_b)
                        elif square_object == 'knight':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wknight_b)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bknight_b)
                        elif square_object == 'bishop':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wbishop_b)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bbishop_b)
                        elif square_object == 'queen':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wqueen_b)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bqueen_b)
                        elif square_object == 'king':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wking_b)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bking_b)
                        elif square_object == 'none':
                            button.setIcon(self.icon_grey)

                #Highlight the new pushbutton
                if row_index >= 0 and row_index <= 7 and col_index >= 0 and col_index <= 7:
                        self.current_focus = self.chessboard_matrix[row_index][col_index]
                        new_button_index = self.dict_squares[self.current_focus]
                        new_button = self.array[new_button_index]
                        new_button.setFocus()

                        #Read if the focused square is occupied by one piece or if it's empty
                        with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                                live_situation = yaml.load(live_file.read())

                        if live_situation[self.current_focus][0] == 'none':
                            square_object = 'none'
                        else:
                            square_object = self.pieces_coordinates[live_situation[self.current_focus][0]][1]['name']
                            square_object_color = live_situation[self.current_focus][1]

                        if self.chessboard_colors[self.current_focus] == 'white':
                            if square_object == 'pawn':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wpawn_w_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bpawn_w_frame)
                            elif square_object == 'rook':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wrook_w_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Brook_w_frame)
                            elif square_object == 'knight':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wknight_w_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bknight_w_frame)
                            elif square_object == 'bishop':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wbishop_w_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bbishop_w_frame)
                            elif square_object == 'queen':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wqueen_w_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bqueen_w_frame)
                            elif square_object == 'king':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wking_w_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bking_w_frame)
                            elif square_object == 'none':
                                new_button.setIcon(self.icon_white_frame)
                        elif self.chessboard_colors[self.current_focus] == 'black':
                            if square_object == 'pawn':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wpawn_b_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bpawn_b_frame)
                            elif square_object == 'rook':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wrook_b_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Brook_b_frame)
                            elif square_object == 'knight':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wknight_b_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bknight_b_frame)
                            elif square_object == 'bishop':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wbishop_b_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bbishop_b_frame)
                            elif square_object == 'queen':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wqueen_b_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bqueen_b_frame)
                            elif square_object == 'king':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wking_b_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bking_b_frame)
                            elif square_object == 'none':
                                new_button.setIcon(self.icon_grey_frame)

                        print('NEW FOCUS: ' + str(self.current_focus))
                        self.already_changed = True
                else:
                        rospy.logwarn('The square does not exist')

    def SelectPiece(self, data):
        global square_object
        global square_captured_piece
        if data:
            if state == 11: #If we are in game mode.
                #Change the current_focus variable.
                #self.current_focus = data
                print('CURRENT FOCUS: ' + str(self.current_focus))

                #If no other square is selected, select the square and change the color of it in red.
                if not self.selection_flag and not self.yellow_selection_flag:
                        #Read if the selected square is occupied by one piece or if it's empty.
                        with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                                live_situation = yaml.load(live_file.read())
                        square_object = live_situation[self.current_focus][0]
                        if square_object == 'none':
                                rospy.logwarn('There is no piece to move in the selected square')
                                self.selection_flag = not self.selection_flag
                        else:
                            self.color_to_move = live_situation[self.current_focus][1]
                            if self.pieces_coordinates[square_object][1]['name'] == 'pawn':
                                if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Wpawn)
                                elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Bpawn)
                            elif self.pieces_coordinates[square_object][1]['name'] == 'rook':
                                if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Wrook)
                                elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Brook)
                            elif self.pieces_coordinates[square_object][1]['name'] == 'knight':
                                if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Wknight)
                                elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Bknight)
                            elif self.pieces_coordinates[square_object][1]['name'] == 'bishop':
                                if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Wbishop)
                                elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Bbishop)
                            elif self.pieces_coordinates[square_object][1]['name'] == 'queen':
                                if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Wqueen)
                                elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Bqueen)
                            elif self.pieces_coordinates[square_object][1]['name'] == 'king':
                                if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Wking)
                                elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Bking)
                            self.current_red_icon = self.current_focus #Save the icon that currently is highlighted.
                        self.selection_flag = not self.selection_flag

                elif self.selection_flag and not self.yellow_selection_flag:
                        #Read which piece is in the currently highlighted square
                        with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                                live_situation = yaml.load(live_file.read())
                        self.piece_to_move = self.pieces_coordinates[live_situation[self.current_red_icon][0]][1]['name']
                        piece_to_capture = live_situation[self.current_focus][0]
                        self.color_to_move = live_situation[self.current_red_icon][1]
                        
                        #Save the color of the destination square and of the starting square
                        destination_color = self.chessboard_colors[self.current_focus]
                        starting_color = self.chessboard_colors[self.current_red_icon]

                        #Initialize variables to keep trak of the intentions of the player
                        self.icon_changed_1 = 'none'
                        self.old_icon_color_1 = 'none'
                        self.icon_changed_2 = 'none'
                        self.old_icon_color_2 = 'none'
                        self.icon_changed_3 = 'none'
                        self.old_icon_color_3 = 'none'
                        self.old_icon_piece_3 = 'none'
                        self.old_icon_piece_color_3 = 'none'
                        self.icon_changed_4 = 'none'
                        self.old_icon_color_4 = 'none'
                        self.old_icon_piece_4 = 'none'
                        self.old_icon_piece_color_4 = 'none'

                        #Change the destination icon and the starting square icon (destination color in yellow, cause I need to implement move confirmation)
                        if self.piece_to_move == 'pawn':
                            if self.color_to_move == 'white':
                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[0])
                                self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                            elif self.color_to_move == 'black':
                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[1])
                                self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                            #Implement the en-passant special move (only for white. For black will be implemented in the other chessboard GUI script):
                            count = 0
                            for i in self.chessboard_matrix:
                                if self.current_red_icon in i:
                                    prev_index = i.index(self.current_red_icon)
                                    capture_row = count
                                if self.current_focus in i:
                                    new_index = i.index(self.current_focus)
                                count += 1
                            if prev_index != new_index and piece_to_capture == 'none': #If a capture has been performed by the pawn and there was no piece in the ending square, en-passant happened.
                                #self.en_passant = True
                                square_captured_piece = self.chessboard_matrix[capture_row][new_index]
                                if self.chessboard_colors[square_captured_piece] == 'black':
                                    self.array[self.dict_squares[square_captured_piece]].setIcon(self.icon_grey)
                                    self.icon_changed_3 = self.array[self.dict_squares[square_captured_piece]]
                                    self.old_icon_color_3 = 'black'
                                    self.old_icon_piece_3 = 'pawn'
                                    self.old_icon_piece_color_3 = 'black' #Cause I know I'm playing with white
                                elif self.chessboard_colors[square_captured_piece] == 'white':
                                    self.array[self.dict_squares[square_captured_piece]].setIcon(QtGui.QIcon())
                                    self.icon_changed_3 = self.array[self.dict_squares[square_captured_piece]]
                                    self.old_icon_color_3 = 'white'
                                    self.old_icon_piece_3 = 'pawn'
                                    self.old_icon_piece_color_3 = 'black' #Cause I know I'm playing with white
                                #en_passant_square_to_move_publisher.publish(square_captured_piece)

                        elif self.piece_to_move == 'rook':
                                if self.color_to_move == 'white':
                                        self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[2])
                                        self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                        self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                                elif self.color_to_move == 'black':
                                        self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[3])
                                        self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                        self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                        elif self.piece_to_move == 'knight':
                                if self.color_to_move == 'white':
                                        self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[4])
                                        self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                        self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                                elif self.color_to_move == 'black':
                                        self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[5])
                                        self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                        self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                        elif self.piece_to_move == 'bishop':
                                if self.color_to_move == 'white':
                                        self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[6])
                                        self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                        self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                                elif self.color_to_move == 'black':
                                        self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[7])
                                        self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                        self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                        elif self.piece_to_move == 'queen':
                                if self.color_to_move == 'white':
                                        self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[8])
                                        self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                        self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                                elif self.color_to_move == 'black':
                                        self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[9])
                                        self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                        self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                        elif self.piece_to_move == 'king':
                                if self.color_to_move == 'white':
                                        self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[10])
                                        self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                        self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                                elif self.color_to_move == 'black':
                                        self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[11])
                                        self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                        self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                                #Implement the castle special move (only for white. For black it will be implemented in the ther chessboard GUI script):
                                if (self.current_focus == 'g1' or self.current_focus == 'c1') and (self.current_red_icon == 'e1'):
                                        if self.current_focus == 'g1': #Short castle
                                                #self.short_castle = True
                                                self.f1.setIcon(self.icons_array_yellow[2])
                                                self.icon_changed_2 = self.array[self.dict_squares['f1']]
                                                self.old_icon_color_2 = self.chessboard_colors['f1']
                                                self.h1.setIcon(QtGui.QIcon())
                                        elif self.current_focus == 'c1': #Long castle 
                                                #self.long_castle = True
                                                self.d1.setIcon(self.icons_array_yellow[2])
                                                self.icon_changed_2 = self.array[self.dict_squares['d1']]
                                                self.old_icon_color_2 = self.chessboard_colors['d1']
                                                self.a1.setIcon(self.icon_grey)
                        if starting_color == 'white':
                                self.array[self.dict_squares[self.current_red_icon]].setIcon(QtGui.QIcon())
                                self.icon_changed_4 = self.array[self.dict_squares[self.current_red_icon]]
                                self.old_icon_color_4 = 'white'
                                self.old_icon_piece_4 = self.piece_to_move
                                self.old_icon_piece_color_4 = self.color_to_move
                        elif starting_color == 'black':
                                self.array[self.dict_squares[self.current_red_icon]].setIcon(self.icon_grey)
                                self.icon_changed_4 = self.array[self.dict_squares[self.current_red_icon]]
                                self.old_icon_color_4 = 'black'
                                self.old_icon_piece_4 = self.piece_to_move
                                self.old_icon_piece_color_4 = self.color_to_move

                        self.yellow_selection_flag = not self.yellow_selection_flag
                        self.selection_flag = not self.selection_flag
                        self.intention_icon = self.current_focus #Save the square that the player is intentioned to play to.

                elif not self.selection_flag and self.yellow_selection_flag:
                    if self.current_focus == self.intention_icon: #If the player clicked on the same square again, the move is confirmed.
                        #Read which piece is in the currently highlighted square
                        with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                                live_situation = yaml.load(live_file.read())
                        self.piece_to_move = self.pieces_coordinates[live_situation[self.current_red_icon][0]][1]['name']
                        piece_to_capture = live_situation[self.current_focus][0]
                        self.color_to_move = live_situation[self.current_red_icon][1]
                        
                        #Save the color of the destination square and of the starting square
                        destination_color = self.chessboard_colors[self.current_focus]
                        starting_color = self.chessboard_colors[self.current_red_icon]

                        #Initialize the promoted_piece, the en_passant and the castle variables
                        #promoted_piece = 'none'
                        self.en_passant = False
                        self.short_castle = False
                        self.long_castle = False

                        if destination_color == 'black':
                                if self.piece_to_move == 'pawn':
                                        #Implement promotion (only for white. For black will be implemented in the other chessboard GUI script)
                                        count = 0
                                        for r in self.chessboard_matrix:
                                            if self.current_focus in r:
                                                    dest_row = count
                                                    break
                                            count += 1
                                        if dest_row == 7: #Promotion is happening
                                                state_publisher.publish(12) #Go to the state of promoted piece question
                                                #Make the user choose the promoted piece
                                                if self.color_to_move == 'white':
                                                    self.promotion.emit(1)
                                                elif self.color_to_move == 'black':
                                                    self.promotion.emit(2)                            
                                        else:
                                                if self.color_to_move == 'white':
                                                        self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[0])
                                                elif self.color_to_move == 'black':
                                                        self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[1])
                                                #Implement the en-passant special move (only for white. For black will be implemented in the other chessboard GUI script):
                                                count = 0
                                                for i in self.chessboard_matrix:
                                                        if self.current_red_icon in i:
                                                                prev_index = i.index(self.current_red_icon)
                                                                capture_row = count
                                                        if self.current_focus in i:
                                                                new_index = i.index(self.current_focus)
                                                        count += 1
                                                if prev_index != new_index and piece_to_capture == 'none': #If a capture has been performed by the pawn and there was no piece in the ending square, en-passant happened.
                                                        self.en_passant = True
                                                        square_captured_piece = self.chessboard_matrix[capture_row][new_index]
                                                        en_passant_square_to_move_publisher.publish(square_captured_piece)
                                elif self.piece_to_move == 'rook':
                                        if self.color_to_move == 'white':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[2])
                                        elif self.color_to_move == 'black':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[3])
                                elif self.piece_to_move == 'knight':
                                        if self.color_to_move == 'white':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[4])
                                        elif self.color_to_move == 'black':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[5])
                                elif self.piece_to_move == 'bishop':
                                        if self.color_to_move == 'white':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[6])
                                        elif self.color_to_move == 'black':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[7])
                                elif self.piece_to_move == 'queen':
                                        if self.color_to_move == 'white':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[8])
                                        elif self.color_to_move == 'black':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[9])
                                elif self.piece_to_move == 'king':
                                        if self.color_to_move == 'white':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[10])
                                        elif self.color_to_move == 'black':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[11])
                                        #Implement the castle special move (only for white. For black it will be implemented in the ther chessboard GUI script):
                                        if (self.current_focus == 'g1' or self.current_focus == 'c1') and (self.current_red_icon == 'e1'):
                                                if self.current_focus == 'g1': #Short castle
                                                        self.short_castle = True
                                                        self.f1.setIcon(self.icons_array_white[2])
                                                        self.h1.setIcon(QtGui.QIcon())
                                                elif self.current_focus == 'c1': #Long castle 
                                                        self.long_castle = True
                                                        self.d1.setIcon(self.icons_array_white[2])
                                                        self.a1.setIcon(self.icon_grey)
                        elif destination_color == 'white':
                                if self.piece_to_move == 'pawn':
                                        #Implement promotion
                                        count = 0
                                        for r in self.chessboard_matrix:
                                            if self.current_focus in r:
                                                    dest_row = count
                                                    break
                                            count += 1
                                        if dest_row == 7: #Promotion is happening
                                                state_publisher.publish(12) #Go to the state of promoted piece question
                                                #Make the user choose the promoted piece
                                                if self.color_to_move == 'white':
                                                    self.promotion.emit(1)
                                                elif self.color_to_move == 'black':
                                                    self.promotion.emit(2)                                       
                                        else:
                                                if self.color_to_move == 'white':
                                                        self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[0])
                                                elif self.color_to_move == 'black':
                                                        self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[1])
                                                #Implement the en-passant special move:
                                                count = 0
                                                for i in self.chessboard_matrix:
                                                        count += 1
                                                        if self.current_red_icon in i:
                                                                prev_index = i.index(self.current_red_icon)
                                                                capture_row = count
                                                        if self.current_focus in i:
                                                                new_index = i.index(self.current_focus)
                                                if prev_index != new_index and piece_to_capture == 'none': #If a capture has been performed by the pawn and there was no piece in the ending square, en-passant happened.
                                                        self.en_passant = True
                                                        square_captured_piece = self.chessboard_matrix[capture_row][new_index]
                                                        en_passant_square_to_move_publisher.publish(square_captured_piece)

                                elif self.piece_to_move == 'rook':
                                        if self.color_to_move == 'white':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[2])
                                        elif self.color_to_move == 'black':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[3])
                                elif self.piece_to_move == 'knight':
                                        if self.color_to_move == 'white':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[4])
                                        elif self.color_to_move == 'black':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[5])
                                elif self.piece_to_move == 'bishop':
                                        if self.color_to_move == 'white':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[6])
                                        elif self.color_to_move == 'black':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[7])
                                elif self.piece_to_move == 'queen':
                                        if self.color_to_move == 'white':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[8])
                                        elif self.color_to_move == 'black':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[9])
                                elif self.piece_to_move == 'king':
                                        if self.color_to_move == 'white':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[10])
                                        elif self.color_to_move == 'black':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[11])
                        #Send a message containing the start square and the end square, so that TIAGo can execute it.
                        if self.short_castle:
                            self.start_square_move_publisher.publish('short')
                            self.end_square_move_publisher.publish('short')
                        elif self.long_castle:
                            self.start_square_move_publisher.publish('long')
                            self.end_square_move_publisher.publish('long')
                        else:
                            self.start_square_move_publisher.publish(self.current_red_icon)
                            self.end_square_move_publisher.publish(self.current_focus)
                            #self.promoted_piece_move_publisher.publish(promoted_piece)

                        #Change operational status.
                    if state != 12:
                        state_publisher.publish(13)
                        update_flag = True

                    else: #If the player is not confirming the move (by clicking in another square), get back to the state where he has to chosen the move.
                        with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                            live_situation = yaml.load(live_file.read())
                        if self.icon_changed_1 != 'none':
                            pezzo = self.pieces_coordinates[live_situation[self.intention_icon][0]][1]['name'] ####
                            pezzo_color = live_situation[self.intention_icon][1]
                            if self.old_icon_color_1 == 'white':
                                if pezzo == 'pawn':
                                    if pezzo_color == 'white':
                                        self.icon_changed_1.setIcon(self.icons_array_white[0])
                                    elif pezzo_color == 'black':
                                        self.icon_changed_1.setIcon(self.icons_array_white[1])
                                elif pezzo == 'bishop':
                                    if pezzo_color == 'white':
                                        self.icon_changed_1.setIcon(self.icons_array_white[6])
                                    elif pezzo_color == 'black':
                                        self.icon_changed_1.setIcon(self.icons_array_white[7])
                                elif pezzo == 'knight':
                                    if pezzo_color == 'white':
                                        self.icon_changed_1.setIcon(self.icons_array_white[4])
                                    elif pezzo_color == 'black':
                                        self.icon_changed_1.setIcon(self.icons_array_white[5])
                                elif pezzo == 'rook':
                                    if pezzo_color == 'white':
                                        self.icon_changed_1.setIcon(self.icons_array_white[2])
                                    elif pezzo_color == 'black':
                                        self.icon_changed_1.setIcon(self.icons_array_white[3])
                                elif pezzo == 'king':
                                    if pezzo_color == 'white':
                                        self.icon_changed_1.setIcon(self.icons_array_white[10])
                                    elif pezzo_color == 'black':
                                        self.icon_changed_1.setIcon(self.icons_array_white[11])
                                elif pezzo == 'queen':
                                    if pezzo_color == 'white':
                                        self.icon_changed_1.setIcon(self.icons_array_white[8])
                                    elif pezzo_color == 'black':
                                        self.icon_changed_1.setIcon(self.icons_array_white[9])
                                else:
                                    self.icon_changed_1.setIcon(QtGui.QIcon())
                            elif self.old_icon_color_1 == 'black':
                                if pezzo == 'pawn':
                                    if pezzo_color == 'white':
                                        self.icon_changed_1.setIcon(self.icons_array_black[0])
                                    elif pezzo_color == 'black':
                                        self.icon_changed_1.setIcon(self.icons_array_black[1])
                                elif pezzo == 'bishop':
                                    if pezzo_color == 'white':
                                        self.icon_changed_1.setIcon(self.icons_array_black[6])
                                    elif pezzo_color == 'black':
                                        self.icon_changed_1.setIcon(self.icons_array_black[7])
                                elif pezzo == 'knight':
                                    if pezzo_color == 'white':
                                        self.icon_changed_1.setIcon(self.icons_array_black[4])
                                    elif pezzo_color == 'black':
                                        self.icon_changed_1.setIcon(self.icons_array_black[5])
                                elif pezzo == 'rook':
                                    if pezzo_color == 'white':
                                        self.icon_changed_1.setIcon(self.icons_array_black[2])
                                    elif pezzo_color == 'black':
                                        self.icon_changed_1.setIcon(self.icons_array_black[3])
                                elif pezzo == 'king':
                                    if pezzo_color == 'white':
                                        self.icon_changed_1.setIcon(self.icons_array_black[10])
                                    elif pezzo_color == 'black':
                                        self.icon_changed_1.setIcon(self.icons_array_black[11])
                                elif pezzo == 'queen':
                                    if pezzo_color == 'white':
                                        self.icon_changed_1.setIcon(self.icons_array_black[8])
                                    elif pezzo_color == 'black':
                                        self.icon_changed_1.setIcon(self.icons_array_black[9])
                                else:
                                    self.icon_changed_1.setIcon(self.icon_grey)

                        if self.icon_changed_3 != 'none':
                            if self.old_icon_color_3 == 'white':
                                if self.old_icon_piece_3 == 'pawn':
                                    if self.old_icon_piece_color_3 == 'white':
                                        self.icon_changed_3.setIcon(self.icons_array_white[0])
                                    elif self.old_icon_piece_color_3 == 'black':
                                        self.icon_changed_3.setIcon(self.icons_array_white[1])
                                elif self.old_icon_piece_3 == 'rook':
                                    if self.old_icon_piece_color_3 == 'white':
                                        self.icon_changed_3.setIcon(self.icons_array_white[2])
                                    elif self.old_icon_piece_color_3 == 'black':
                                        self.icon_changed_3.setIcon(self.icons_array_white[3])
                                elif self.old_icon_piece_3 == 'knight':
                                    if self.old_icon_piece_color_3 == 'white':
                                        self.icon_changed_3.setIcon(self.icons_array_white[4])
                                    elif self.old_icon_piece_color_3 == 'black':
                                        self.icon_changed_3.setIcon(self.icons_array_white[5])
                                elif self.old_icon_piece_3 == 'bishop':
                                    if self.old_icon_piece_color_3 == 'white':
                                        self.icon_changed_3.setIcon(self.icons_array_white[6])
                                    elif self.old_icon_piece_color_3 == 'black':
                                        self.icon_changed_3.setIcon(self.icons_array_white[7])
                                elif self.old_icon_piece_3 == 'king':
                                    if self.old_icon_piece_color_3 == 'white':
                                        self.icon_changed_3.setIcon(self.icons_array_white[10])
                                    elif self.old_icon_piece_color_3 == 'black':
                                        self.icon_changed_3.setIcon(self.icons_array_white[11])
                                elif self.old_icon_piece_3 == 'queen':
                                    if self.old_icon_piece_color_3 == 'white':
                                        self.icon_changed_3.setIcon(self.icons_array_white[8])
                                    elif self.old_icon_piece_color_3 == 'black':
                                        self.icon_changed_3.setIcon(self.icons_array_white[9])
                            elif self.old_icon_color_3 == 'black':
                                if self.old_icon_piece_3 == 'pawn':
                                    if self.old_icon_piece_color_3 == 'white':
                                        self.icon_changed_3.setIcon(self.icons_array_black[0])
                                    elif self.old_icon_piece_color_3 == 'black':
                                        self.icon_changed_3.setIcon(self.icons_array_black[1])
                                elif self.old_icon_piece_3 == 'rook':
                                    if self.old_icon_piece_color_3 == 'white':
                                        self.icon_changed_3.setIcon(self.icons_array_black[2])
                                    elif self.old_icon_piece_color_3 == 'black':
                                        self.icon_changed_3.setIcon(self.icons_array_black[3])
                                elif self.old_icon_piece_3 == 'knight':
                                    if self.old_icon_piece_color_3 == 'white':
                                        self.icon_changed_3.setIcon(self.icons_array_black[4])
                                    elif self.old_icon_piece_color_3 == 'black':
                                        self.icon_changed_3.setIcon(self.icons_array_black[5])
                                elif self.old_icon_piece_3 == 'bishop':
                                    if self.old_icon_piece_color_3 == 'white':
                                        self.icon_changed_3.setIcon(self.icons_array_black[6])
                                    elif self.old_icon_piece_color_3 == 'black':
                                        self.icon_changed_3.setIcon(self.icons_array_black[7])
                                elif self.old_icon_piece_3 == 'king':
                                    if self.old_icon_piece_color_3 == 'white':
                                        self.icon_changed_3.setIcon(self.icons_array_black[10])
                                    elif self.old_icon_piece_color_3 == 'black':
                                        self.icon_changed_3.setIcon(self.icons_array_black[11])
                                elif self.old_icon_piece_3 == 'queen':
                                    if self.old_icon_piece_color_3 == 'white':
                                        self.icon_changed_3.setIcon(self.icons_array_black[8])
                                    elif self.old_icon_piece_color_3 == 'black':
                                        self.icon_changed_3.setIcon(self.icons_array_black[9])

                        if self.icon_changed_4 != 'none':
                            if self.old_icon_color_4 == 'white':
                                if self.old_icon_piece_4 == 'pawn':
                                    if self.old_icon_piece_color_4 == 'white':
                                        self.icon_changed_4.setIcon(self.icons_array_white[0])
                                    elif self.old_icon_piece_color_4 == 'black':
                                        self.icon_changed_4.setIcon(self.icons_array_white[1])
                                elif self.old_icon_piece_4 == 'rook':
                                    if self.old_icon_piece_color_4 == 'white':
                                        self.icon_changed_4.setIcon(self.icons_array_white[2])
                                    elif self.old_icon_piece_color_4 == 'black':
                                        self.icon_changed_4.setIcon(self.icons_array_white[3])
                                elif self.old_icon_piece_4 == 'knight':
                                    if self.old_icon_piece_color_4 == 'white':
                                        self.icon_changed_4.setIcon(self.icons_array_white[4])
                                    elif self.old_icon_piece_color_4 == 'black':
                                        self.icon_changed_4.setIcon(self.icons_array_white[5])
                                elif self.old_icon_piece_4 == 'bishop':
                                    if self.old_icon_piece_color_4 == 'white':
                                        self.icon_changed_4.setIcon(self.icons_array_white[6])
                                    elif self.old_icon_piece_color_4 == 'black':
                                        self.icon_changed_4.setIcon(self.icons_array_white[7])
                                elif self.old_icon_piece_4 == 'king':
                                    if self.old_icon_piece_color_4 == 'white':
                                        self.icon_changed_4.setIcon(self.icons_array_white[10])
                                    elif self.old_icon_piece_color_4 == 'black':
                                        self.icon_changed_4.setIcon(self.icons_array_white[11])
                                elif self.old_icon_piece_4 == 'queen':
                                    if self.old_icon_piece_color_4 == 'white':
                                        self.icon_changed_4.setIcon(self.icons_array_white[8])
                                    elif self.old_icon_piece_color_4 == 'black':
                                        self.icon_changed_4.setIcon(self.icons_array_white[9])
                            elif self.old_icon_color_4 == 'black':
                                if self.old_icon_piece_4 == 'pawn':
                                    if self.old_icon_piece_color_4 == 'white':
                                        self.icon_changed_4.setIcon(self.icons_array_black[0])
                                    elif self.old_icon_piece_color_4 == 'black':
                                        self.icon_changed_4.setIcon(self.icons_array_black[1])
                                elif self.old_icon_piece_4 == 'rook':
                                    if self.old_icon_piece_color_4 == 'white':
                                        self.icon_changed_4.setIcon(self.icons_array_black[2])
                                    elif self.old_icon_piece_color_4 == 'black':
                                        self.icon_changed_4.setIcon(self.icons_array_black[3])
                                elif self.old_icon_piece_4 == 'knight':
                                    if self.old_icon_piece_color_4 == 'white':
                                        self.icon_changed_4.setIcon(self.icons_array_black[4])
                                    elif self.old_icon_piece_color_4 == 'black':
                                        self.icon_changed_4.setIcon(self.icons_array_black[5])
                                elif self.old_icon_piece_4 == 'bishop':
                                    if self.old_icon_piece_color_4 == 'white':
                                        self.icon_changed_4.setIcon(self.icons_array_black[6])
                                    elif self.old_icon_piece_color_4 == 'black':
                                        self.icon_changed_4.setIcon(self.icons_array_black[7])
                                elif self.old_icon_piece_4 == 'king':
                                    if self.old_icon_piece_color_4 == 'white':
                                        self.icon_changed_4.setIcon(self.icons_array_black[10])
                                    elif self.old_icon_piece_color_4 == 'black':
                                        self.icon_changed_4.setIcon(self.icons_array_black[11])
                                elif self.old_icon_piece_4 == 'queen':
                                    if self.old_icon_piece_color_4 == 'white':
                                        self.icon_changed_4.setIcon(self.icons_array_black[8])
                                    elif self.old_icon_piece_color_4 == 'black':
                                        self.icon_changed_4.setIcon(self.icons_array_black[9])
                    #Change the yellow selection flag
                    self.yellow_selection_flag = not self.yellow_selection_flag

            elif state == 70: #Manual mode.
                print('ACTION: '+str(action))
                with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                        live_situation = yaml.load(live_file.read())
                if action ==  'delete':
                    square_object = live_situation[square][0]
                    print('SQUARE OBJ: '+str(square_object))
                    if square_object == 'none':
                            rospy.logwarn('There is no piece to delete in the selected square')
                    else:
                        #Save the color of the square
                        square_color = self.chessboard_colors[square]
                        if square_color == 'white':
                            self.array[self.dict_squares[square]].setIcon(QtGui.QIcon())
                            #Change the live chessboard situation
                            live_situation[square][0] = 'none'
                            live_situation[square][1] = 'none'
                            with open(self.dir_live_chessboard_situation, "w") as t_p:
                                yaml.dump(live_situation, t_p)
                        elif square_color == 'black':
                            self.array[self.dict_squares[square]].setIcon(self.icon_grey)
                            #Change the live chessboard situation
                            live_situation[square][0] = 'none'
                            live_situation[square][1] = 'none'
                            with open(self.dir_live_chessboard_situation, "w") as t_p:
                                yaml.dump(live_situation, t_p)

                else: #If the action is to manually place a piece in the chessboard.
                    #Save the color of the square
                    square_color = self.chessboard_colors[square]

                    #Save the pieces that are currently in the live situation
                    present_bishops_w = []
                    present_rooks_w = []
                    present_knights_w = []
                    present_pawns_w = []
                    present_bishops_b = []
                    present_rooks_b = []
                    present_knights_b = []
                    present_pawns_b = []

                    for casella in live_situation:
                        if casella[0] == 'bishop_c1' or casella[0] == 'bishop_f1':
                            present_bishops_w.append(casella[0])
                        elif casella[0] == 'bishop_c8' or casella[0] == 'bishop_f8':
                            present_bishops_b.append(casella[0])
                        elif casella[0] == 'knight_b1' or casella[0] == 'knight_g1':
                            present_knights_w.append(casella[0])
                        elif casella[0] == 'knight_b8' or casella[0] == 'knight_g8':
                            present_knights_b.append(casella[0])
                        elif casella[0] == 'rook_a1' or casella[0] == 'rook_h1':
                            present_rooks_w.append(casella[0])
                        elif casella[0] == 'rook_a8' or casella[0] == 'rook_h8':
                            present_rooks_b.append(casella[0])
                        elif casella[0] == 'pawn_a2' or casella[0] == 'pawn_b2' or casella[0] == 'pawn_c2' or casella[0] == 'pawn_d2' or casella[0] == 'pawn_e2' or casella[0] == 'pawn_f2' or casella[0] == 'pawn_g2' or casella[0] == 'pawn_h2':
                            present_pawns_w.append(casella[0])
                        elif casella[0] == 'pawn_a7' or casella[0] == 'pawn_b7' or casella[0] == 'pawn_c7' or casella[0] == 'pawn_d7' or casella[0] == 'pawn_e7' or casella[0] == 'pawn_f7' or casella[0] == 'pawn_g7' or casella[0] == 'pawn_h7':
                            present_pawns_b.append(casella[0])

                    if square_color == 'white':
                        if action == 'w_pawn':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_white[0])
                            #Change the live chessboard situation
                            if square_object in pawns_w:
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'white'
                            else:
                                if len(present_pawns_w) == 0:
                                    live_situation[square][0] = 'pawn_a2'
                                    live_situation[square][1] = 'white'
                                elif len(present_pawns_w) == 8:
                                    live_situation[square][0] = 'inserted_pawn'
                                    live_situation[square][1] = 'white'
                                else:
                                    self.white_pawns = cfg_w.white_pawns
                                    for pawn in self.white_pawns:
                                        if pawn not in present_pawns_w:
                                            live_situation[square][0] = pawn
                                            live_situation[square][1] = 'white'
                                            break
                        elif action == 'b_pawn':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_white[1])
                            #Change the live chessboard situation
                            if square_object in pawns_b:
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'black'
                            else:
                                if len(present_pawns_b) == 0:
                                    live_situation[square][0] = 'pawn_a7'
                                    live_situation[square][1] = 'black'
                                elif len(present_pawns_b) == 8:
                                    live_situation[square][0] = 'inserted_pawn'
                                    live_situation[square][1] = 'black'
                                else:
                                    self.black_pawns = cfg_w.black_pawns
                                    for pawn in self.black_pawns:
                                        if pawn not in present_pawns_b:
                                            live_situation[square][0] = pawn
                                            live_situation[square][1] = 'black'
                                            break
                        elif action == 'w_knight':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_white[4])
                            #Change the live chessboard situation
                            if square_object == 'knight_g1' or square_object == 'knight_b1':
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'white'
                            else:
                                if len(present_knights_w) == 1:
                                    live_situation[square][0] = present_knights_w[0]
                                    live_situation[square][1] = 'white'
                                elif len(present_knights_w) == 2:
                                    live_situation[square][0] = 'inserted_knight'
                                    live_situation[square][1] = 'white'
                                elif len(present_knights_w) == 0:
                                    live_situation[square][0] = 'knight_b1'
                                    live_situation[square][1] = 'white'                    
                        elif action == 'b_knight':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_white[5])
                            #Change the live chessboard situation
                            if square_object == 'knight_g8' or square_object == 'knight_b8':
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'black'
                            else:
                                if len(present_knights_b) == 1:
                                    live_situation[square][0] = present_knights_b[0]
                                    live_situation[square][1] = 'black'
                                elif len(present_knights_b) == 2:
                                    live_situation[square][0] = 'inserted_knight'
                                    live_situation[square][1] = 'black'
                                elif len(present_knights_b) == 0:
                                    live_situation[square][0] = 'knight_b8'
                                    live_situation[square][1] = 'black'                        
                        elif action == 'w_bishop':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_white[6])
                            #Change the live chessboard situation
                            if square_object == 'bishop_c1' or square_object == 'bishop_f1':
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'white'
                            else:
                                if len(present_bishops_w) == 1:
                                    live_situation[square][0] = present_bishops_w[0]
                                    live_situation[square][1] = 'white'
                                elif len(present_bishops_w) == 2:
                                    live_situation[square][0] = 'inserted_bishop'
                                    live_situation[square][1] = 'white'
                                elif len(present_bishops_w) == 0:
                                    live_situation[square][0] = 'bishop_c1'
                                    live_situation[square][1] = 'white'                        
                        elif action == 'b_bishop':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_white[7])
                            #Change the live chessboard situation
                            if square_object == 'bishop_c8' or square_object == 'bishop_f8':
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'black'
                            else:
                                if len(present_bishops_b) == 1:
                                    live_situation[square][0] = present_bishops_b[0]
                                    live_situation[square][1] = 'black'
                                elif len(present_bishops_b) == 2:
                                    live_situation[square][0] = 'inserted_bishop'
                                    live_situation[square][1] = 'black'
                                elif len(present_bishops_b) == 0:
                                    live_situation[square][0] = 'bishop_c8'
                                    live_situation[square][1] = 'black'                        
                        elif action == 'w_rook':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_white[2])
                            #Change the live chessboard situation
                            if square_object == 'rook_a1' or square_object == 'rook_h1':
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'white'
                            else:
                                if len(present_rooks_w) == 1:
                                    live_situation[square][0] = present_rooks_w[0]
                                    live_situation[square][1] = 'white'
                                elif len(present_rooks_w) == 2:
                                    live_situation[square][0] = 'inserted_rook'
                                    live_situation[square][1] = 'white'
                                elif len(present_rooks_w) == 0:
                                    live_situation[square][0] = 'rook_a1'
                                    live_situation[square][1] = 'white'                        
                        elif action == 'b_rook':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_white[3])
                            #Change the live chessboard situation
                            if square_object == 'rook_a8' or square_object == 'rook_h8':
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'black'
                            else:
                                if len(present_rooks_b) == 1:
                                    live_situation[square][0] = present_rooks_b[0]
                                    live_situation[square][1] = 'black'
                                elif len(present_rooks_b) == 2:
                                    live_situation[square][0] = 'inserted_rook'
                                    live_situation[square][1] = 'black'
                                elif len(present_rooks_b) == 0:
                                    live_situation[square][0] = 'rook_a8'
                                    live_situation[square][1] = 'black'                       
                        elif action == 'w_queen':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_white[8])
                            #Change the live chessboard situation
                            live_situation[square][0] = 'queen_d1'
                            live_situation[square][1] = 'white'                        
                        elif action == 'b_queen':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_white[9])
                            #Change the live chessboard situation
                            live_situation[square][0] = 'queen_d8'
                            live_situation[square][1] = 'black'                        
                        elif action == 'w_king':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_white[10])
                            #Change the live chessboard situation
                            live_situation[square][0] = 'king_e1'
                            live_situation[square][1] = 'white'                        
                        elif action == 'b_king':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_white[11])
                            #Change the live chessboard situation
                            live_situation[square][0] = 'king_e8'
                            live_situation[square][1] = 'black'                        
                    elif square_color == 'black':
                        if action == 'w_pawn':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_black[0])
                            #Change the live chessboard situation
                            if square_object in pawns_w:
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'white'
                            else:
                                if len(present_pawns_w) == 0:
                                    live_situation[square][0] = 'pawn_a2'
                                    live_situation[square][1] = 'white'
                                elif len(present_pawns_w) == 8:
                                    live_situation[square][0] = 'inserted_pawn'
                                    live_situation[square][1] = 'white'
                                else:
                                    self.white_pawns = cfg_w.white_pawns
                                    for pawn in self.white_pawns:
                                        if pawn not in present_pawns_w:
                                            live_situation[square][0] = pawn
                                            live_situation[square][1] = 'white'
                                            break
                        elif action == 'b_pawn':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_black[1])
                            #Change the live chessboard situation
                            if square_object in pawns_b:
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'black'
                            else:
                                if len(present_pawns_b) == 0:
                                    live_situation[square][0] = 'pawn_a7'
                                    live_situation[square][1] = 'black'
                                elif len(present_pawns_b) == 8:
                                    live_situation[square][0] = 'inserted_pawn'
                                    live_situation[square][1] = 'black'
                                else:
                                    self.black_pawns = cfg_w.black_pawns
                                    for pawn in self.black_pawns:
                                        if pawn not in present_pawns_b:
                                            live_situation[square][0] = pawn
                                            live_situation[square][1] = 'black'
                                            break
                        elif action == 'w_knight':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_black[4])
                            #Change the live chessboard situation
                            if square_object == 'knight_g1' or square_object == 'knight_b1':
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'white'
                            else:
                                if len(present_knights_w) == 1:
                                    live_situation[square][0] = present_knights_w[0]
                                    live_situation[square][1] = 'white'
                                elif len(present_knights_w) == 2:
                                    live_situation[square][0] = 'inserted_knight'
                                    live_situation[square][1] = 'white'
                                elif len(present_knights_w) == 0:
                                    live_situation[square][0] = 'knight_b1'
                                    live_situation[square][1] = 'white'                    
                        elif action == 'b_knight':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_black[5])
                            #Change the live chessboard situation
                            if square_object == 'knight_g8' or square_object == 'knight_b8':
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'black'
                            else:
                                if len(present_knights_b) == 1:
                                    live_situation[square][0] = present_knights_b[0]
                                    live_situation[square][1] = 'black'
                                elif len(present_knights_b) == 2:
                                    live_situation[square][0] = 'inserted_knight'
                                    live_situation[square][1] = 'black'
                                elif len(present_knights_b) == 0:
                                    live_situation[square][0] = 'knight_b8'
                                    live_situation[square][1] = 'black'                        
                        elif action == 'w_bishop':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_black[6])
                            #Change the live chessboard situation
                            if square_object == 'bishop_c1' or square_object == 'bishop_f1':
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'white'
                            else:
                                if len(present_bishops_w) == 1:
                                    live_situation[square][0] = present_bishops_w[0]
                                    live_situation[square][1] = 'white'
                                elif len(present_bishops_w) == 2:
                                    live_situation[square][0] = 'inserted_bishop'
                                    live_situation[square][1] = 'white'
                                elif len(present_bishops_w) == 0:
                                    live_situation[square][0] = 'bishop_c1'
                                    live_situation[square][1] = 'white'                        
                        elif action == 'b_bishop':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_black[7])
                            #Change the live chessboard situation
                            if square_object == 'bishop_c8' or square_object == 'bishop_f8':
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'black'
                            else:
                                if len(present_bishops_b) == 1:
                                    live_situation[square][0] = present_bishops_b[0]
                                    live_situation[square][1] = 'black'
                                elif len(present_bishops_b) == 2:
                                    live_situation[square][0] = 'inserted_bishop'
                                    live_situation[square][1] = 'black'
                                elif len(present_bishops_b) == 0:
                                    live_situation[square][0] = 'bishop_c8'
                                    live_situation[square][1] = 'black'                        
                        elif action == 'w_rook':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_black[2])
                            #Change the live chessboard situation
                            if square_object == 'rook_a1' or square_object == 'rook_h1':
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'white'
                            else:
                                if len(present_rooks_w) == 1:
                                    live_situation[square][0] = present_rooks_w[0]
                                    live_situation[square][1] = 'white'
                                elif len(present_rooks_w) == 2:
                                    live_situation[square][0] = 'inserted_rook'
                                    live_situation[square][1] = 'white'
                                elif len(present_rooks_w) == 0:
                                    live_situation[square][0] = 'rook_a1'
                                    live_situation[square][1] = 'white'                        
                        elif action == 'b_rook':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_black[3])
                            #Change the live chessboard situation
                            if square_object == 'rook_a8' or square_object == 'rook_h8':
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'black'
                            else:
                                if len(present_rooks_b) == 1:
                                    live_situation[square][0] = present_rooks_b[0]
                                    live_situation[square][1] = 'black'
                                elif len(present_rooks_b) == 2:
                                    live_situation[square][0] = 'inserted_rook'
                                    live_situation[square][1] = 'black'
                                elif len(present_rooks_b) == 0:
                                    live_situation[square][0] = 'rook_a8'
                                    live_situation[square][1] = 'black'                       
                        elif action == 'w_queen':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_black[8])
                            #Change the live chessboard situation
                            live_situation[square][0] = 'queen_d1'
                            live_situation[square][1] = 'white'                        
                        elif action == 'b_queen':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_black[9])
                            #Change the live chessboard situation
                            live_situation[square][0] = 'queen_d8'
                            live_situation[square][1] = 'black'                        
                        elif action == 'w_king':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_black[10])
                            #Change the live chessboard situation
                            live_situation[square][0] = 'king_e1'
                            live_situation[square][1] = 'white'                        
                        elif action == 'b_king':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_black[11])
                            #Change the live chessboard situation
                            live_situation[square][0] = 'king_e8'
                            live_situation[square][1] = 'black'
                    with open(self.dir_live_chessboard_situation, "w") as t_p:
                            yaml.dump(live_situation, t_p)

    def click_and_focus(self, square):
        global square_object
        global square_captured_piece
        if state == 11: #If we are in game mode.
            #Change the current_focus variable.
            self.current_focus = square
            print('CURRENT FOCUS: ' + str(self.current_focus))

            #If no other square is selected, select the square and change the color of it in red.
            if not self.selection_flag and not self.yellow_selection_flag:
                    #Read if the selected square is occupied by one piece or if it's empty.
                    with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                            live_situation = yaml.load(live_file.read())
                    square_object = live_situation[self.current_focus][0]
                    if square_object == 'none':
                            rospy.logwarn('There is no piece to move in the selected square')
                            self.selection_flag = not self.selection_flag
                    else:
                        self.color_to_move = live_situation[self.current_focus][1]
                        if self.pieces_coordinates[square_object][1]['name'] == 'pawn':
                            if self.color_to_move == 'white':
                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Wpawn)
                            elif self.color_to_move == 'black':
                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Bpawn)
                        elif self.pieces_coordinates[square_object][1]['name'] == 'rook':
                            if self.color_to_move == 'white':
                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Wrook)
                            elif self.color_to_move == 'black':
                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Brook)
                        elif self.pieces_coordinates[square_object][1]['name'] == 'knight':
                            if self.color_to_move == 'white':
                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Wknight)
                            elif self.color_to_move == 'black':
                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Bknight)
                        elif self.pieces_coordinates[square_object][1]['name'] == 'bishop':
                            if self.color_to_move == 'white':
                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Wbishop)
                            elif self.color_to_move == 'black':
                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Bbishop)
                        elif self.pieces_coordinates[square_object][1]['name'] == 'queen':
                            if self.color_to_move == 'white':
                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Wqueen)
                            elif self.color_to_move == 'black':
                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Bqueen)
                        elif self.pieces_coordinates[square_object][1]['name'] == 'king':
                            if self.color_to_move == 'white':
                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Wking)
                            elif self.color_to_move == 'black':
                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Bking)
                        self.current_red_icon = self.current_focus #Save the icon that currently is highlighted.
                    self.selection_flag = not self.selection_flag

            elif self.selection_flag and not self.yellow_selection_flag:
                    #Read which piece is in the currently highlighted square
                    with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                            live_situation = yaml.load(live_file.read())
                    self.piece_to_move = self.pieces_coordinates[live_situation[self.current_red_icon][0]][1]['name']
                    piece_to_capture = live_situation[self.current_focus][0]
                    self.color_to_move = live_situation[self.current_red_icon][1]
                    
                    #Save the color of the destination square and of the starting square
                    destination_color = self.chessboard_colors[self.current_focus]
                    starting_color = self.chessboard_colors[self.current_red_icon]

                    #Initialize variables to keep trak of the intentions of the player
                    self.icon_changed_1 = 'none'
                    self.old_icon_color_1 = 'none'
                    self.icon_changed_2 = 'none'
                    self.old_icon_color_2 = 'none'
                    self.icon_changed_3 = 'none'
                    self.old_icon_color_3 = 'none'
                    self.old_icon_piece_3 = 'none'
                    self.old_icon_piece_color_3 = 'none'
                    self.icon_changed_4 = 'none'
                    self.old_icon_color_4 = 'none'
                    self.old_icon_piece_4 = 'none'
                    self.old_icon_piece_color_4 = 'none'

                    #Change the destination icon and the starting square icon (destination color in yellow, cause I need to implement move confirmation)
                    if self.piece_to_move == 'pawn':
                        if self.color_to_move == 'white':
                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[0])
                            self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                            self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                        elif self.color_to_move == 'black':
                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[1])
                            self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                            self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                        #Implement the en-passant special move (only for white. For black will be implemented in the other chessboard GUI script):
                        count = 0
                        for i in self.chessboard_matrix:
                            if self.current_red_icon in i:
                                prev_index = i.index(self.current_red_icon)
                                capture_row = count
                            if self.current_focus in i:
                                new_index = i.index(self.current_focus)
                            count += 1
                        if prev_index != new_index and piece_to_capture == 'none': #If a capture has been performed by the pawn and there was no piece in the ending square, en-passant happened.
                            #self.en_passant = True
                            square_captured_piece = self.chessboard_matrix[capture_row][new_index]
                            if self.chessboard_colors[square_captured_piece] == 'black':
                                self.array[self.dict_squares[square_captured_piece]].setIcon(self.icon_grey)
                                self.icon_changed_3 = self.array[self.dict_squares[square_captured_piece]]
                                self.old_icon_color_3 = 'black'
                                self.old_icon_piece_3 = 'pawn'
                                self.old_icon_piece_color_3 = 'black' #Cause I know I'm playing with white
                            elif self.chessboard_colors[square_captured_piece] == 'white':
                                self.array[self.dict_squares[square_captured_piece]].setIcon(QtGui.QIcon())
                                self.icon_changed_3 = self.array[self.dict_squares[square_captured_piece]]
                                self.old_icon_color_3 = 'white'
                                self.old_icon_piece_3 = 'pawn'
                                self.old_icon_piece_color_3 = 'black' #Cause I know I'm playing with white
                            #en_passant_square_to_move_publisher.publish(square_captured_piece)

                    elif self.piece_to_move == 'rook':
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[2])
                                    self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                    self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[3])
                                    self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                    self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                    elif self.piece_to_move == 'knight':
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[4])
                                    self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                    self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[5])
                                    self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                    self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                    elif self.piece_to_move == 'bishop':
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[6])
                                    self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                    self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[7])
                                    self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                    self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                    elif self.piece_to_move == 'queen':
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[8])
                                    self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                    self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[9])
                                    self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                    self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                    elif self.piece_to_move == 'king':
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[10])
                                    self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                    self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[11])
                                    self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                    self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                            #Implement the castle special move (only for white. For black it will be implemented in the ther chessboard GUI script):
                            if (self.current_focus == 'g1' or self.current_focus == 'c1') and (self.current_red_icon == 'e1'):
                                    if self.current_focus == 'g1': #Short castle
                                            #self.short_castle = True
                                            self.f1.setIcon(self.icons_array_yellow[2])
                                            self.icon_changed_2 = self.array[self.dict_squares['f1']]
                                            self.old_icon_color_2 = self.chessboard_colors['f1']
                                            self.h1.setIcon(QtGui.QIcon())
                                    elif self.current_focus == 'c1': #Long castle 
                                            #self.long_castle = True
                                            self.d1.setIcon(self.icons_array_yellow[2])
                                            self.icon_changed_2 = self.array[self.dict_squares['d1']]
                                            self.old_icon_color_2 = self.chessboard_colors['d1']
                                            self.a1.setIcon(self.icon_grey)
                    if starting_color == 'white':
                            self.array[self.dict_squares[self.current_red_icon]].setIcon(QtGui.QIcon())
                            self.icon_changed_4 = self.array[self.dict_squares[self.current_red_icon]]
                            self.old_icon_color_4 = 'white'
                            self.old_icon_piece_4 = self.piece_to_move
                            self.old_icon_piece_color_4 = self.color_to_move
                    elif starting_color == 'black':
                            self.array[self.dict_squares[self.current_red_icon]].setIcon(self.icon_grey)
                            self.icon_changed_4 = self.array[self.dict_squares[self.current_red_icon]]
                            self.old_icon_color_4 = 'black'
                            self.old_icon_piece_4 = self.piece_to_move
                            self.old_icon_piece_color_4 = self.color_to_move

                    self.yellow_selection_flag = not self.yellow_selection_flag
                    self.selection_flag = not self.selection_flag
                    self.intention_icon = self.current_focus #Save the square that the player is intentioned to play to.

            elif not self.selection_flag and self.yellow_selection_flag:
                if self.current_focus == self.intention_icon: #If the player clicked on the same square again, the move is confirmed.
                    #Read which piece is in the currently highlighted square
                    with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                            live_situation = yaml.load(live_file.read())
                    self.piece_to_move = self.pieces_coordinates[live_situation[self.current_red_icon][0]][1]['name']
                    piece_to_capture = live_situation[self.current_focus][0]
                    self.color_to_move = live_situation[self.current_red_icon][1]
                    
                    #Save the color of the destination square and of the starting square
                    destination_color = self.chessboard_colors[self.current_focus]
                    starting_color = self.chessboard_colors[self.current_red_icon]

                    #Initialize the promoted_piece, the en_passant and the castle variables
                    #promoted_piece = 'none'
                    self.en_passant = False
                    self.short_castle = False
                    self.long_castle = False

                    if destination_color == 'black':
                            if self.piece_to_move == 'pawn':
                                    #Implement promotion (only for white. For black will be implemented in the other chessboard GUI script)
                                    count = 0
                                    for r in self.chessboard_matrix:
                                        if self.current_focus in r:
                                                dest_row = count
                                                break
                                        count += 1
                                    if dest_row == 7: #Promotion is happening
                                            state_publisher.publish(12) #Go to the state of promoted piece question
                                            #Make the user choose the promoted piece
                                            if self.color_to_move == 'white':
                                                self.promotion.emit(1)
                                            elif self.color_to_move == 'black':
                                                self.promotion.emit(2)                            
                                    else:
                                            if self.color_to_move == 'white':
                                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[0])
                                            elif self.color_to_move == 'black':
                                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[1])
                                            #Implement the en-passant special move (only for white. For black will be implemented in the other chessboard GUI script):
                                            count = 0
                                            for i in self.chessboard_matrix:
                                                    if self.current_red_icon in i:
                                                            prev_index = i.index(self.current_red_icon)
                                                            capture_row = count
                                                    if self.current_focus in i:
                                                            new_index = i.index(self.current_focus)
                                                    count += 1
                                            if prev_index != new_index and piece_to_capture == 'none': #If a capture has been performed by the pawn and there was no piece in the ending square, en-passant happened.
                                                    self.en_passant = True
                                                    square_captured_piece = self.chessboard_matrix[capture_row][new_index]
                                                    en_passant_square_to_move_publisher.publish(square_captured_piece)
                            elif self.piece_to_move == 'rook':
                                    if self.color_to_move == 'white':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[2])
                                    elif self.color_to_move == 'black':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[3])
                            elif self.piece_to_move == 'knight':
                                    if self.color_to_move == 'white':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[4])
                                    elif self.color_to_move == 'black':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[5])
                            elif self.piece_to_move == 'bishop':
                                    if self.color_to_move == 'white':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[6])
                                    elif self.color_to_move == 'black':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[7])
                            elif self.piece_to_move == 'queen':
                                    if self.color_to_move == 'white':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[8])
                                    elif self.color_to_move == 'black':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[9])
                            elif self.piece_to_move == 'king':
                                    if self.color_to_move == 'white':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[10])
                                    elif self.color_to_move == 'black':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[11])
                                    #Implement the castle special move (only for white. For black it will be implemented in the ther chessboard GUI script):
                                    if (self.current_focus == 'g1' or self.current_focus == 'c1') and (self.current_red_icon == 'e1'):
                                            if self.current_focus == 'g1': #Short castle
                                                    self.short_castle = True
                                                    self.f1.setIcon(self.icons_array_white[2])
                                                    self.h1.setIcon(QtGui.QIcon())
                                            elif self.current_focus == 'c1': #Long castle 
                                                    self.long_castle = True
                                                    self.d1.setIcon(self.icons_array_white[2])
                                                    self.a1.setIcon(self.icon_grey)
                    elif destination_color == 'white':
                            if self.piece_to_move == 'pawn':
                                    #Implement promotion
                                    count = 0
                                    for r in self.chessboard_matrix:
                                        if self.current_focus in r:
                                                dest_row = count
                                                break
                                        count += 1
                                    if dest_row == 7: #Promotion is happening
                                            state_publisher.publish(12) #Go to the state of promoted piece question
                                            #Make the user choose the promoted piece
                                            if self.color_to_move == 'white':
                                                self.promotion.emit(1)
                                            elif self.color_to_move == 'black':
                                                self.promotion.emit(2)                                       
                                    else:
                                            if self.color_to_move == 'white':
                                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[0])
                                            elif self.color_to_move == 'black':
                                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[1])
                                            #Implement the en-passant special move:
                                            count = 0
                                            for i in self.chessboard_matrix:                                                    
                                                    if self.current_red_icon in i:
                                                            prev_index = i.index(self.current_red_icon)
                                                            capture_row = count
                                                    if self.current_focus in i:
                                                            new_index = i.index(self.current_focus)
                                                    count += 1
                                            if prev_index != new_index and piece_to_capture == 'none': #If a capture has been performed by the pawn and there was no piece in the ending square, en-passant happened.
                                                    self.en_passant = True
                                                    square_captured_piece = self.chessboard_matrix[capture_row][new_index]
                                                    en_passant_square_to_move_publisher.publish(square_captured_piece)

                            elif self.piece_to_move == 'rook':
                                    if self.color_to_move == 'white':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[2])
                                    elif self.color_to_move == 'black':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[3])
                            elif self.piece_to_move == 'knight':
                                    if self.color_to_move == 'white':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[4])
                                    elif self.color_to_move == 'black':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[5])
                            elif self.piece_to_move == 'bishop':
                                    if self.color_to_move == 'white':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[6])
                                    elif self.color_to_move == 'black':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[7])
                            elif self.piece_to_move == 'queen':
                                    if self.color_to_move == 'white':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[8])
                                    elif self.color_to_move == 'black':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[9])
                            elif self.piece_to_move == 'king':
                                    if self.color_to_move == 'white':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[10])
                                    elif self.color_to_move == 'black':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[11])
                    #Send a message containing the start square and the end square, so that TIAGo can execute it.
                    if self.short_castle:
                        self.start_square_move_publisher.publish('short')
                        self.end_square_move_publisher.publish('short')
                    elif self.long_castle:
                        self.start_square_move_publisher.publish('long')
                        self.end_square_move_publisher.publish('long')
                    else:
                        self.start_square_move_publisher.publish(self.current_red_icon)
                        self.end_square_move_publisher.publish(self.current_focus)
                        #self.promoted_piece_move_publisher.publish(promoted_piece)

                    #Change operational status.
                    if state != 12:
                        state_publisher.publish(13)
                        update_flag = True

                else: #If the player is not confirming the move (by clicking in another square), get back to the state where he has to chosen the move.
                    if self.icon_changed_1 != 'none':
                        if self.old_icon_color_1 == 'white':
                            self.icon_changed_1.setIcon(QtGui.QIcon())
                        elif self.old_icon_color_1 == 'black':
                            self.icon_changed_1.setIcon(self.icon_grey)

                    if self.icon_changed_2 != 'none':
                        if self.old_icon_color_2 == 'white':
                            self.icon_changed_2.setIcon(QtGui.QIcon())
                        elif self.old_icon_color_2 == 'black':
                            self.icon_changed_2.setIcon(self.icon_grey)

                    if self.icon_changed_3 != 'none':
                        if self.old_icon_color_3 == 'white':
                            if self.old_icon_piece_3 == 'pawn':
                                if self.old_icon_piece_color_3 == 'white':
                                    self.icon_changed_3.setIcon(self.icons_array_white[0])
                                elif self.old_icon_piece_color_3 == 'black':
                                    self.icon_changed_3.setIcon(self.icons_array_white[1])
                            elif self.old_icon_piece_3 == 'rook':
                                if self.old_icon_piece_color_3 == 'white':
                                    self.icon_changed_3.setIcon(self.icons_array_white[2])
                                elif self.old_icon_piece_color_3 == 'black':
                                    self.icon_changed_3.setIcon(self.icons_array_white[3])
                            elif self.old_icon_piece_3 == 'knight':
                                if self.old_icon_piece_color_3 == 'white':
                                    self.icon_changed_3.setIcon(self.icons_array_white[4])
                                elif self.old_icon_piece_color_3 == 'black':
                                    self.icon_changed_3.setIcon(self.icons_array_white[5])
                            elif self.old_icon_piece_3 == 'bishop':
                                if self.old_icon_piece_color_3 == 'white':
                                    self.icon_changed_3.setIcon(self.icons_array_white[6])
                                elif self.old_icon_piece_color_3 == 'black':
                                    self.icon_changed_3.setIcon(self.icons_array_white[7])
                            elif self.old_icon_piece_3 == 'king':
                                if self.old_icon_piece_color_3 == 'white':
                                    self.icon_changed_3.setIcon(self.icons_array_white[10])
                                elif self.old_icon_piece_color_3 == 'black':
                                    self.icon_changed_3.setIcon(self.icons_array_white[11])
                            elif self.old_icon_piece_3 == 'queen':
                                if self.old_icon_piece_color_3 == 'white':
                                    self.icon_changed_3.setIcon(self.icons_array_white[8])
                                elif self.old_icon_piece_color_3 == 'black':
                                    self.icon_changed_3.setIcon(self.icons_array_white[9])
                        elif self.old_icon_color_3 == 'black':
                            if self.old_icon_piece_3 == 'pawn':
                                if self.old_icon_piece_color_3 == 'white':
                                    self.icon_changed_3.setIcon(self.icons_array_black[0])
                                elif self.old_icon_piece_color_3 == 'black':
                                    self.icon_changed_3.setIcon(self.icons_array_black[1])
                            elif self.old_icon_piece_3 == 'rook':
                                if self.old_icon_piece_color_3 == 'white':
                                    self.icon_changed_3.setIcon(self.icons_array_black[2])
                                elif self.old_icon_piece_color_3 == 'black':
                                    self.icon_changed_3.setIcon(self.icons_array_black[3])
                            elif self.old_icon_piece_3 == 'knight':
                                if self.old_icon_piece_color_3 == 'white':
                                    self.icon_changed_3.setIcon(self.icons_array_black[4])
                                elif self.old_icon_piece_color_3 == 'black':
                                    self.icon_changed_3.setIcon(self.icons_array_black[5])
                            elif self.old_icon_piece_3 == 'bishop':
                                if self.old_icon_piece_color_3 == 'white':
                                    self.icon_changed_3.setIcon(self.icons_array_black[6])
                                elif self.old_icon_piece_color_3 == 'black':
                                    self.icon_changed_3.setIcon(self.icons_array_black[7])
                            elif self.old_icon_piece_3 == 'king':
                                if self.old_icon_piece_color_3 == 'white':
                                    self.icon_changed_3.setIcon(self.icons_array_black[10])
                                elif self.old_icon_piece_color_3 == 'black':
                                    self.icon_changed_3.setIcon(self.icons_array_black[11])
                            elif self.old_icon_piece_3 == 'queen':
                                if self.old_icon_piece_color_3 == 'white':
                                    self.icon_changed_3.setIcon(self.icons_array_black[8])
                                elif self.old_icon_piece_color_3 == 'black':
                                    self.icon_changed_3.setIcon(self.icons_array_black[9])

                    if self.icon_changed_4 != 'none':
                        if self.old_icon_color_4 == 'white':
                            if self.old_icon_piece_4 == 'pawn':
                                if self.old_icon_piece_color_4 == 'white':
                                    self.icon_changed_4.setIcon(self.icons_array_white[0])
                                elif self.old_icon_piece_color_4 == 'black':
                                    self.icon_changed_4.setIcon(self.icons_array_white[1])
                            elif self.old_icon_piece_4 == 'rook':
                                if self.old_icon_piece_color_4 == 'white':
                                    self.icon_changed_4.setIcon(self.icons_array_white[2])
                                elif self.old_icon_piece_color_4 == 'black':
                                    self.icon_changed_4.setIcon(self.icons_array_white[3])
                            elif self.old_icon_piece_4 == 'knight':
                                if self.old_icon_piece_color_4 == 'white':
                                    self.icon_changed_4.setIcon(self.icons_array_white[4])
                                elif self.old_icon_piece_color_4 == 'black':
                                    self.icon_changed_4.setIcon(self.icons_array_white[5])
                            elif self.old_icon_piece_4 == 'bishop':
                                if self.old_icon_piece_color_4 == 'white':
                                    self.icon_changed_4.setIcon(self.icons_array_white[6])
                                elif self.old_icon_piece_color_4 == 'black':
                                    self.icon_changed_4.setIcon(self.icons_array_white[7])
                            elif self.old_icon_piece_4 == 'king':
                                if self.old_icon_piece_color_4 == 'white':
                                    self.icon_changed_4.setIcon(self.icons_array_white[10])
                                elif self.old_icon_piece_color_4 == 'black':
                                    self.icon_changed_4.setIcon(self.icons_array_white[11])
                            elif self.old_icon_piece_4 == 'queen':
                                if self.old_icon_piece_color_4 == 'white':
                                    self.icon_changed_4.setIcon(self.icons_array_white[8])
                                elif self.old_icon_piece_color_4 == 'black':
                                    self.icon_changed_4.setIcon(self.icons_array_white[9])
                        elif self.old_icon_color_4 == 'black':
                            if self.old_icon_piece_4 == 'pawn':
                                if self.old_icon_piece_color_4 == 'white':
                                    self.icon_changed_4.setIcon(self.icons_array_black[0])
                                elif self.old_icon_piece_color_4 == 'black':
                                    self.icon_changed_4.setIcon(self.icons_array_black[1])
                            elif self.old_icon_piece_4 == 'rook':
                                if self.old_icon_piece_color_4 == 'white':
                                    self.icon_changed_4.setIcon(self.icons_array_black[2])
                                elif self.old_icon_piece_color_4 == 'black':
                                    self.icon_changed_4.setIcon(self.icons_array_black[3])
                            elif self.old_icon_piece_4 == 'knight':
                                if self.old_icon_piece_color_4 == 'white':
                                    self.icon_changed_4.setIcon(self.icons_array_black[4])
                                elif self.old_icon_piece_color_4 == 'black':
                                    self.icon_changed_4.setIcon(self.icons_array_black[5])
                            elif self.old_icon_piece_4 == 'bishop':
                                if self.old_icon_piece_color_4 == 'white':
                                    self.icon_changed_4.setIcon(self.icons_array_black[6])
                                elif self.old_icon_piece_color_4 == 'black':
                                    self.icon_changed_4.setIcon(self.icons_array_black[7])
                            elif self.old_icon_piece_4 == 'king':
                                if self.old_icon_piece_color_4 == 'white':
                                    self.icon_changed_4.setIcon(self.icons_array_black[10])
                                elif self.old_icon_piece_color_4 == 'black':
                                    self.icon_changed_4.setIcon(self.icons_array_black[11])
                            elif self.old_icon_piece_4 == 'queen':
                                if self.old_icon_piece_color_4 == 'white':
                                    self.icon_changed_4.setIcon(self.icons_array_black[8])
                                elif self.old_icon_piece_color_4 == 'black':
                                    self.icon_changed_4.setIcon(self.icons_array_black[9])
                #Change the yellow selection flag
                self.yellow_selection_flag = not self.yellow_selection_flag

        elif state == 70: #Manual mode.
            with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                    live_situation = yaml.load(live_file.read())
            if action ==  'delete':
                square_object = live_situation[square][0]
                if square_object == 'none':
                        rospy.logwarn('There is no piece to delete in the selected square')
                else:
                    #Save the color of the square
                    square_color = self.chessboard_colors[square]
                    if square_color == 'white':
                        self.array[self.dict_squares[square]].setIcon(QtGui.QIcon())
                        #Change the live chessboard situation
                        live_situation[square][0] = 'none'
                        live_situation[square][1] = 'none'
                        with open(self.dir_live_chessboard_situation, "w") as t_p:
                            yaml.dump(live_situation, t_p)
                    elif square_color == 'black':
                        self.array[self.dict_squares[square]].setIcon(self.icon_grey)
                        #Change the live chessboard situation
                        live_situation[square][0] = 'none'
                        live_situation[square][1] = 'none'
                        with open(self.dir_live_chessboard_situation, "w") as t_p:
                            yaml.dump(live_situation, t_p)

            else: #If the action is to manually place a piece in the chessboard.
                #Save the color of the square
                print('SQUARE OBJ: '+str(square_object))
                square_color = self.chessboard_colors[square]

                #Save the pieces that are currently in the live situation
                present_bishops_w = []
                present_rooks_w = []
                present_knights_w = []
                present_pawns_w = []
                present_bishops_b = []
                present_rooks_b = []
                present_knights_b = []
                present_pawns_b = []

                for casella in live_situation:
                    if casella[0] == 'bishop_c1' or casella[0] == 'bishop_f1':
                        present_bishops_w.append(casella[0])
                    elif casella[0] == 'bishop_c8' or casella[0] == 'bishop_f8':
                        present_bishops_b.append(casella[0])
                    elif casella[0] == 'knight_b1' or casella[0] == 'knight_g1':
                        present_knights_w.append(casella[0])
                    elif casella[0] == 'knight_b8' or casella[0] == 'knight_g8':
                        present_knights_b.append(casella[0])
                    elif casella[0] == 'rook_a1' or casella[0] == 'rook_h1':
                        present_rooks_w.append(casella[0])
                    elif casella[0] == 'rook_a8' or casella[0] == 'rook_h8':
                        present_rooks_b.append(casella[0])
                    elif casella[0] == 'pawn_a2' or casella[0] == 'pawn_b2' or casella[0] == 'pawn_c2' or casella[0] == 'pawn_d2' or casella[0] == 'pawn_e2' or casella[0] == 'pawn_f2' or casella[0] == 'pawn_g2' or casella[0] == 'pawn_h2':
                        present_pawns_w.append(casella[0])
                    elif casella[0] == 'pawn_a7' or casella[0] == 'pawn_b7' or casella[0] == 'pawn_c7' or casella[0] == 'pawn_d7' or casella[0] == 'pawn_e7' or casella[0] == 'pawn_f7' or casella[0] == 'pawn_g7' or casella[0] == 'pawn_h7':
                        present_pawns_b.append(casella[0])

                if square_color == 'white':
                    if action == 'w_pawn':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_white[0])
                        #Change the live chessboard situation
                        if square_object in pawns_w:
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'white'
                        else:
                            if len(present_pawns_w) == 0:
                                live_situation[square][0] = 'pawn_a2'
                                live_situation[square][1] = 'white'
                            elif len(present_pawns_w) == 8:
                                live_situation[square][0] = 'inserted_pawn'
                                live_situation[square][1] = 'white'
                            else:
                                self.white_pawns = cfg_w.white_pawns
                                for pawn in self.white_pawns:
                                    if pawn not in present_pawns_w:
                                        live_situation[square][0] = pawn
                                        live_situation[square][1] = 'white'
                                        break
                    elif action == 'b_pawn':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_white[1])
                        #Change the live chessboard situation
                        if square_object in pawns_b:
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'black'
                        else:
                            if len(present_pawns_b) == 0:
                                live_situation[square][0] = 'pawn_a7'
                                live_situation[square][1] = 'black'
                            elif len(present_pawns_b) == 8:
                                live_situation[square][0] = 'inserted_pawn'
                                live_situation[square][1] = 'black'
                            else:
                                self.black_pawns = cfg_w.black_pawns
                                for pawn in self.black_pawns:
                                    if pawn not in present_pawns_b:
                                        live_situation[square][0] = pawn
                                        live_situation[square][1] = 'black'
                                        break
                    elif action == 'w_knight':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_white[4])
                        #Change the live chessboard situation
                        if square_object == 'knight_g1' or square_object == 'knight_b1':
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'white'
                        else:
                            if len(present_knights_w) == 1:
                                live_situation[square][0] = present_knights_w[0]
                                live_situation[square][1] = 'white'
                            elif len(present_knights_w) == 2:
                                live_situation[square][0] = 'inserted_knight'
                                live_situation[square][1] = 'white'
                            elif len(present_knights_w) == 0:
                                live_situation[square][0] = 'knight_b1'
                                live_situation[square][1] = 'white'                    
                    elif action == 'b_knight':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_white[5])
                        #Change the live chessboard situation
                        if square_object == 'knight_g8' or square_object == 'knight_b8':
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'black'
                        else:
                            if len(present_knights_b) == 1:
                                live_situation[square][0] = present_knights_b[0]
                                live_situation[square][1] = 'black'
                            elif len(present_knights_b) == 2:
                                live_situation[square][0] = 'inserted_knight'
                                live_situation[square][1] = 'black'
                            elif len(present_knights_b) == 0:
                                live_situation[square][0] = 'knight_b8'
                                live_situation[square][1] = 'black'                        
                    elif action == 'w_bishop':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_white[6])
                        #Change the live chessboard situation
                        if square_object == 'bishop_c1' or square_object == 'bishop_f1':
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'white'
                        else:
                            if len(present_bishops_w) == 1:
                                live_situation[square][0] = present_bishops_w[0]
                                live_situation[square][1] = 'white'
                            elif len(present_bishops_w) == 2:
                                live_situation[square][0] = 'inserted_bishop'
                                live_situation[square][1] = 'white'
                            elif len(present_bishops_w) == 0:
                                live_situation[square][0] = 'bishop_c1'
                                live_situation[square][1] = 'white'                        
                    elif action == 'b_bishop':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_white[7])
                        #Change the live chessboard situation
                        if square_object == 'bishop_c8' or square_object == 'bishop_f8':
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'black'
                        else:
                            if len(present_bishops_b) == 1:
                                live_situation[square][0] = present_bishops_b[0]
                                live_situation[square][1] = 'black'
                            elif len(present_bishops_b) == 2:
                                live_situation[square][0] = 'inserted_bishop'
                                live_situation[square][1] = 'black'
                            elif len(present_bishops_b) == 0:
                                live_situation[square][0] = 'bishop_c8'
                                live_situation[square][1] = 'black'                        
                    elif action == 'w_rook':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_white[2])
                        #Change the live chessboard situation
                        if square_object == 'rook_a1' or square_object == 'rook_h1':
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'white'
                        else:
                            if len(present_rooks_w) == 1:
                                live_situation[square][0] = present_rooks_w[0]
                                live_situation[square][1] = 'white'
                            elif len(present_rooks_w) == 2:
                                live_situation[square][0] = 'inserted_rook'
                                live_situation[square][1] = 'white'
                            elif len(present_rooks_w) == 0:
                                live_situation[square][0] = 'rook_a1'
                                live_situation[square][1] = 'white'                        
                    elif action == 'b_rook':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_white[3])
                        #Change the live chessboard situation
                        if square_object == 'rook_a8' or square_object == 'rook_h8':
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'black'
                        else:
                            if len(present_rooks_b) == 1:
                                live_situation[square][0] = present_rooks_b[0]
                                live_situation[square][1] = 'black'
                            elif len(present_rooks_b) == 2:
                                live_situation[square][0] = 'inserted_rook'
                                live_situation[square][1] = 'black'
                            elif len(present_rooks_b) == 0:
                                live_situation[square][0] = 'rook_a8'
                                live_situation[square][1] = 'black'                       
                    elif action == 'w_queen':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_white[8])
                        #Change the live chessboard situation
                        live_situation[square][0] = 'queen_d1'
                        live_situation[square][1] = 'white'                        
                    elif action == 'b_queen':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_white[9])
                        #Change the live chessboard situation
                        live_situation[square][0] = 'queen_d8'
                        live_situation[square][1] = 'black'                        
                    elif action == 'w_king':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_white[10])
                        #Change the live chessboard situation
                        live_situation[square][0] = 'king_e1'
                        live_situation[square][1] = 'white'                        
                    elif action == 'b_king':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_white[11])
                        #Change the live chessboard situation
                        live_situation[square][0] = 'king_e8'
                        live_situation[square][1] = 'black'                        
                elif square_color == 'black':
                    if action == 'w_pawn':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_black[0])
                        #Change the live chessboard situation
                        if square_object in pawns_w:
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'white'
                        else:
                            if len(present_pawns_w) == 0:
                                live_situation[square][0] = 'pawn_a2'
                                live_situation[square][1] = 'white'
                            elif len(present_pawns_w) == 8:
                                live_situation[square][0] = 'inserted_pawn'
                                live_situation[square][1] = 'white'
                            else:
                                self.white_pawns = cfg_w.white_pawns
                                for pawn in self.white_pawns:
                                    if pawn not in present_pawns_w:
                                        live_situation[square][0] = pawn
                                        live_situation[square][1] = 'white'
                                        break
                    elif action == 'b_pawn':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_black[1])
                        #Change the live chessboard situation
                        if square_object in pawns_b:
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'black'
                        else:
                            if len(present_pawns_b) == 0:
                                live_situation[square][0] = 'pawn_a7'
                                live_situation[square][1] = 'black'
                            elif len(present_pawns_b) == 8:
                                live_situation[square][0] = 'inserted_pawn'
                                live_situation[square][1] = 'black'
                            else:
                                self.black_pawns = cfg_w.black_pawns
                                for pawn in self.black_pawns:
                                    if pawn not in present_pawns_b:
                                        live_situation[square][0] = pawn
                                        live_situation[square][1] = 'black'
                                        break
                    elif action == 'w_knight':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_black[4])
                        #Change the live chessboard situation
                        if square_object == 'knight_g1' or square_object == 'knight_b1':
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'white'
                        else:
                            if len(present_knights_w) == 1:
                                live_situation[square][0] = present_knights_w[0]
                                live_situation[square][1] = 'white'
                            elif len(present_knights_w) == 2:
                                live_situation[square][0] = 'inserted_knight'
                                live_situation[square][1] = 'white'
                            elif len(present_knights_w) == 0:
                                live_situation[square][0] = 'knight_b1'
                                live_situation[square][1] = 'white'                    
                    elif action == 'b_knight':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_black[5])
                        #Change the live chessboard situation
                        if square_object == 'knight_g8' or square_object == 'knight_b8':
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'black'
                        else:
                            if len(present_knights_b) == 1:
                                live_situation[square][0] = present_knights_b[0]
                                live_situation[square][1] = 'black'
                            elif len(present_knights_b) == 2:
                                live_situation[square][0] = 'inserted_knight'
                                live_situation[square][1] = 'black'
                            elif len(present_knights_b) == 0:
                                live_situation[square][0] = 'knight_b8'
                                live_situation[square][1] = 'black'                        
                    elif action == 'w_bishop':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_black[6])
                        #Change the live chessboard situation
                        if square_object == 'bishop_c1' or square_object == 'bishop_f1':
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'white'
                        else:
                            if len(present_bishops_w) == 1:
                                live_situation[square][0] = present_bishops_w[0]
                                live_situation[square][1] = 'white'
                            elif len(present_bishops_w) == 2:
                                live_situation[square][0] = 'inserted_bishop'
                                live_situation[square][1] = 'white'
                            elif len(present_bishops_w) == 0:
                                live_situation[square][0] = 'bishop_c1'
                                live_situation[square][1] = 'white'                        
                    elif action == 'b_bishop':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_black[7])
                        #Change the live chessboard situation
                        if square_object == 'bishop_c8' or square_object == 'bishop_f8':
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'black'
                        else:
                            if len(present_bishops_b) == 1:
                                live_situation[square][0] = present_bishops_b[0]
                                live_situation[square][1] = 'black'
                            elif len(present_bishops_b) == 2:
                                live_situation[square][0] = 'inserted_bishop'
                                live_situation[square][1] = 'black'
                            elif len(present_bishops_b) == 0:
                                live_situation[square][0] = 'bishop_c8'
                                live_situation[square][1] = 'black'                        
                    elif action == 'w_rook':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_black[2])
                        #Change the live chessboard situation
                        if square_object == 'rook_a1' or square_object == 'rook_h1':
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'white'
                        else:
                            if len(present_rooks_w) == 1:
                                live_situation[square][0] = present_rooks_w[0]
                                live_situation[square][1] = 'white'
                            elif len(present_rooks_w) == 2:
                                live_situation[square][0] = 'inserted_rook'
                                live_situation[square][1] = 'white'
                            elif len(present_rooks_w) == 0:
                                live_situation[square][0] = 'rook_a1'
                                live_situation[square][1] = 'white'                        
                    elif action == 'b_rook':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_black[3])
                        #Change the live chessboard situation
                        if square_object == 'rook_a8' or square_object == 'rook_h8':
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'black'
                        else:
                            if len(present_rooks_b) == 1:
                                live_situation[square][0] = present_rooks_b[0]
                                live_situation[square][1] = 'black'
                            elif len(present_rooks_b) == 2:
                                live_situation[square][0] = 'inserted_rook'
                                live_situation[square][1] = 'black'
                            elif len(present_rooks_b) == 0:
                                live_situation[square][0] = 'rook_a8'
                                live_situation[square][1] = 'black'                       
                    elif action == 'w_queen':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_black[8])
                        #Change the live chessboard situation
                        live_situation[square][0] = 'queen_d1'
                        live_situation[square][1] = 'white'                        
                    elif action == 'b_queen':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_black[9])
                        #Change the live chessboard situation
                        live_situation[square][0] = 'queen_d8'
                        live_situation[square][1] = 'black'                        
                    elif action == 'w_king':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_black[10])
                        #Change the live chessboard situation
                        live_situation[square][0] = 'king_e1'
                        live_situation[square][1] = 'white'                        
                    elif action == 'b_king':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_black[11])
                        #Change the live chessboard situation
                        live_situation[square][0] = 'king_e8'
                        live_situation[square][1] = 'black'
                with open(self.dir_live_chessboard_situation, "w") as t_p:
                        yaml.dump(live_situation, t_p)

    def Promotion(self, data):
        global casella_end
        global casella_start
        self.promoted_piece_msg = data.data
        self.promotion_happened = True

        #destination_color_promotion = self.chessboard_colors[self.opponent_move_end_square]
        with open(self.dir_live_chessboard_situation, 'rb') as live_file:
            live_situation = yaml.load(live_file.read())

        if state == 16 or state == 17 or state == 11: #Promotion is executed by the opponent
            casella_end = self.opponent_move_end_square
            casella_start = self.opponent_move_start_square
            destination_color_promotion = self.chessboard_colors[self.opponent_move_end_square]
        elif state == 12: #Promotion is executed by TIAGo.
            casella_end = self.current_focus
            casella_start = self.current_red_icon
            destination_color_promotion = self.chessboard_colors[casella_end]

        if self.color_to_move == 'white' and destination_color_promotion == 'white':
            if self.promoted_piece_msg == 'rook':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_white[2])
                    live_situation[casella_end][0] = 'rook_a1'
            elif self.promoted_piece_msg == 'knight':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_white[4])
                    live_situation[casella_end][0] = 'knight_b1'
            elif self.promoted_piece_msg == 'bishop':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_white[6])
                    live_situation[casella_end][0] = 'bishop_c1'
            elif self.promoted_piece_msg == 'queen':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_white[8])
                    live_situation[casella_end][0] = 'queen_d1'
            self.close_promotion.emit(1)

        elif self.color_to_move == 'white' and destination_color_promotion == 'black':
            if self.promoted_piece_msg == 'rook':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_black[2])
                    live_situation[casella_end][0] = 'rook_a1'
            elif self.promoted_piece_msg == 'knight':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_black[4])
                    live_situation[casella_end][0] = 'knight_b1'
            elif self.promoted_piece_msg == 'bishop':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_black[6])
                    live_situation[casella_end][0] = 'bishop_c1'
            elif self.promoted_piece_msg == 'queen':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_black[8])
                    live_situation[casella_end][0] = 'queen_d1'
            self.close_promotion.emit(1)

        elif self.color_to_move == 'black' and destination_color_promotion == 'white':
            if self.promoted_piece_msg == 'rook':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_white[3])
                    live_situation[casella_end][0] = 'rook_a8'
            elif self.promoted_piece_msg == 'knight':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_white[5])
                    live_situation[casella_end][0] = 'knight_b8'
            elif self.promoted_piece_msg == 'bishop':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_white[7])
                    live_situation[casella_end][0] = 'bishop_c8'
            elif self.promoted_piece_msg == 'queen':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_white[9])
                    live_situation[casella_end][0] = 'queen_d8'
            self.close_promotion.emit(2)

        elif self.color_to_move == 'black' and destination_color_promotion == 'black':
            if self.promoted_piece_msg == 'rook':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_black[3])
                    live_situation[casella_end][0] = 'rook_a8'
            elif self.promoted_piece_msg == 'knight':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_black[5])
                    live_situation[casella_end][0] = 'knight_b8'
            elif self.promoted_piece_msg == 'bishop':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_black[7])
                    live_situation[casella_end][0] = 'bishop_c8'
            elif self.promoted_piece_msg == 'queen':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_black[9])
                    live_situation[casella_end][0] = 'queen_d8'
            self.close_promotion.emit(2)

        live_situation[casella_end][1] = self.color_to_move

        with open(self.dir_live_chessboard_situation, "w") as t_p:
                    yaml.dump(live_situation, t_p)
        #######
        #Update the planning scene.
        with open(simul_config) as file:
            square_centers = yaml.load(file)
        self.populate_pieces(square_centers, live_situation)
        ###########

        self.promotion_happened = False

    def PromotionHappened(self, data):
        #Change the promotion flag
        self.opponent_promotion_happened = True
        self.color_to_move = data.data
        #Make the user choose the promoted piece
        if self.color_to_move == 'white':
            self.promotion.emit(1)
        elif self.color_to_move == 'black':
            self.promotion.emit(2)

    def OpponentMoveStartSquare(self, data):
        self.opponent_move_start_square = data.data
        state_publisher.publish(17) #Pass to the state of GUI updating.

        if self.opponent_move_start_square == 'castle':
            pass

        else:
            #Save the color of the starting square
            starting_color = self.chessboard_colors[self.opponent_move_start_square]

            #Change the starting square icon
            if starting_color == 'white':
                self.array[self.dict_squares[self.opponent_move_start_square]].setIcon(QtGui.QIcon())
            elif starting_color == 'black':
                self.array[self.dict_squares[self.opponent_move_start_square]].setIcon(self.icon_grey)

    def OpponentMoveEndSquare(self, data):
        self.opponent_move_end_square = data.data

        rospy.sleep(1)

        if self.opponent_move_end_square == 'castle':
            #Change the GUI icons.
            self.array[self.dict_squares[self.castle_squares[0]]].setIcon(self.icons_array_white[11])
            self.array[self.dict_squares[self.castle_squares[1]]].setIcon(self.icons_array_black[3])
            self.array[self.dict_squares['e8']].setIcon(QtGui.QIcon())
            if self.castle_squares[1] == 'f8':
                self.array[self.dict_squares['h8']].setIcon(self.icon_grey)
            elif self.castle_squares[1] == 'd8':
                self.array[self.dict_squares['a8']].setIcon(self.icon_grey)

            #Update the live chessboard situation
            with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                    live_situation = yaml.load(live_file.read())
            if self.castle_squares[0] == 'c8':
                self.array[self.dict_squares['a8']].setIcon(QtGui.QIcon())
                live_situation['c8'][0] = 'king_e8'
                live_situation['c8'][1] = 'black'
                live_situation['d8'][0] = 'rook_a8'
                live_situation['d8'][1] = 'black'
                live_situation['a8'][0] = 'none'
                live_situation['a8'][1] = 'none'
                live_situation['e8'][0] = 'none'
                live_situation['e8'][1] = 'none'
            elif self.castle_squares[0] == 'g8':
                self.array[self.dict_squares['h1']].setIcon(self.icon_grey)
                live_situation['g8'][0] = 'king_e8'
                live_situation['g8'][1] = 'black'
                live_situation['f8'][0] = 'rook_h8'
                live_situation['f8'][1] = 'black'
                live_situation['h8'][0] = 'none'
                live_situation['h8'][1] = 'none'
                live_situation['e8'][0] = 'none'
                live_situation['e8'][1] = 'none'

            with open(self.dir_live_chessboard_situation, "w") as t_p:
                    yaml.dump(live_situation, t_p)
            #######
            #Update the planning scene.
            with open(simul_config) as file:
                square_centers = yaml.load(file)
            self.populate_pieces(square_centers, live_situation)
            ###########
            if state != 16:
                state_publisher.publish(11) #Go back to the state of waiting for the move to perform with TIAGo.

        else:
            #Read which piece is in the currently highlighted square
            with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                    live_situation = yaml.load(live_file.read())
            self.piece_to_move = self.pieces_coordinates[live_situation[self.opponent_move_start_square][0]][1]['name']
            piece_to_capture = live_situation[self.opponent_move_end_square][0]
            self.color_to_move = live_situation[self.opponent_move_start_square][1]
            #Save the color of the destination square and of the starting square
            destination_color = self.chessboard_colors[self.opponent_move_end_square]
            starting_color = self.chessboard_colors[self.opponent_move_start_square]

            #Change the destination icon and the starting square icon
            if destination_color == 'black':
                    if self.piece_to_move == 'pawn':
                        if self.color_to_move == 'white':
                                self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_black[0])
                        elif self.color_to_move == 'black':
                                self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_black[1])
                    elif self.piece_to_move == 'rook':
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_black[2])
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_black[3])
                    elif self.piece_to_move == 'knight':
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_black[4])
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_black[5])
                    elif self.piece_to_move == 'bishop':
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_black[6])
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_black[7])
                    elif self.piece_to_move == 'queen':
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_black[8])
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_black[9])
                    elif self.piece_to_move == 'king':
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_black[10])
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_black[11])

            elif destination_color == 'white':
                    if self.piece_to_move == 'pawn':
                        if not self.opponent_promotion_happened:
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_white[0])
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_white[1])
                        elif self.opponent_promotion_happened:
                            self.opponent_promotion_happened = False #Set the flag back to false
                    elif self.piece_to_move == 'rook':
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_white[2])
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_white[3])
                    elif self.piece_to_move == 'knight':
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_white[4])
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_white[5])
                    elif self.piece_to_move == 'bishop':
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_white[6])
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_white[7])
                    elif self.piece_to_move == 'queen':
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_white[8])
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_white[9])
                    elif self.piece_to_move == 'king':
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_white[10])
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_white[11])

            #Update the live chessboard situation
            live_situation[self.opponent_move_end_square][0] = live_situation[self.opponent_move_start_square][0]
            if self.color_to_move == 'white':
                live_situation[self.opponent_move_end_square][1] = 'white'
            elif self.color_to_move == 'black':
                live_situation[self.opponent_move_end_square][1] = 'black'

            live_situation[self.opponent_move_start_square][0] = 'none'
            live_situation[self.opponent_move_start_square][1] = 'none'

            if self.en_passant_happened:
                #Save the color of the starting square
                square_color = self.chessboard_colors[self.en_passant_square]
                #Change the starting square icon
                if square_color == 'white':
                    self.array[self.dict_squares[self.en_passant_square]].setIcon(QtGui.QIcon())
                elif square_color == 'black':
                    self.array[self.dict_squares[self.en_passant_square]].setIcon(self.icon_grey)
                #Update the live chessboard situation
                live_situation[self.en_passant_square][0] = 'none'
                live_situation[self.en_passant_square][1] = 'none'
                self.en_passant_happened = False #Bring back the flag to False

            with open(self.dir_live_chessboard_situation, "w") as t_p:
                    yaml.dump(live_situation, t_p)
            #######
            #Update the planning scene.
            with open(simul_config) as file:
                square_centers = yaml.load(file)
            self.populate_pieces(square_centers, live_situation)
            ###########
            if state != 16:
                state_publisher.publish(11) #Go back to the state of waiting for the move to perform with TIAGo.


    def populate_pieces(self, base_centers, live_chessboard_situation):
        #Function to populate the planning scene with the pieces in their current position.
        #It receives as input the coordinates of the squares' centeres in the base_footprint reference frame and live_chessboard_situation that keeps track of the pieces positions thoughout the game.
        #This function needs to be used whenever TIAGo sees that a piece has been moved from the opponent.

        #In case there is something in the world form previous executions (maybe the robot was closer or further from the chessboard)
        for name in self.pieces_coordinates:
            self.scene.remove_world_object(name)

        key_list = list(live_chessboard_situation.keys())
        val = []
        for elem in key_list:
            val.append(live_chessboard_situation[elem][0])
        val_list = list(val)

        #Add cylinders corresponding to the pieces to the planning scene.
        print('ADDING PIECES TO PLANNING SCENE') ###
        for name in self.pieces_coordinates:
            if name in val_list:
                position = val_list.index(name) #Get the position (index) of the current piece in the chessboard
                square = key_list[position] #Get the square where the current piece is.
                pose = PoseStamped()
                pose.header.frame_id = 'base_footprint'
                pose.header.stamp = rospy.Time.now()
                pose.pose = Pose(Point(base_centers[self.squares_to_index[square]].point.x, base_centers[self.squares_to_index[square]].point.y, base_centers[self.squares_to_index[square]].point.z + self.pieces_coordinates[name][1]['height']/2), Quaternion(0, 0, 0, 1))
                print('position: '+str(position))###
                print('square: '+str(square))###
                self.scene.add_cylinder(name, pose, height = self.pieces_coordinates[name][1]['height'], radius = self.pieces_coordinates[name][1]['diameter']/2)

    def EnPassant(self, data):
        self.en_passant_square = data.data
        self.en_passant_happened = True

    def Castle(self, data):
        self.castle_square = data.data
        if self.castle_square == 'c8' or self.castle_square == 'd8':
            self.castle_squares = ['c8', 'd8']
        elif self.castle_square == 'g8' or self.castle_square == 'f8':
            self.castle_squares = ['g8', 'f8']

    def UpdateLiveSituation(self, data):
        print('Update flag: '+str(update_flag))
        if data.data == 14 and update_flag: #State of waiting for the opponent move.
            print('UPDATING............')
            #Update the live chessboard situation with the move just executed by TIAGo.
            with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                live_situation = yaml.load(live_file.read())

            if not self.promotion_happened:
                live_situation[self.current_focus][0] = live_situation[self.current_red_icon][0]
                if self.color_to_move == 'white':
                    live_situation[self.current_focus][1] = 'white'
                elif self.color_to_move == 'black':
                    live_situation[self.current_focus][1] = 'black'
                if self.en_passant:
                        live_situation[square_captured_piece][0] = 'none'
                        live_situation[square_captured_piece][1] = 'none'
                if self.short_castle:
                        live_situation['f1'][0] = 'rook_h1'
                        live_situation['f1'][1] = 'white'
                        live_situation['h1'][0] = 'none'
                        live_situation['h1'][1] = 'none'
                if self.long_castle:
                        live_situation['d1'][0] = 'rook_a1'
                        live_situation['d1'][1] = 'white'
                        live_situation['a1'][0] = 'none'
                        live_situation['a1'][1] = 'none'
                live_situation[self.current_red_icon][0] = 'none'
                live_situation[self.current_red_icon][1] = 'none'
                '''
                elif self.promotion_happened:
                    #if self.color_to_move == 'white':
                    if self.promoted_piece_msg == 'rook':
                            live_situation[casella_end][0] = 'rook_a8'
                            live_situation[casella_end][1] = 'black'
                            live_situation[casella_start][0] = 'none'
                            live_situation[casella_start][1] = 'none'
                    elif self.promoted_piece_msg == 'knight':
                            live_situation[casella_end][0] = 'knight_b8'
                            live_situation[casella_end][1] = 'black'
                            live_situation[casella_start][0] = 'none'
                            live_situation[casella_start][1] = 'none'
                    elif self.promoted_piece_msg == 'bishop':
                            live_situation[casella_end][0] = 'bishop_c8'
                            live_situation[casella_end][1] = 'black'
                            live_situation[casella_start][0] = 'none'
                            live_situation[casella_start][1] = 'none'
                    elif self.promoted_piece_msg == 'queen':
                            live_situation[casella_end][0] = 'queen_d8'
                            live_situation[casella_end][1] = 'black'
                            live_situation[casella_start][0] = 'none'
                            live_situation[casella_start][1] = 'none'
                '''
                self.promotion_happened = False #Change back the flag

            self.promotion_happened = False #Change back the flag

            with open(self.dir_live_chessboard_situation, "w") as t_p:
                yaml.dump(live_situation, t_p)

            #######
            #Update the planning scene.
            with open(simul_config) as file:
                square_centers = yaml.load(file)
            self.populate_pieces(square_centers, live_situation)
            ###########

    def ManualMode(self):
        self.PlayGameModePushbutton.setEnabled(True)
        self.ManualModePushbutton.setEnabled(False)
        self.manual_mode.emit(1)
        state_publisher.publish(70)

    def BackToGame(self):
        self.PlayGameModePushbutton.setEnabled(False)
        self.ManualModePushbutton.setEnabled(True)
        self.back_to_game.emit(1)
        state_publisher.publish(11)


###################################################################


class ChessboardBlackWindow(QtWidgets.QWidget, Ui_Chessboard_Black):
    #Signals initialization
    clicked = pyqtSignal(object)
    promotion = QtCore.pyqtSignal(int)
    close_promotion = QtCore.pyqtSignal(int)
    manual_mode = QtCore.pyqtSignal(int)
    back_to_game = QtCore.pyqtSignal(int)
    global update_flag

    def __init__(self):
        super(ChessboardBlackWindow, self).__init__()
        self.app = MyApplication(sys.argv)
        self.app_focus = self.app.focusWidget() 
        self.setupUi(self, self.app_focus)

        #Subscribers initialization
        rospy.Subscriber("/cursor_step_eleven", Point, self.MoveHighlight)
        rospy.Subscriber("/button_eleven", Bool, self.SelectPiece)
        rospy.Subscriber("/promotion_happened", String, self.PromotionHappened)
        rospy.Subscriber("/promoted_piece", String, self.Promotion)
        rospy.Subscriber("/opponent_move_start_square", String, self.OpponentMoveStartSquare)
        rospy.Subscriber("/opponent_move_end_square", String, self.OpponentMoveEndSquare)
        rospy.Subscriber("/en_passant_square", String, self.EnPassant)
        rospy.Subscriber("/castle_square", String, self.Castle)
        rospy.Subscriber('/state', Int16, self.UpdateLiveSituation)

        #Publishers initialization
        self.start_square_move_publisher = rospy.Publisher('/start_square', String, queue_size = 10)
        self.end_square_move_publisher = rospy.Publisher('/end_square', String, queue_size = 10)

        #Variables initialization
        self.promoted_piece_msg = 'none'
        self.opponent_promotion_happened = False
        self.promotion_happened = False

        #Import configurations to map the chessboard squares
        self.dict_squares = cfg_b.chessboard_numbers
        self.pieces_coordinates = cfg_b.pieces_coordinates
        self.chessboard_colors = cfg_b.chessboard_colors
        self.chessboard_matrix = cfg_b.matrix
        self.array = [self.h8, self.g8, self.f8, self.e8, self.d8, self.c8, self.b8, self.a8, self.h7, self.g7, self.f7, self.e7,
        self.d7, self.c7, self.b7, self.a7, self.h6, self.g6, self.f6, self.e6, self.d6, self.c6, self.b6, self.a6, self.h5, self.g5,
        self.f5, self.e5, self.d5, self.c5, self.b5, self.a5, self.h4, self.g4, self.f4, self.e4, self.d4, self.c4, self.b4, self.a4,
        self.h3, self.g3, self.f3, self.e3, self.d3, self.c3, self.b3, self.a3, self.h2, self.g2, self.f2, self.e2, self.d2, self.c2,
        self.b2, self.a2, self.h1, self.g1, self.f1, self.e1, self.d1, self.c1, self.b1, self.a1]

        self.icons_array_black = [self.icon_Wpawn_b, self.icon_Bpawn_b, self.icon_Wrook_b, self.icon_Brook_b, self.icon_Wknight_b,
        self.icon_Bknight_b, self.icon_Wbishop_b, self.icon_Bbishop_b, self.icon_Wqueen_b, self.icon_Bqueen_b, self.icon_Wking_b, self.icon_Bking_b]
        self.icons_array_white = [self.icon_Wpawn_w, self.icon_Bpawn_w, self.icon_Wrook_w, self.icon_Brook_w, self.icon_Wknight_w,
        self.icon_Bknight_w, self.icon_Wbishop_w, self.icon_Bbishop_w, self.icon_Wqueen_w, self.icon_Bqueen_w, self.icon_Wking_w, self.icon_Bking_w]
        self.icons_array_yellow = [self.icon_Wpawn_y, self.icon_Bpawn_y, self.icon_Wrook_y, self.icon_Brook_y, self.icon_Wknight_y,
        self.icon_Bknight_y, self.icon_Wbishop_y, self.icon_Bbishop_y, self.icon_Wqueen_y, self.icon_Bqueen_y, self.icon_Wking_y, self.icon_Bking_y]

        #Selection flag initializiation
        self.yellow_selection_flag = False
        self.selection_flag = False

        #Focus changed flag initialization
        self.already_changed = False

        #Opponent en-passant happened flag
        self.en_passant_happened = False

        #Save the live chessboard situation yaml file directory
        self.dir_live_chessboard_situation = PLAYCHESS_PKG_DIR + "/scripts/live_chessboard_situation.yaml" #TO CHANGE IF THE PC IS CHANGED

        self.pieces_coordinates = cfg.pieces_coordinates
        self.squares_to_index = cfg.squares_to_index_black
        self.scene = moveit_commander.PlanningSceneInterface() #The interface with the world surrounding the robot

        self.h8.clicked.connect(lambda: self.click_and_focus('h8'))
        self.g8.clicked.connect(lambda: self.click_and_focus('g8'))
        self.f8.clicked.connect(lambda: self.click_and_focus('f8'))
        self.e8.clicked.connect(lambda: self.click_and_focus('e8'))
        self.d8.clicked.connect(lambda: self.click_and_focus('d8'))
        self.c8.clicked.connect(lambda: self.click_and_focus('c8'))
        self.b8.clicked.connect(lambda: self.click_and_focus('b8'))
        self.a8.clicked.connect(lambda: self.click_and_focus('a8'))
        self.h7.clicked.connect(lambda: self.click_and_focus('h7'))
        self.g7.clicked.connect(lambda: self.click_and_focus('g7'))
        self.f7.clicked.connect(lambda: self.click_and_focus('f7'))
        self.e7.clicked.connect(lambda: self.click_and_focus('e7'))
        self.d7.clicked.connect(lambda: self.click_and_focus('d7'))
        self.c7.clicked.connect(lambda: self.click_and_focus('c7'))
        self.b7.clicked.connect(lambda: self.click_and_focus('b7'))
        self.a7.clicked.connect(lambda: self.click_and_focus('a7'))
        self.h6.clicked.connect(lambda: self.click_and_focus('h6'))
        self.g6.clicked.connect(lambda: self.click_and_focus('g6'))
        self.f6.clicked.connect(lambda: self.click_and_focus('f6'))
        self.e6.clicked.connect(lambda: self.click_and_focus('e6'))
        self.d6.clicked.connect(lambda: self.click_and_focus('d6'))
        self.c6.clicked.connect(lambda: self.click_and_focus('c6'))
        self.b6.clicked.connect(lambda: self.click_and_focus('b6'))
        self.a6.clicked.connect(lambda: self.click_and_focus('a6'))
        self.h5.clicked.connect(lambda: self.click_and_focus('h5'))
        self.g5.clicked.connect(lambda: self.click_and_focus('g5'))
        self.f5.clicked.connect(lambda: self.click_and_focus('f5'))
        self.e5.clicked.connect(lambda: self.click_and_focus('e5'))
        self.d5.clicked.connect(lambda: self.click_and_focus('d5'))
        self.c5.clicked.connect(lambda: self.click_and_focus('c5'))
        self.b5.clicked.connect(lambda: self.click_and_focus('b5'))
        self.a5.clicked.connect(lambda: self.click_and_focus('a5'))
        self.h4.clicked.connect(lambda: self.click_and_focus('h4'))
        self.g4.clicked.connect(lambda: self.click_and_focus('g4'))
        self.f4.clicked.connect(lambda: self.click_and_focus('f4'))
        self.e4.clicked.connect(lambda: self.click_and_focus('e4'))
        self.d4.clicked.connect(lambda: self.click_and_focus('d4'))
        self.c4.clicked.connect(lambda: self.click_and_focus('c4'))
        self.b4.clicked.connect(lambda: self.click_and_focus('b4'))
        self.a4.clicked.connect(lambda: self.click_and_focus('a4'))
        self.h3.clicked.connect(lambda: self.click_and_focus('h3'))
        self.g3.clicked.connect(lambda: self.click_and_focus('g3'))
        self.f3.clicked.connect(lambda: self.click_and_focus('f3'))
        self.e3.clicked.connect(lambda: self.click_and_focus('e3'))
        self.d3.clicked.connect(lambda: self.click_and_focus('d3'))
        self.c3.clicked.connect(lambda: self.click_and_focus('c3'))
        self.b3.clicked.connect(lambda: self.click_and_focus('b3'))
        self.a3.clicked.connect(lambda: self.click_and_focus('a3'))
        self.h2.clicked.connect(lambda: self.click_and_focus('h2'))
        self.g2.clicked.connect(lambda: self.click_and_focus('g2'))
        self.f2.clicked.connect(lambda: self.click_and_focus('f2'))
        self.e2.clicked.connect(lambda: self.click_and_focus('e2'))
        self.d2.clicked.connect(lambda: self.click_and_focus('d2'))
        self.c2.clicked.connect(lambda: self.click_and_focus('c2'))
        self.b2.clicked.connect(lambda: self.click_and_focus('b2'))
        self.a2.clicked.connect(lambda: self.click_and_focus('a2'))
        self.h1.clicked.connect(lambda: self.click_and_focus('h1'))
        self.g1.clicked.connect(lambda: self.click_and_focus('g1'))
        self.f1.clicked.connect(lambda: self.click_and_focus('f1'))
        self.e1.clicked.connect(lambda: self.click_and_focus('e1'))
        self.d1.clicked.connect(lambda: self.click_and_focus('d1'))
        self.c1.clicked.connect(lambda: self.click_and_focus('c1'))
        self.b1.clicked.connect(lambda: self.click_and_focus('b1'))
        self.a1.clicked.connect(lambda: self.click_and_focus('a1'))
        ###
        self.ManualModePushbutton.clicked.connect(self.ManualMode)
        self.PlayGameModePushbutton.clicked.connect(self.BackToGame)
        ###

    def MoveHighlight(self, data):
        #Whenever a joystick movement is detected, a motion of the highlights pushbutton on the GUI is performed
        #If a movement in both the directions is captured, ignore it.
        if data.x != 0 and data.y != 0:
                pass

        #If a movement in the x direction, understand its direction and perform the corresponding movement on the GUI.
        if data.x != 0 and data.y == 0:
                print('MOVE HIGHLIGHT')
                for row in self.chessboard_matrix:
                        if self.current_focus in row: #str(focus) in row:
                                row_index = self.chessboard_matrix.index(row)
                                col_index = row.index(self.current_focus)

                if data.x == 1:
                        #Highlight the pushbutton to the right
                        col_index += 1
                        if col_index == 8:
                                if row_index == 7:
                                        rospy.logwarn('The square does not exist')
                                else:
                                        col_index = 0
                                        row_index += 1 
                elif data.x == 2:
                        #Highlight the pushbutton to the left
                        col_index -= 1
                        if col_index == -1:
                                if row_index == 0:
                                        rospy.logwarn('The square does not exist')
                                else:
                                        col_index = 7
                                        row_index -= 1

                #Remove the highlight from the previous focused square, if the focus has already been changed at least one time.
                if (self.already_changed and not self.selection_flag) or (self.already_changed and self.selection_flag and self.current_focus != self.current_red_icon):
                    button_index = self.dict_squares[self.current_focus]
                    button = self.array[button_index]
                    
                    #Read if the focused square is occupied by one piece or if it's empty
                    with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                            live_situation = yaml.load(live_file.read())

                    if live_situation[self.current_focus][0] == 'none':
                        square_object = 'none'
                    else:
                        square_object = self.pieces_coordinates[live_situation[self.current_focus][0]][1]['name']
                        square_object_color = live_situation[self.current_focus][1]

                    if self.yellow_selection_flag: #If the square is yellow:
                        if self.self.piece_to_move == 'pawn':
                            if self.color_to_move == 'white':
                                button.setIcon(self.icon_Wpawn_y)
                            elif self.color_to_move == 'black':
                                button.setIcon(self.icon_Bpawn_y)
                        elif self.self.piece_to_move == 'rook':
                            if self.color_to_move == 'white':
                                button.setIcon(self.icon_Wrook_y)
                            elif self.color_to_move == 'black':
                                button.setIcon(self.icon_Brook_y)
                        elif self.self.piece_to_move == 'knight':
                            if self.color_to_move == 'white':
                                button.setIcon(self.icon_Wknight_y)
                            elif self.color_to_move == 'black':
                                button.setIcon(self.icon_Bknight_y)
                        elif self.self.piece_to_move == 'bishop':
                            if self.color_to_move == 'white':
                                button.setIcon(self.icon_Wbishop_y)
                            elif self.color_to_move == 'black':
                                button.setIcon(self.icon_Bbishop_y)
                        elif self.self.piece_to_move == 'queen':
                            if self.color_to_move == 'white':
                                button.setIcon(self.icon_Wqueen_y)
                            elif self.color_to_move == 'black':
                                button.setIcon(self.icon_Bqueen_y)
                        elif self.self.piece_to_move == 'king':
                            if self.color_to_move == 'white':
                                button.setIcon(self.icon_Wking_y)
                            elif self.color_to_move == 'black':
                                button.setIcon(self.icon_Bking_y)

                    elif self.chessboard_colors[self.current_focus] == 'white':
                        if square_object == 'pawn':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wpawn_w)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bpawn_w)
                        elif square_object == 'rook':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wrook_w)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Brook_w)
                        elif square_object == 'knight':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wknight_w)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bknight_w)
                        elif square_object == 'bishop':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wbishop_w)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bbishop_w)
                        elif square_object == 'queen':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wqueen_w)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bqueen_w)
                        elif square_object == 'king':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wking_w)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bking_w)
                        elif square_object == 'none':
                            button.setIcon(QtGui.QIcon())
                    elif self.chessboard_colors[self.current_focus] == 'black':
                        if square_object == 'pawn':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wpawn_b)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bpawn_b)
                        elif square_object == 'rook':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wrook_b)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Brook_b)
                        elif square_object == 'knight':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wknight_b)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bknight_b)
                        elif square_object == 'bishop':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wbishop_b)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bbishop_b)
                        elif square_object == 'queen':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wqueen_b)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bqueen_b)
                        elif square_object == 'king':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wking_b)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bking_b)
                        elif square_object == 'none':
                            button.setIcon(self.icon_grey)

                #Highlight the new pushbutton
                if row_index >= 0 and row_index <= 7 and col_index >= 0 and col_index <= 7:
                        self.current_focus = self.chessboard_matrix[row_index][col_index]
                        new_button_index = self.dict_squares[self.current_focus]
                        new_button = self.array[new_button_index]
                        new_button.setFocus()

                        #Read if the focused square is occupied by one piece or if it's empty
                        with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                                live_situation = yaml.load(live_file.read())

                        if live_situation[self.current_focus][0] == 'none':
                            square_object = 'none'
                        else:
                            square_object = self.pieces_coordinates[live_situation[self.current_focus][0]][1]['name']
                            square_object_color = live_situation[self.current_focus][1]

                        if self.chessboard_colors[self.current_focus] == 'white':
                            if square_object == 'pawn':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wpawn_w_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bpawn_w_frame)
                            elif square_object == 'rook':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wrook_w_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Brook_w_frame)
                            elif square_object == 'knight':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wknight_w_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bknight_w_frame)
                            elif square_object == 'bishop':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wbishop_w_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bbishop_w_frame)
                            elif square_object == 'queen':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wqueen_w_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bqueen_w_frame)
                            elif square_object == 'king':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wking_w_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bking_w_frame)
                            elif square_object == 'none':
                                new_button.setIcon(self.icon_white_frame)
                        elif self.chessboard_colors[self.current_focus] == 'black':
                            if square_object == 'pawn':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wpawn_b_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bpawn_b_frame)
                            elif square_object == 'rook':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wrook_b_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Brook_b_frame)
                            elif square_object == 'knight':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wknight_b_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bknight_b_frame)
                            elif square_object == 'bishop':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wbishop_b_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bbishop_b_frame)
                            elif square_object == 'queen':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wqueen_b_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bqueen_b_frame)
                            elif square_object == 'king':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wking_b_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bking_b_frame)
                            elif square_object == 'none':
                                new_button.setIcon(self.icon_grey_frame)

                        print('NEW FOCUS: ' + str(self.current_focus))
                        self.already_changed = True
                else:
                        rospy.logwarn('The square does not exist')

        #If a movement in the y direction, understand its direction and perform the corresponding movement on the GUI.
        if data.y != 0 and data.x == 0:
                #Save the row and the column index of the widget that currently has focus
                for row in self.chessboard_matrix:
                        if self.current_focus in row:
                                row_index = self.chessboard_matrix.index(row)
                                col_index = row.index(self.current_focus)
                if data.y == 1:
                        #Highlight the upper pushbutton
                        row_index += 1
                        if row_index == 8:
                                if col_index == 7:
                                        rospy.logwarn('The square does not exist')
                                else:
                                        row_index = 0
                                        col_index += 1
                elif data.y == 2:
                        #Highlight the lower pushbutton
                        row_index -= 1
                        if row_index == -1:
                                if col_index == 0:
                                        rospy.logwarn('The square does not exist')
                                else:
                                        row_index = 7
                                        col_index -= 1

                #Remove the highlight from the previous focused square, if the focus has already been changed at least one time.
                if (self.already_changed and not self.selection_flag) or (self.already_changed and self.selection_flag and self.current_focus != self.current_red_icon):
                    button_index = self.dict_squares[self.current_focus]
                    button_index = self.dict_squares[self.current_focus]
                    button = self.array[button_index]
                    
                    #Read if the focused square is occupied by one piece or if it's empty
                    with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                            live_situation = yaml.load(live_file.read())

                    if live_situation[self.current_focus][0] == 'none':
                        square_object = 'none'
                    else:
                        square_object = self.pieces_coordinates[live_situation[self.current_focus][0]][1]['name']
                        square_object_color = live_situation[self.current_focus][1]

                    if self.chessboard_colors[self.current_focus] == 'white':
                        if square_object == 'pawn':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wpawn_w)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bpawn_w)
                        elif square_object == 'rook':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wrook_w)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Brook_w)
                        elif square_object == 'knight':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wknight_w)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bknight_w)
                        elif square_object == 'bishop':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wbishop_w)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bbishop_w)
                        elif square_object == 'queen':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wqueen_w)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bqueen_w)
                        elif square_object == 'king':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wking_w)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bking_w)
                        elif square_object == 'none':
                            button.setIcon(QtGui.QIcon())
                    elif self.chessboard_colors[self.current_focus] == 'black':
                        if square_object == 'pawn':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wpawn_b)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bpawn_b)
                        elif square_object == 'rook':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wrook_b)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Brook_b)
                        elif square_object == 'knight':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wknight_b)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bknight_b)
                        elif square_object == 'bishop':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wbishop_b)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bbishop_b)
                        elif square_object == 'queen':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wqueen_b)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bqueen_b)
                        elif square_object == 'king':
                            if square_object_color == 'white':
                                button.setIcon(self.icon_Wking_b)
                            elif square_object_color == 'black':
                                button.setIcon(self.icon_Bking_b)
                        elif square_object == 'none':
                            button.setIcon(self.icon_grey)

                #Highlight the new pushbutton
                if row_index >= 0 and row_index <= 7 and col_index >= 0 and col_index <= 7:
                        self.current_focus = self.chessboard_matrix[row_index][col_index]
                        new_button_index = self.dict_squares[self.current_focus]
                        new_button = self.array[new_button_index]
                        new_button.setFocus()

                        #Read if the focused square is occupied by one piece or if it's empty
                        with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                                live_situation = yaml.load(live_file.read())

                        if live_situation[self.current_focus][0] == 'none':
                            square_object = 'none'
                        else:
                            square_object = self.pieces_coordinates[live_situation[self.current_focus][0]][1]['name']
                            square_object_color = live_situation[self.current_focus][1]

                        if self.chessboard_colors[self.current_focus] == 'white':
                            if square_object == 'pawn':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wpawn_w_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bpawn_w_frame)
                            elif square_object == 'rook':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wrook_w_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Brook_w_frame)
                            elif square_object == 'knight':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wknight_w_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bknight_w_frame)
                            elif square_object == 'bishop':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wbishop_w_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bbishop_w_frame)
                            elif square_object == 'queen':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wqueen_w_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bqueen_w_frame)
                            elif square_object == 'king':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wking_w_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bking_w_frame)
                            elif square_object == 'none':
                                new_button.setIcon(self.icon_white_frame)
                        elif self.chessboard_colors[self.current_focus] == 'black':
                            if square_object == 'pawn':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wpawn_b_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bpawn_b_frame)
                            elif square_object == 'rook':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wrook_b_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Brook_b_frame)
                            elif square_object == 'knight':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wknight_b_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bknight_b_frame)
                            elif square_object == 'bishop':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wbishop_b_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bbishop_b_frame)
                            elif square_object == 'queen':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wqueen_b_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bqueen_b_frame)
                            elif square_object == 'king':
                                if square_object_color == 'white':
                                    new_button.setIcon(self.icon_Wking_b_frame)
                                elif square_object_color == 'black':
                                    new_button.setIcon(self.icon_Bking_b_frame)
                            elif square_object == 'none':
                                new_button.setIcon(self.icon_grey_frame)

                        print('NEW FOCUS: ' + str(self.current_focus))
                        self.already_changed = True
                else:
                        rospy.logwarn('The square does not exist')

    def SelectPiece(self, data):
        global square_object
        global square_captured_piece
        if data:
            if state == 11: #If we are in game mode.
                #Change the current_focus variable.
                #self.current_focus = square
                print('CURRENT FOCUS: ' + str(self.current_focus))

                #If no other square is selected, select the square and change the color of it in red.
                if not self.selection_flag and not self.yellow_selection_flag:
                        #Read if the selected square is occupied by one piece or if it's empty.
                        with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                                live_situation = yaml.load(live_file.read())
                        square_object = live_situation[self.current_focus][0]
                        if square_object == 'none':
                                rospy.logwarn('There is no piece to move in the selected square')
                                self.selection_flag = not self.selection_flag
                        else:
                            self.color_to_move = live_situation[self.current_focus][1]
                            if self.pieces_coordinates[square_object][1]['name'] == 'pawn':
                                if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Wpawn)
                                elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Bpawn)
                            elif self.pieces_coordinates[square_object][1]['name'] == 'rook':
                                if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Wrook)
                                elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Brook)
                            elif self.pieces_coordinates[square_object][1]['name'] == 'knight':
                                if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Wknight)
                                elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Bknight)
                            elif self.pieces_coordinates[square_object][1]['name'] == 'bishop':
                                if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Wbishop)
                                elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Bbishop)
                            elif self.pieces_coordinates[square_object][1]['name'] == 'queen':
                                if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Wqueen)
                                elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Bqueen)
                            elif self.pieces_coordinates[square_object][1]['name'] == 'king':
                                if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Wking)
                                elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Bking)
                            self.current_red_icon = self.current_focus #Save the icon that currently is highlighted.
                        self.selection_flag = not self.selection_flag

                elif self.selection_flag and not self.yellow_selection_flag:
                        #Read which piece is in the currently highlighted square
                        with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                                live_situation = yaml.load(live_file.read())
                        self.piece_to_move = self.pieces_coordinates[live_situation[self.current_red_icon][0]][1]['name']
                        piece_to_capture = live_situation[self.current_focus][0]
                        self.color_to_move = live_situation[self.current_red_icon][1]
                        
                        #Save the color of the destination square and of the starting square
                        destination_color = self.chessboard_colors[self.current_focus]
                        starting_color = self.chessboard_colors[self.current_red_icon]

                        #Initialize variables to keep trak of the intentions of the player
                        self.icon_changed_1 = 'none'
                        self.old_icon_color_1 = 'none'
                        self.icon_changed_2 = 'none'
                        self.old_icon_color_2 = 'none'
                        self.icon_changed_3 = 'none'
                        self.old_icon_color_3 = 'none'
                        self.old_icon_piece_3 = 'none'
                        self.old_icon_piece_color_3 = 'none'
                        self.icon_changed_4 = 'none'
                        self.old_icon_color_4 = 'none'
                        self.old_icon_piece_4 = 'none'
                        self.old_icon_piece_color_4 = 'none'

                        #Change the destination icon and the starting square icon (destination color in yellow, cause I need to implement move confirmation)
                        if self.piece_to_move == 'pawn':
                            if self.color_to_move == 'white':
                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[0])
                                self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                            elif self.color_to_move == 'black':
                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[1])
                                self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                            #Implement the en-passant special move (only for white. For black will be implemented in the other chessboard GUI script):
                            count = 0
                            for i in self.chessboard_matrix:
                                if self.current_red_icon in i:
                                    prev_index = i.index(self.current_red_icon)
                                    capture_row = count
                                if self.current_focus in i:
                                    new_index = i.index(self.current_focus)
                                count += 1
                            if prev_index != new_index and piece_to_capture == 'none': #If a capture has been performed by the pawn and there was no piece in the ending square, en-passant happened.
                                #self.en_passant = True
                                square_captured_piece = self.chessboard_matrix[capture_row][new_index]
                                if self.chessboard_colors[square_captured_piece] == 'black':
                                    self.array[self.dict_squares[square_captured_piece]].setIcon(self.icon_grey)
                                    self.icon_changed_3 = self.array[self.dict_squares[square_captured_piece]]
                                    self.old_icon_color_3 = 'black'
                                    self.old_icon_piece_3 = 'pawn'
                                    self.old_icon_piece_color_3 = 'black' #Cause I know I'm playing with white
                                elif self.chessboard_colors[square_captured_piece] == 'white':
                                    self.array[self.dict_squares[square_captured_piece]].setIcon(QtGui.QIcon())
                                    self.icon_changed_3 = self.array[self.dict_squares[square_captured_piece]]
                                    self.old_icon_color_3 = 'white'
                                    self.old_icon_piece_3 = 'pawn'
                                    self.old_icon_piece_color_3 = 'black' #Cause I know I'm playing with white
                                #en_passant_square_to_move_publisher.publish(square_captured_piece)

                        elif self.piece_to_move == 'rook':
                                if self.color_to_move == 'white':
                                        self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[2])
                                        self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                        self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                                elif self.color_to_move == 'black':
                                        self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[3])
                                        self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                        self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                        elif self.piece_to_move == 'knight':
                                if self.color_to_move == 'white':
                                        self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[4])
                                        self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                        self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                                elif self.color_to_move == 'black':
                                        self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[5])
                                        self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                        self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                        elif self.piece_to_move == 'bishop':
                                if self.color_to_move == 'white':
                                        self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[6])
                                        self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                        self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                                elif self.color_to_move == 'black':
                                        self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[7])
                                        self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                        self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                        elif self.piece_to_move == 'queen':
                                if self.color_to_move == 'white':
                                        self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[8])
                                        self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                        self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                                elif self.color_to_move == 'black':
                                        self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[9])
                                        self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                        self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                        elif self.piece_to_move == 'king':
                                if self.color_to_move == 'white':
                                        self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[10])
                                        self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                        self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                                elif self.color_to_move == 'black':
                                        self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[11])
                                        self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                        self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                                #Implement the castle special move (only for black. For white it will be implemented in the ther chessboard GUI script):
                                if (self.current_focus == 'g8' or self.current_focus == 'c8') and (self.current_red_icon == 'e8'):
                                        if self.current_focus == 'g8': #Short castle
                                                #self.short_castle = True
                                                self.f8.setIcon(self.icons_array_yellow[3])
                                                self.icon_changed_2 = self.array[self.dict_squares['f8']]
                                                self.old_icon_color_2 = self.chessboard_colors['f8']
                                                self.h8.setIcon(QtGui.QIcon())
                                        elif self.current_focus == 'c8': #Long castle 
                                                #self.long_castle = True
                                                self.d8.setIcon(self.icons_array_yellow[3])
                                                self.icon_changed_2 = self.array[self.dict_squares['d8']]
                                                self.old_icon_color_2 = self.chessboard_colors['d8']
                                                self.a8.setIcon(self.icon_grey)
                        if starting_color == 'white':
                                self.array[self.dict_squares[self.current_red_icon]].setIcon(QtGui.QIcon())
                                self.icon_changed_4 = self.array[self.dict_squares[self.current_red_icon]]
                                self.old_icon_color_4 = 'white'
                                self.old_icon_piece_4 = self.piece_to_move
                                self.old_icon_piece_color_4 = self.color_to_move
                        elif starting_color == 'black':
                                self.array[self.dict_squares[self.current_red_icon]].setIcon(self.icon_grey)
                                self.icon_changed_4 = self.array[self.dict_squares[self.current_red_icon]]
                                self.old_icon_color_4 = 'black'
                                self.old_icon_piece_4 = self.piece_to_move
                                self.old_icon_piece_color_4 = self.color_to_move

                        self.yellow_selection_flag = not self.yellow_selection_flag
                        self.selection_flag = not self.selection_flag
                        self.intention_icon = self.current_focus #Save the square that the player is intentioned to play to.

                elif not self.selection_flag and self.yellow_selection_flag:
                    if self.current_focus == self.intention_icon: #If the player clicked on the same square again, the move is confirmed.
                        #Read which piece is in the currently highlighted square
                        with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                                live_situation = yaml.load(live_file.read())
                        self.piece_to_move = self.pieces_coordinates[live_situation[self.current_red_icon][0]][1]['name']
                        piece_to_capture = live_situation[self.current_focus][0]
                        self.color_to_move = live_situation[self.current_red_icon][1]
                        
                        #Save the color of the destination square and of the starting square
                        destination_color = self.chessboard_colors[self.current_focus]
                        starting_color = self.chessboard_colors[self.current_red_icon]

                        #Initialize the promoted_piece, the en_passant and the castle variables
                        #promoted_piece = 'none'
                        self.en_passant = False
                        self.short_castle = False
                        self.long_castle = False

                        if destination_color == 'black':
                                if self.piece_to_move == 'pawn':
                                        #Implement promotion (only for white. For black will be implemented in the other chessboard GUI script)
                                        count = 0
                                        for r in self.chessboard_matrix:
                                            if self.current_focus in r:
                                                    dest_row = count
                                                    break
                                            count += 1
                                        if dest_row == 7: #Promotion is happening
                                                state_publisher.publish(12) #Go to the state of promoted piece question
                                                #Make the user choose the promoted piece
                                                if self.color_to_move == 'white':
                                                    self.promotion.emit(1)
                                                elif self.color_to_move == 'black':
                                                    self.promotion.emit(2)                            
                                        else:
                                                if self.color_to_move == 'white':
                                                        self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[0])
                                                elif self.color_to_move == 'black':
                                                        self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[1])
                                                #Implement the en-passant special move (only for white. For black will be implemented in the other chessboard GUI script):
                                                count = 0
                                                for i in self.chessboard_matrix:
                                                        if self.current_red_icon in i:
                                                                prev_index = i.index(self.current_red_icon)
                                                                capture_row = count
                                                        if self.current_focus in i:
                                                                new_index = i.index(self.current_focus)
                                                        count += 1
                                                if prev_index != new_index and piece_to_capture == 'none': #If a capture has been performed by the pawn and there was no piece in the ending square, en-passant happened.
                                                        self.en_passant = True
                                                        square_captured_piece = self.chessboard_matrix[capture_row][new_index]
                                                        en_passant_square_to_move_publisher.publish(square_captured_piece)
                                elif self.piece_to_move == 'rook':
                                        if self.color_to_move == 'white':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[2])
                                        elif self.color_to_move == 'black':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[3])
                                elif self.piece_to_move == 'knight':
                                        if self.color_to_move == 'white':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[4])
                                        elif self.color_to_move == 'black':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[5])
                                elif self.piece_to_move == 'bishop':
                                        if self.color_to_move == 'white':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[6])
                                        elif self.color_to_move == 'black':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[7])
                                elif self.piece_to_move == 'queen':
                                        if self.color_to_move == 'white':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[8])
                                        elif self.color_to_move == 'black':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[9])
                                elif self.piece_to_move == 'king':
                                        if self.color_to_move == 'white':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[10])
                                        elif self.color_to_move == 'black':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[11])
                                        
                        elif destination_color == 'white':
                                if self.piece_to_move == 'pawn':
                                        #Implement promotion
                                        count = 0
                                        for r in self.chessboard_matrix:
                                            if self.current_focus in r:
                                                    dest_row = count
                                                    break
                                            count += 1
                                        if dest_row == 7: #Promotion is happening
                                                state_publisher.publish(12) #Go to the state of promoted piece question
                                                #Make the user choose the promoted piece
                                                if self.color_to_move == 'white':
                                                    self.promotion.emit(1)
                                                elif self.color_to_move == 'black':
                                                    self.promotion.emit(2)                                       
                                        else:
                                                if self.color_to_move == 'white':
                                                        self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[0])
                                                elif self.color_to_move == 'black':
                                                        self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[1])
                                                #Implement the en-passant special move:
                                                count = 0
                                                for i in self.chessboard_matrix:
                                                        if self.current_red_icon in i:
                                                                prev_index = i.index(self.current_red_icon)
                                                                capture_row = count
                                                        if self.current_focus in i:
                                                                new_index = i.index(self.current_focus)
                                                        count += 1
                                                if prev_index != new_index and piece_to_capture == 'none': #If a capture has been performed by the pawn and there was no piece in the ending square, en-passant happened.
                                                        self.en_passant = True
                                                        square_captured_piece = self.chessboard_matrix[capture_row][new_index]
                                                        en_passant_square_to_move_publisher.publish(square_captured_piece)

                                elif self.piece_to_move == 'rook':
                                        if self.color_to_move == 'white':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[2])
                                        elif self.color_to_move == 'black':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[3])
                                elif self.piece_to_move == 'knight':
                                        if self.color_to_move == 'white':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[4])
                                        elif self.color_to_move == 'black':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[5])
                                elif self.piece_to_move == 'bishop':
                                        if self.color_to_move == 'white':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[6])
                                        elif self.color_to_move == 'black':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[7])
                                elif self.piece_to_move == 'queen':
                                        if self.color_to_move == 'white':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[8])
                                        elif self.color_to_move == 'black':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[9])
                                elif self.piece_to_move == 'king':
                                        if self.color_to_move == 'white':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[10])
                                        elif self.color_to_move == 'black':
                                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[11])
                                        #Implement the castle special move (only for white. For black it will be implemented in the ther chessboard GUI script):
                                        if (self.current_focus == 'g8' or self.current_focus == 'c8') and (self.current_red_icon == 'e8'):
                                            if self.current_focus == 'g8': #Short castle
                                                    self.short_castle = True
                                                    self.f8.setIcon(self.icons_array_black[3])
                                                    self.h8.setIcon(QtGui.QIcon())
                                            elif self.current_focus == 'c8': #Long castle 
                                                    self.long_castle = True
                                                    self.d8.setIcon(self.icons_array_black[3])
                                                    self.a8.setIcon(self.icon_grey)
                        #Send a message containing the start square and the end square, so that TIAGo can execute it.
                        if self.short_castle:
                            self.start_square_move_publisher.publish('short')
                            self.end_square_move_publisher.publish('short')
                        elif self.long_castle:
                            self.start_square_move_publisher.publish('long')
                            self.end_square_move_publisher.publish('long')
                        else:
                            self.start_square_move_publisher.publish(self.current_red_icon)
                            self.end_square_move_publisher.publish(self.current_focus)
                            #self.promoted_piece_move_publisher.publish(promoted_piece)

                        #Change operational status.
                        if state != 12:
                            state_publisher.publish(13)
                            update_flag = True

                    else: #If the player is not confirming the move (by clicking in another square), get back to the state where he has to chosen the move.
                        if self.icon_changed_1 != 'none':

                            if self.old_icon_color_1 == 'white':
                                self.icon_changed_1.setIcon(QtGui.QIcon())
                            elif self.old_icon_color_1 == 'black':
                                self.icon_changed_1.setIcon(self.icon_grey)

                        if self.icon_changed_2 != 'none':
                            if self.old_icon_color_2 == 'white':
                                self.icon_changed_2.setIcon(QtGui.QIcon())
                            elif self.old_icon_color_2 == 'black':
                                self.icon_changed_2.setIcon(self.icon_grey)

                        if self.icon_changed_3 != 'none':
                            if self.old_icon_color_3 == 'white':
                                if self.old_icon_piece_3 == 'pawn':
                                    if self.old_icon_piece_color_3 == 'white':
                                        self.icon_changed_3.setIcon(self.icons_array_white[0])
                                    elif self.old_icon_piece_color_3 == 'black':
                                        self.icon_changed_3.setIcon(self.icons_array_white[1])
                                elif self.old_icon_piece_3 == 'rook':
                                    if self.old_icon_piece_color_3 == 'white':
                                        self.icon_changed_3.setIcon(self.icons_array_white[2])
                                    elif self.old_icon_piece_color_3 == 'black':
                                        self.icon_changed_3.setIcon(self.icons_array_white[3])
                                elif self.old_icon_piece_3 == 'knight':
                                    if self.old_icon_piece_color_3 == 'white':
                                        self.icon_changed_3.setIcon(self.icons_array_white[4])
                                    elif self.old_icon_piece_color_3 == 'black':
                                        self.icon_changed_3.setIcon(self.icons_array_white[5])
                                elif self.old_icon_piece_3 == 'bishop':
                                    if self.old_icon_piece_color_3 == 'white':
                                        self.icon_changed_3.setIcon(self.icons_array_white[6])
                                    elif self.old_icon_piece_color_3 == 'black':
                                        self.icon_changed_3.setIcon(self.icons_array_white[7])
                                elif self.old_icon_piece_3 == 'king':
                                    if self.old_icon_piece_color_3 == 'white':
                                        self.icon_changed_3.setIcon(self.icons_array_white[10])
                                    elif self.old_icon_piece_color_3 == 'black':
                                        self.icon_changed_3.setIcon(self.icons_array_white[11])
                                elif self.old_icon_piece_3 == 'queen':
                                    if self.old_icon_piece_color_3 == 'white':
                                        self.icon_changed_3.setIcon(self.icons_array_white[8])
                                    elif self.old_icon_piece_color_3 == 'black':
                                        self.icon_changed_3.setIcon(self.icons_array_white[9])
                            elif self.old_icon_color_3 == 'black':
                                if self.old_icon_piece_3 == 'pawn':
                                    if self.old_icon_piece_color_3 == 'white':
                                        self.icon_changed_3.setIcon(self.icons_array_black[0])
                                    elif self.old_icon_piece_color_3 == 'black':
                                        self.icon_changed_3.setIcon(self.icons_array_black[1])
                                elif self.old_icon_piece_3 == 'rook':
                                    if self.old_icon_piece_color_3 == 'white':
                                        self.icon_changed_3.setIcon(self.icons_array_black[2])
                                    elif self.old_icon_piece_color_3 == 'black':
                                        self.icon_changed_3.setIcon(self.icons_array_black[3])
                                elif self.old_icon_piece_3 == 'knight':
                                    if self.old_icon_piece_color_3 == 'white':
                                        self.icon_changed_3.setIcon(self.icons_array_black[4])
                                    elif self.old_icon_piece_color_3 == 'black':
                                        self.icon_changed_3.setIcon(self.icons_array_black[5])
                                elif self.old_icon_piece_3 == 'bishop':
                                    if self.old_icon_piece_color_3 == 'white':
                                        self.icon_changed_3.setIcon(self.icons_array_black[6])
                                    elif self.old_icon_piece_color_3 == 'black':
                                        self.icon_changed_3.setIcon(self.icons_array_black[7])
                                elif self.old_icon_piece_3 == 'king':
                                    if self.old_icon_piece_color_3 == 'white':
                                        self.icon_changed_3.setIcon(self.icons_array_black[10])
                                    elif self.old_icon_piece_color_3 == 'black':
                                        self.icon_changed_3.setIcon(self.icons_array_black[11])
                                elif self.old_icon_piece_3 == 'queen':
                                    if self.old_icon_piece_color_3 == 'white':
                                        self.icon_changed_3.setIcon(self.icons_array_black[8])
                                    elif self.old_icon_piece_color_3 == 'black':
                                        self.icon_changed_3.setIcon(self.icons_array_black[9])

                        if self.icon_changed_4 != 'none':
                            if self.old_icon_color_4 == 'white':
                                if self.old_icon_piece_4 == 'pawn':
                                    if self.old_icon_piece_color_4 == 'white':
                                        self.icon_changed_4.setIcon(self.icons_array_white[0])
                                    elif self.old_icon_piece_color_4 == 'black':
                                        self.icon_changed_4.setIcon(self.icons_array_white[1])
                                elif self.old_icon_piece_4 == 'rook':
                                    if self.old_icon_piece_color_4 == 'white':
                                        self.icon_changed_4.setIcon(self.icons_array_white[2])
                                    elif self.old_icon_piece_color_4 == 'black':
                                        self.icon_changed_4.setIcon(self.icons_array_white[3])
                                elif self.old_icon_piece_4 == 'knight':
                                    if self.old_icon_piece_color_4 == 'white':
                                        self.icon_changed_4.setIcon(self.icons_array_white[4])
                                    elif self.old_icon_piece_color_4 == 'black':
                                        self.icon_changed_4.setIcon(self.icons_array_white[5])
                                elif self.old_icon_piece_4 == 'bishop':
                                    if self.old_icon_piece_color_4 == 'white':
                                        self.icon_changed_4.setIcon(self.icons_array_white[6])
                                    elif self.old_icon_piece_color_4 == 'black':
                                        self.icon_changed_4.setIcon(self.icons_array_white[7])
                                elif self.old_icon_piece_4 == 'king':
                                    if self.old_icon_piece_color_4 == 'white':
                                        self.icon_changed_4.setIcon(self.icons_array_white[10])
                                    elif self.old_icon_piece_color_4 == 'black':
                                        self.icon_changed_4.setIcon(self.icons_array_white[11])
                                elif self.old_icon_piece_4 == 'queen':
                                    if self.old_icon_piece_color_4 == 'white':
                                        self.icon_changed_4.setIcon(self.icons_array_white[8])
                                    elif self.old_icon_piece_color_4 == 'black':
                                        self.icon_changed_4.setIcon(self.icons_array_white[9])
                            elif self.old_icon_color_4 == 'black':
                                if self.old_icon_piece_4 == 'pawn':
                                    if self.old_icon_piece_color_4 == 'white':
                                        self.icon_changed_4.setIcon(self.icons_array_black[0])
                                    elif self.old_icon_piece_color_4 == 'black':
                                        self.icon_changed_4.setIcon(self.icons_array_black[1])
                                elif self.old_icon_piece_4 == 'rook':
                                    if self.old_icon_piece_color_4 == 'white':
                                        self.icon_changed_4.setIcon(self.icons_array_black[2])
                                    elif self.old_icon_piece_color_4 == 'black':
                                        self.icon_changed_4.setIcon(self.icons_array_black[3])
                                elif self.old_icon_piece_4 == 'knight':
                                    if self.old_icon_piece_color_4 == 'white':
                                        self.icon_changed_4.setIcon(self.icons_array_black[4])
                                    elif self.old_icon_piece_color_4 == 'black':
                                        self.icon_changed_4.setIcon(self.icons_array_black[5])
                                elif self.old_icon_piece_4 == 'bishop':
                                    if self.old_icon_piece_color_4 == 'white':
                                        self.icon_changed_4.setIcon(self.icons_array_black[6])
                                    elif self.old_icon_piece_color_4 == 'black':
                                        self.icon_changed_4.setIcon(self.icons_array_black[7])
                                elif self.old_icon_piece_4 == 'king':
                                    if self.old_icon_piece_color_4 == 'white':
                                        self.icon_changed_4.setIcon(self.icons_array_black[10])
                                    elif self.old_icon_piece_color_4 == 'black':
                                        self.icon_changed_4.setIcon(self.icons_array_black[11])
                                elif self.old_icon_piece_4 == 'queen':
                                    if self.old_icon_piece_color_4 == 'white':
                                        self.icon_changed_4.setIcon(self.icons_array_black[8])
                                    elif self.old_icon_piece_color_4 == 'black':
                                        self.icon_changed_4.setIcon(self.icons_array_black[9])
                    #Change the yellow selection flag
                    self.yellow_selection_flag = not self.yellow_selection_flag


            elif state == 70: #Manual mode.
                with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                        live_situation = yaml.load(live_file.read())
                if action ==  'delete':
                    square_object = live_situation[square][0]
                    if square_object == 'none':
                            rospy.logwarn('There is no piece to delete in the selected square')
                    else:
                        #Save the color of the square
                        square_color = self.chessboard_colors[square]
                        if square_color == 'white':
                            self.array[self.dict_squares[square]].setIcon(QtGui.QIcon())
                            #Change the live chessboard situation
                            live_situation[square][0] = 'none'
                            live_situation[square][1] = 'none'
                            with open(self.dir_live_chessboard_situation, "w") as t_p:
                                yaml.dump(live_situation, t_p)
                        elif square_color == 'black':
                            self.array[self.dict_squares[square]].setIcon(self.icon_grey)
                            #Change the live chessboard situation
                            live_situation[square][0] = 'none'
                            live_situation[square][1] = 'none'
                            with open(self.dir_live_chessboard_situation, "w") as t_p:
                                yaml.dump(live_situation, t_p)

                else: #If the action is to manually place a piece in the chessboard.
                    #Save the color of the square
                    square_color = self.chessboard_colors[square]

                    #Save the pieces that are currently in the live situation
                    present_bishops_w = []
                    present_rooks_w = []
                    present_knights_w = []
                    present_pawns_w = []
                    present_bishops_b = []
                    present_rooks_b = []
                    present_knights_b = []
                    present_pawns_b = []

                    for casella in live_situation:
                        if casella[0] == 'bishop_c1' or casella[0] == 'bishop_f1':
                            present_bishops_w.append(casella[0])
                        elif casella[0] == 'bishop_c8' or casella[0] == 'bishop_f8':
                            present_bishops_b.append(casella[0])
                        elif casella[0] == 'knight_b1' or casella[0] == 'knight_g1':
                            present_knights_w.append(casella[0])
                        elif casella[0] == 'knight_b8' or casella[0] == 'knight_g8':
                            present_knights_b.append(casella[0])
                        elif casella[0] == 'rook_a1' or casella[0] == 'rook_h1':
                            present_rooks_w.append(casella[0])
                        elif casella[0] == 'rook_a8' or casella[0] == 'rook_h8':
                            present_rooks_b.append(casella[0])
                        elif casella[0] == 'pawn_a2' or casella[0] == 'pawn_b2' or casella[0] == 'pawn_c2' or casella[0] == 'pawn_d2' or casella[0] == 'pawn_e2' or casella[0] == 'pawn_f2' or casella[0] == 'pawn_g2' or casella[0] == 'pawn_h2':
                            present_pawns_w.append(casella[0])
                        elif casella[0] == 'pawn_a7' or casella[0] == 'pawn_b7' or casella[0] == 'pawn_c7' or casella[0] == 'pawn_d7' or casella[0] == 'pawn_e7' or casella[0] == 'pawn_f7' or casella[0] == 'pawn_g7' or casella[0] == 'pawn_h7':
                            present_pawns_b.append(casella[0])

                    if square_color == 'white':
                        if action == 'w_pawn':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_white[0])
                            #Change the live chessboard situation
                            if square_object in pawns_w:
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'white'
                            else:
                                if len(present_pawns_w) == 0:
                                    live_situation[square][0] = 'pawn_a2'
                                    live_situation[square][1] = 'white'
                                elif len(present_pawns_w) == 8:
                                    live_situation[square][0] = 'inserted_pawn'
                                    live_situation[square][1] = 'white'
                                else:
                                    self.white_pawns = cfg_w.white_pawns
                                    for pawn in self.white_pawns:
                                        if pawn not in present_pawns_w:
                                            live_situation[square][0] = pawn
                                            live_situation[square][1] = 'white'
                                            break
                        elif action == 'b_pawn':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_white[1])
                            #Change the live chessboard situation
                            if square_object in pawns_b:
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'black'
                            else:
                                if len(present_pawns_b) == 0:
                                    live_situation[square][0] = 'pawn_a7'
                                    live_situation[square][1] = 'black'
                                elif len(present_pawns_b) == 8:
                                    live_situation[square][0] = 'inserted_pawn'
                                    live_situation[square][1] = 'black'
                                else:
                                    self.black_pawns = cfg_w.black_pawns
                                    for pawn in self.black_pawns:
                                        if pawn not in present_pawns_b:
                                            live_situation[square][0] = pawn
                                            live_situation[square][1] = 'black'
                                            break
                        elif action == 'w_knight':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_white[4])
                            #Change the live chessboard situation
                            if square_object == 'knight_g1' or square_object == 'knight_b1':
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'white'
                            else:
                                if len(present_knights_w) == 1:
                                    live_situation[square][0] = present_knights_w[0]
                                    live_situation[square][1] = 'white'
                                elif len(present_knights_w) == 2:
                                    live_situation[square][0] = 'inserted_knight'
                                    live_situation[square][1] = 'white'
                                elif len(present_knights_w) == 0:
                                    live_situation[square][0] = 'knight_b1'
                                    live_situation[square][1] = 'white'                    
                        elif action == 'b_knight':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_white[5])
                            #Change the live chessboard situation
                            if square_object == 'knight_g8' or square_object == 'knight_b8':
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'black'
                            else:
                                if len(present_knights_b) == 1:
                                    live_situation[square][0] = present_knights_b[0]
                                    live_situation[square][1] = 'black'
                                elif len(present_knights_b) == 2:
                                    live_situation[square][0] = 'inserted_knight'
                                    live_situation[square][1] = 'black'
                                elif len(present_knights_b) == 0:
                                    live_situation[square][0] = 'knight_b8'
                                    live_situation[square][1] = 'black'                        
                        elif action == 'w_bishop':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_white[6])
                            #Change the live chessboard situation
                            if square_object == 'bishop_c1' or square_object == 'bishop_f1':
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'white'
                            else:
                                if len(present_bishops_w) == 1:
                                    live_situation[square][0] = present_bishops_w[0]
                                    live_situation[square][1] = 'white'
                                elif len(present_bishops_w) == 2:
                                    live_situation[square][0] = 'inserted_bishop'
                                    live_situation[square][1] = 'white'
                                elif len(present_bishops_w) == 0:
                                    live_situation[square][0] = 'bishop_c1'
                                    live_situation[square][1] = 'white'                        
                        elif action == 'b_bishop':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_white[7])
                            #Change the live chessboard situation
                            if square_object == 'bishop_c8' or square_object == 'bishop_f8':
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'black'
                            else:
                                if len(present_bishops_b) == 1:
                                    live_situation[square][0] = present_bishops_b[0]
                                    live_situation[square][1] = 'black'
                                elif len(present_bishops_b) == 2:
                                    live_situation[square][0] = 'inserted_bishop'
                                    live_situation[square][1] = 'black'
                                elif len(present_bishops_b) == 0:
                                    live_situation[square][0] = 'bishop_c8'
                                    live_situation[square][1] = 'black'                        
                        elif action == 'w_rook':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_white[2])
                            #Change the live chessboard situation
                            if square_object == 'rook_a1' or square_object == 'rook_h1':
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'white'
                            else:
                                if len(present_rooks_w) == 1:
                                    live_situation[square][0] = present_rooks_w[0]
                                    live_situation[square][1] = 'white'
                                elif len(present_rooks_w) == 2:
                                    live_situation[square][0] = 'inserted_rook'
                                    live_situation[square][1] = 'white'
                                elif len(present_rooks_w) == 0:
                                    live_situation[square][0] = 'rook_a1'
                                    live_situation[square][1] = 'white'                        
                        elif action == 'b_rook':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_white[3])
                            #Change the live chessboard situation
                            if square_object == 'rook_a8' or square_object == 'rook_h8':
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'black'
                            else:
                                if len(present_rooks_b) == 1:
                                    live_situation[square][0] = present_rooks_b[0]
                                    live_situation[square][1] = 'black'
                                elif len(present_rooks_b) == 2:
                                    live_situation[square][0] = 'inserted_rook'
                                    live_situation[square][1] = 'black'
                                elif len(present_rooks_b) == 0:
                                    live_situation[square][0] = 'rook_a8'
                                    live_situation[square][1] = 'black'                       
                        elif action == 'w_queen':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_white[8])
                            #Change the live chessboard situation
                            live_situation[square][0] = 'queen_d1'
                            live_situation[square][1] = 'white'                        
                        elif action == 'b_queen':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_white[9])
                            #Change the live chessboard situation
                            live_situation[square][0] = 'queen_d8'
                            live_situation[square][1] = 'black'                        
                        elif action == 'w_king':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_white[10])
                            #Change the live chessboard situation
                            live_situation[square][0] = 'king_e1'
                            live_situation[square][1] = 'white'                        
                        elif action == 'b_king':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_white[11])
                            #Change the live chessboard situation
                            live_situation[square][0] = 'king_e8'
                            live_situation[square][1] = 'black'                        
                    elif square_color == 'black':
                        if action == 'w_pawn':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_black[0])
                            #Change the live chessboard situation
                            if square_object in pawns_w:
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'white'
                            else:
                                if len(present_pawns_w) == 0:
                                    live_situation[square][0] = 'pawn_a2'
                                    live_situation[square][1] = 'white'
                                elif len(present_pawns_w) == 8:
                                    live_situation[square][0] = 'inserted_pawn'
                                    live_situation[square][1] = 'white'
                                else:
                                    self.white_pawns = cfg_w.white_pawns
                                    for pawn in self.white_pawns:
                                        if pawn not in present_pawns_w:
                                            live_situation[square][0] = pawn
                                            live_situation[square][1] = 'white'
                                            break
                        elif action == 'b_pawn':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_black[1])
                            #Change the live chessboard situation
                            if square_object in pawns_b:
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'black'
                            else:
                                if len(present_pawns_b) == 0:
                                    live_situation[square][0] = 'pawn_a7'
                                    live_situation[square][1] = 'black'
                                elif len(present_pawns_b) == 8:
                                    live_situation[square][0] = 'inserted_pawn'
                                    live_situation[square][1] = 'black'
                                else:
                                    self.black_pawns = cfg_w.black_pawns
                                    for pawn in self.black_pawns:
                                        if pawn not in present_pawns_b:
                                            live_situation[square][0] = pawn
                                            live_situation[square][1] = 'black'
                                            break
                        elif action == 'w_knight':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_black[4])
                            #Change the live chessboard situation
                            if square_object == 'knight_g1' or square_object == 'knight_b1':
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'white'
                            else:
                                if len(present_knights_w) == 1:
                                    live_situation[square][0] = present_knights_w[0]
                                    live_situation[square][1] = 'white'
                                elif len(present_knights_w) == 2:
                                    live_situation[square][0] = 'inserted_knight'
                                    live_situation[square][1] = 'white'
                                elif len(present_knights_w) == 0:
                                    live_situation[square][0] = 'knight_b1'
                                    live_situation[square][1] = 'white'                    
                        elif action == 'b_knight':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_black[5])
                            #Change the live chessboard situation
                            if square_object == 'knight_g8' or square_object == 'knight_b8':
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'black'
                            else:
                                if len(present_knights_b) == 1:
                                    live_situation[square][0] = present_knights_b[0]
                                    live_situation[square][1] = 'black'
                                elif len(present_knights_b) == 2:
                                    live_situation[square][0] = 'inserted_knight'
                                    live_situation[square][1] = 'black'
                                elif len(present_knights_b) == 0:
                                    live_situation[square][0] = 'knight_b8'
                                    live_situation[square][1] = 'black'                        
                        elif action == 'w_bishop':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_black[6])
                            #Change the live chessboard situation
                            if square_object == 'bishop_c1' or square_object == 'bishop_f1':
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'white'
                            else:
                                if len(present_bishops_w) == 1:
                                    live_situation[square][0] = present_bishops_w[0]
                                    live_situation[square][1] = 'white'
                                elif len(present_bishops_w) == 2:
                                    live_situation[square][0] = 'inserted_bishop'
                                    live_situation[square][1] = 'white'
                                elif len(present_bishops_w) == 0:
                                    live_situation[square][0] = 'bishop_c1'
                                    live_situation[square][1] = 'white'                        
                        elif action == 'b_bishop':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_black[7])
                            #Change the live chessboard situation
                            if square_object == 'bishop_c8' or square_object == 'bishop_f8':
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'black'
                            else:
                                if len(present_bishops_b) == 1:
                                    live_situation[square][0] = present_bishops_b[0]
                                    live_situation[square][1] = 'black'
                                elif len(present_bishops_b) == 2:
                                    live_situation[square][0] = 'inserted_bishop'
                                    live_situation[square][1] = 'black'
                                elif len(present_bishops_b) == 0:
                                    live_situation[square][0] = 'bishop_c8'
                                    live_situation[square][1] = 'black'                        
                        elif action == 'w_rook':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_black[2])
                            #Change the live chessboard situation
                            if square_object == 'rook_a1' or square_object == 'rook_h1':
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'white'
                            else:
                                if len(present_rooks_w) == 1:
                                    live_situation[square][0] = present_rooks_w[0]
                                    live_situation[square][1] = 'white'
                                elif len(present_rooks_w) == 2:
                                    live_situation[square][0] = 'inserted_rook'
                                    live_situation[square][1] = 'white'
                                elif len(present_rooks_w) == 0:
                                    live_situation[square][0] = 'rook_a1'
                                    live_situation[square][1] = 'white'                        
                        elif action == 'b_rook':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_black[3])
                            #Change the live chessboard situation
                            if square_object == 'rook_a8' or square_object == 'rook_h8':
                                live_situation[square][0] = square_object
                                live_situation[square][1] = 'black'
                            else:
                                if len(present_rooks_b) == 1:
                                    live_situation[square][0] = present_rooks_b[0]
                                    live_situation[square][1] = 'black'
                                elif len(present_rooks_b) == 2:
                                    live_situation[square][0] = 'inserted_rook'
                                    live_situation[square][1] = 'black'
                                elif len(present_rooks_b) == 0:
                                    live_situation[square][0] = 'rook_a8'
                                    live_situation[square][1] = 'black'                       
                        elif action == 'w_queen':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_black[10])
                            #Change the live chessboard situation
                            live_situation[square][0] = 'queen_d1'
                            live_situation[square][1] = 'white'                        
                        elif action == 'b_queen':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_black[11])
                            #Change the live chessboard situation
                            live_situation[square][0] = 'queen_d8'
                            live_situation[square][1] = 'black'                        
                        elif action == 'w_king':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_black[8])
                            #Change the live chessboard situation
                            live_situation[square][0] = 'king_e1'
                            live_situation[square][1] = 'white'                        
                        elif action == 'b_king':
                            self.array[self.dict_squares[square]].setIcon(self.icons_array_black[9])
                            #Change the live chessboard situation
                            live_situation[square][0] = 'king_e8'
                            live_situation[square][1] = 'black'
                    with open(self.dir_live_chessboard_situation, "w") as t_p:
                            yaml.dump(live_situation, t_p)


    def click_and_focus(self, square):
        global square_object
        global square_captured_piece
        if state == 11: #If we are in game mode.
            #Change the current_focus variable.
            self.current_focus = square
            print('CURRENT FOCUS: ' + str(self.current_focus))

            #If no other square is selected, select the square and change the color of it in red.
            if not self.selection_flag and not self.yellow_selection_flag:
                    #Read if the selected square is occupied by one piece or if it's empty.
                    with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                            live_situation = yaml.load(live_file.read())
                    square_object = live_situation[self.current_focus][0]
                    if square_object == 'none':
                            rospy.logwarn('There is no piece to move in the selected square')
                            self.selection_flag = not self.selection_flag
                    else:
                        self.color_to_move = live_situation[self.current_focus][1]
                        if self.pieces_coordinates[square_object][1]['name'] == 'pawn':
                            if self.color_to_move == 'white':
                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Wpawn)
                            elif self.color_to_move == 'black':
                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Bpawn)
                        elif self.pieces_coordinates[square_object][1]['name'] == 'rook':
                            if self.color_to_move == 'white':
                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Wrook)
                            elif self.color_to_move == 'black':
                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Brook)
                        elif self.pieces_coordinates[square_object][1]['name'] == 'knight':
                            if self.color_to_move == 'white':
                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Wknight)
                            elif self.color_to_move == 'black':
                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Bknight)
                        elif self.pieces_coordinates[square_object][1]['name'] == 'bishop':
                            if self.color_to_move == 'white':
                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Wbishop)
                            elif self.color_to_move == 'black':
                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Bbishop)
                        elif self.pieces_coordinates[square_object][1]['name'] == 'queen':
                            if self.color_to_move == 'white':
                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Wqueen)
                            elif self.color_to_move == 'black':
                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Bqueen)
                        elif self.pieces_coordinates[square_object][1]['name'] == 'king':
                            if self.color_to_move == 'white':
                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Wking)
                            elif self.color_to_move == 'black':
                                self.array[self.dict_squares[self.current_focus]].setIcon(self.icon_red_Bking)
                        self.current_red_icon = self.current_focus #Save the icon that currently is highlighted.
                    self.selection_flag = not self.selection_flag

            elif self.selection_flag and not self.yellow_selection_flag:
                    #Read which piece is in the currently highlighted square
                    with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                            live_situation = yaml.load(live_file.read())
                    self.piece_to_move = self.pieces_coordinates[live_situation[self.current_red_icon][0]][1]['name']
                    piece_to_capture = live_situation[self.current_focus][0]
                    self.color_to_move = live_situation[self.current_red_icon][1]
                    
                    #Save the color of the destination square and of the starting square
                    destination_color = self.chessboard_colors[self.current_focus]
                    starting_color = self.chessboard_colors[self.current_red_icon]

                    #Initialize variables to keep trak of the intentions of the player
                    self.icon_changed_1 = 'none'
                    self.old_icon_color_1 = 'none'
                    self.icon_changed_2 = 'none'
                    self.old_icon_color_2 = 'none'
                    self.icon_changed_3 = 'none'
                    self.old_icon_color_3 = 'none'
                    self.old_icon_piece_3 = 'none'
                    self.old_icon_piece_color_3 = 'none'
                    self.icon_changed_4 = 'none'
                    self.old_icon_color_4 = 'none'
                    self.old_icon_piece_4 = 'none'
                    self.old_icon_piece_color_4 = 'none'

                    #Change the destination icon and the starting square icon (destination color in yellow, cause I need to implement move confirmation)
                    if self.piece_to_move == 'pawn':
                        if self.color_to_move == 'white':
                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[0])
                            self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                            self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                        elif self.color_to_move == 'black':
                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[1])
                            self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                            self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                        #Implement the en-passant special move (only for white. For black will be implemented in the other chessboard GUI script):
                        count = 0
                        for i in self.chessboard_matrix:
                            if self.current_red_icon in i:
                                prev_index = i.index(self.current_red_icon)
                                capture_row = count
                            if self.current_focus in i:
                                new_index = i.index(self.current_focus)
                            count += 1
                        if prev_index != new_index and piece_to_capture == 'none': #If a capture has been performed by the pawn and there was no piece in the ending square, en-passant happened.
                            #self.en_passant = True
                            square_captured_piece = self.chessboard_matrix[capture_row][new_index]
                            if self.chessboard_colors[square_captured_piece] == 'black':
                                self.array[self.dict_squares[square_captured_piece]].setIcon(self.icon_grey)
                                self.icon_changed_3 = self.array[self.dict_squares[square_captured_piece]]
                                self.old_icon_color_3 = 'black'
                                self.old_icon_piece_3 = 'pawn'
                                self.old_icon_piece_color_3 = 'black' #Cause I know I'm playing with white
                            elif self.chessboard_colors[square_captured_piece] == 'white':
                                self.array[self.dict_squares[square_captured_piece]].setIcon(QtGui.QIcon())
                                self.icon_changed_3 = self.array[self.dict_squares[square_captured_piece]]
                                self.old_icon_color_3 = 'white'
                                self.old_icon_piece_3 = 'pawn'
                                self.old_icon_piece_color_3 = 'black' #Cause I know I'm playing with white
                            #en_passant_square_to_move_publisher.publish(square_captured_piece)

                    elif self.piece_to_move == 'rook':
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[2])
                                    self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                    self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[3])
                                    self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                    self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                    elif self.piece_to_move == 'knight':
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[4])
                                    self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                    self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[5])
                                    self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                    self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                    elif self.piece_to_move == 'bishop':
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[6])
                                    self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                    self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[7])
                                    self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                    self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                    elif self.piece_to_move == 'queen':
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[8])
                                    self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                    self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[9])
                                    self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                    self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                    elif self.piece_to_move == 'king':
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[10])
                                    self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                    self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_yellow[11])
                                    self.icon_changed_1 = self.array[self.dict_squares[self.current_focus]]
                                    self.old_icon_color_1 = self.chessboard_colors[self.current_focus]
                            #Implement the castle special move (only for black. For white it will be implemented in the ther chessboard GUI script):
                            if (self.current_focus == 'g8' or self.current_focus == 'c8') and (self.current_red_icon == 'e8'):
                                    if self.current_focus == 'g8': #Short castle
                                            #self.short_castle = True
                                            self.f8.setIcon(self.icons_array_yellow[3])
                                            self.icon_changed_2 = self.array[self.dict_squares['f8']]
                                            self.old_icon_color_2 = self.chessboard_colors['f8']
                                            self.h8.setIcon(self.icon_grey)
                                    elif self.current_focus == 'c8': #Long castle 
                                            #self.long_castle = True
                                            self.d8.setIcon(self.icons_array_yellow[3])
                                            self.icon_changed_2 = self.array[self.dict_squares['d8']]
                                            self.old_icon_color_2 = self.chessboard_colors['d8']
                                            self.a8.setIcon(QtGui.QIcon())
                    if starting_color == 'white':
                            self.array[self.dict_squares[self.current_red_icon]].setIcon(QtGui.QIcon())
                            self.icon_changed_4 = self.array[self.dict_squares[self.current_red_icon]]
                            self.old_icon_color_4 = 'white'
                            self.old_icon_piece_4 = self.piece_to_move
                            self.old_icon_piece_color_4 = self.color_to_move
                    elif starting_color == 'black':
                            self.array[self.dict_squares[self.current_red_icon]].setIcon(self.icon_grey)
                            self.icon_changed_4 = self.array[self.dict_squares[self.current_red_icon]]
                            self.old_icon_color_4 = 'black'
                            self.old_icon_piece_4 = self.piece_to_move
                            self.old_icon_piece_color_4 = self.color_to_move

                    self.yellow_selection_flag = not self.yellow_selection_flag
                    self.selection_flag = not self.selection_flag
                    self.intention_icon = self.current_focus #Save the square that the player is intentioned to play to.

            elif not self.selection_flag and self.yellow_selection_flag:
                if self.current_focus == self.intention_icon: #If the player clicked on the same square again, the move is confirmed.
                    #Read which piece is in the currently highlighted square
                    with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                            live_situation = yaml.load(live_file.read())
                    self.piece_to_move = self.pieces_coordinates[live_situation[self.current_red_icon][0]][1]['name']
                    piece_to_capture = live_situation[self.current_focus][0]
                    self.color_to_move = live_situation[self.current_red_icon][1]
                    
                    #Save the color of the destination square and of the starting square
                    destination_color = self.chessboard_colors[self.current_focus]
                    starting_color = self.chessboard_colors[self.current_red_icon]

                    #Initialize the promoted_piece, the en_passant and the castle variables
                    #promoted_piece = 'none'
                    self.en_passant = False
                    self.short_castle = False
                    self.long_castle = False

                    if destination_color == 'black':
                            if self.piece_to_move == 'pawn':
                                    #Implement promotion (only for white. For black will be implemented in the other chessboard GUI script)
                                    count = 0
                                    for r in self.chessboard_matrix:
                                        if self.current_focus in r:
                                                dest_row = count
                                                break
                                        count += 1
                                    if dest_row == 7: #Promotion is happening
                                            state_publisher.publish(12) #Go to the state of promoted piece question
                                            #Make the user choose the promoted piece
                                            if self.color_to_move == 'white':
                                                self.promotion.emit(1)
                                            elif self.color_to_move == 'black':
                                                self.promotion.emit(2)                            
                                    else:
                                            if self.color_to_move == 'white':
                                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[0])
                                            elif self.color_to_move == 'black':
                                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[1])
                                            #Implement the en-passant special move (only for white. For black will be implemented in the other chessboard GUI script):
                                            count = 0
                                            for i in self.chessboard_matrix:
                                                    if self.current_red_icon in i:
                                                            prev_index = i.index(self.current_red_icon)
                                                            capture_row = count
                                                    if self.current_focus in i:
                                                            new_index = i.index(self.current_focus)
                                                    count += 1
                                            if prev_index != new_index and piece_to_capture == 'none': #If a capture has been performed by the pawn and there was no piece in the ending square, en-passant happened.
                                                    self.en_passant = True
                                                    square_captured_piece = self.chessboard_matrix[capture_row][new_index]
                                                    en_passant_square_to_move_publisher.publish(square_captured_piece)
                            elif self.piece_to_move == 'rook':
                                    if self.color_to_move == 'white':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[2])
                                    elif self.color_to_move == 'black':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[3])
                            elif self.piece_to_move == 'knight':
                                    if self.color_to_move == 'white':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[4])
                                    elif self.color_to_move == 'black':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[5])
                            elif self.piece_to_move == 'bishop':
                                    if self.color_to_move == 'white':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[6])
                                    elif self.color_to_move == 'black':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[7])
                            elif self.piece_to_move == 'queen':
                                    if self.color_to_move == 'white':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[8])
                                    elif self.color_to_move == 'black':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[9])
                            elif self.piece_to_move == 'king':
                                    if self.color_to_move == 'white':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[10])
                                    elif self.color_to_move == 'black':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_black[11])
                                    
                    elif destination_color == 'white':
                            if self.piece_to_move == 'pawn':
                                    #Implement promotion
                                    count = 0
                                    for r in self.chessboard_matrix:
                                        if self.current_focus in r:
                                                dest_row = count
                                                break
                                        count += 1
                                    if dest_row == 7: #Promotion is happening
                                            state_publisher.publish(12) #Go to the state of promoted piece question
                                            #Make the user choose the promoted piece
                                            if self.color_to_move == 'white':
                                                self.promotion.emit(1)
                                            elif self.color_to_move == 'black':
                                                self.promotion.emit(2)                                       
                                    else:
                                            if self.color_to_move == 'white':
                                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[0])
                                            elif self.color_to_move == 'black':
                                                    self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[1])
                                            #Implement the en-passant special move:
                                            count = 0
                                            for i in self.chessboard_matrix:
                                                    if self.current_red_icon in i:
                                                            prev_index = i.index(self.current_red_icon)
                                                            capture_row = count
                                                    if self.current_focus in i:
                                                            new_index = i.index(self.current_focus)
                                                    count += 1
                                            if prev_index != new_index and piece_to_capture == 'none': #If a capture has been performed by the pawn and there was no piece in the ending square, en-passant happened.
                                                    self.en_passant = True
                                                    square_captured_piece = self.chessboard_matrix[capture_row][new_index]
                                                    en_passant_square_to_move_publisher.publish(square_captured_piece)

                            elif self.piece_to_move == 'rook':
                                    if self.color_to_move == 'white':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[2])
                                    elif self.color_to_move == 'black':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[3])
                            elif self.piece_to_move == 'knight':
                                    if self.color_to_move == 'white':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[4])
                                    elif self.color_to_move == 'black':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[5])
                            elif self.piece_to_move == 'bishop':
                                    if self.color_to_move == 'white':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[6])
                                    elif self.color_to_move == 'black':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[7])
                            elif self.piece_to_move == 'queen':
                                    if self.color_to_move == 'white':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[8])
                                    elif self.color_to_move == 'black':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[9])
                            elif self.piece_to_move == 'king':
                                    if self.color_to_move == 'white':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[10])
                                    elif self.color_to_move == 'black':
                                            self.array[self.dict_squares[self.current_focus]].setIcon(self.icons_array_white[11])
                                    #Implement the castle special move (only for black. For white it will be implemented in the ther chessboard GUI script):
                                    if (self.current_focus == 'g8' or self.current_focus == 'c8') and (self.current_red_icon == 'e8'):
                                        if self.current_focus == 'g8': #Short castle
                                                self.short_castle = True
                                                self.f8.setIcon(self.icons_array_black[3])
                                                self.h8.setIcon(self.icon_grey)
                                        elif self.current_focus == 'c8': #Long castle 
                                                self.long_castle = True
                                                self.d8.setIcon(self.icons_array_black[3])
                                                self.a8.setIcon(QtGui.QIcon())

                    #Send a message containing the start square and the end square, so that TIAGo can execute it.
                    if self.short_castle:
                        self.start_square_move_publisher.publish('short')
                        self.end_square_move_publisher.publish('short')
                    elif self.long_castle:
                        self.start_square_move_publisher.publish('long')
                        self.end_square_move_publisher.publish('long')
                    else:
                        self.start_square_move_publisher.publish(self.current_red_icon)
                        self.end_square_move_publisher.publish(self.current_focus)
                        #self.promoted_piece_move_publisher.publish(promoted_piece)

                    #Change operational status.
                    if state != 12:
                        state_publisher.publish(13)
                        update_flag = True

                else: #If the player is not confirming the move (by clicking in another square), get back to the state where he has to chosen the move.
                    with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                        live_situation = yaml.load(live_file.read())
                    if self.icon_changed_1 != 'none':
                        pezzo = self.pieces_coordinates[live_situation[self.intention_icon][0]][1]['name'] ####
                        pezzo_color = live_situation[self.intention_icon][1]
                        if self.old_icon_color_1 == 'white':
                            if pezzo == 'pawn':
                                if pezzo_color == 'white':
                                    self.icon_changed_1.setIcon(self.icons_array_white[0])
                                elif pezzo_color == 'black':
                                    self.icon_changed_1.setIcon(self.icons_array_white[1])
                            elif pezzo == 'bishop':
                                if pezzo_color == 'white':
                                    self.icon_changed_1.setIcon(self.icons_array_white[6])
                                elif pezzo_color == 'black':
                                    self.icon_changed_1.setIcon(self.icons_array_white[7])
                            elif pezzo == 'knight':
                                if pezzo_color == 'white':
                                    self.icon_changed_1.setIcon(self.icons_array_white[4])
                                elif pezzo_color == 'black':
                                    self.icon_changed_1.setIcon(self.icons_array_white[5])
                            elif pezzo == 'rook':
                                if pezzo_color == 'white':
                                    self.icon_changed_1.setIcon(self.icons_array_white[2])
                                elif pezzo_color == 'black':
                                    self.icon_changed_1.setIcon(self.icons_array_white[3])
                            elif pezzo == 'king':
                                if pezzo_color == 'white':
                                    self.icon_changed_1.setIcon(self.icons_array_white[10])
                                elif pezzo_color == 'black':
                                    self.icon_changed_1.setIcon(self.icons_array_white[11])
                            elif pezzo == 'queen':
                                if pezzo_color == 'white':
                                    self.icon_changed_1.setIcon(self.icons_array_white[8])
                                elif pezzo_color == 'black':
                                    self.icon_changed_1.setIcon(self.icons_array_white[9])
                            else:
                                self.icon_changed_1.setIcon(QtGui.QIcon())
                        elif self.old_icon_color_1 == 'black':
                            if pezzo == 'pawn':
                                if pezzo_color == 'white':
                                    self.icon_changed_1.setIcon(self.icons_array_black[0])
                                elif pezzo_color == 'black':
                                    self.icon_changed_1.setIcon(self.icons_array_black[1])
                            elif pezzo == 'bishop':
                                if pezzo_color == 'white':
                                    self.icon_changed_1.setIcon(self.icons_array_black[6])
                                elif pezzo_color == 'black':
                                    self.icon_changed_1.setIcon(self.icons_array_black[7])
                            elif pezzo == 'knight':
                                if pezzo_color == 'white':
                                    self.icon_changed_1.setIcon(self.icons_array_black[4])
                                elif pezzo_color == 'black':
                                    self.icon_changed_1.setIcon(self.icons_array_black[5])
                            elif pezzo == 'rook':
                                if pezzo_color == 'white':
                                    self.icon_changed_1.setIcon(self.icons_array_black[2])
                                elif pezzo_color == 'black':
                                    self.icon_changed_1.setIcon(self.icons_array_black[3])
                            elif pezzo == 'king':
                                if pezzo_color == 'white':
                                    self.icon_changed_1.setIcon(self.icons_array_black[10])
                                elif pezzo_color == 'black':
                                    self.icon_changed_1.setIcon(self.icons_array_black[11])
                            elif pezzo == 'queen':
                                if pezzo_color == 'white':
                                    self.icon_changed_1.setIcon(self.icons_array_black[8])
                                elif pezzo_color == 'black':
                                    self.icon_changed_1.setIcon(self.icons_array_black[9])
                            else:
                                self.icon_changed_1.setIcon(self.icon_grey)

                    if self.icon_changed_3 != 'none':
                        if self.old_icon_color_3 == 'white':
                            if self.old_icon_piece_3 == 'pawn':
                                if self.old_icon_piece_color_3 == 'white':
                                    self.icon_changed_3.setIcon(self.icons_array_white[0])
                                elif self.old_icon_piece_color_3 == 'black':
                                    self.icon_changed_3.setIcon(self.icons_array_white[1])
                            elif self.old_icon_piece_3 == 'rook':
                                if self.old_icon_piece_color_3 == 'white':
                                    self.icon_changed_3.setIcon(self.icons_array_white[2])
                                elif self.old_icon_piece_color_3 == 'black':
                                    self.icon_changed_3.setIcon(self.icons_array_white[3])
                            elif self.old_icon_piece_3 == 'knight':
                                if self.old_icon_piece_color_3 == 'white':
                                    self.icon_changed_3.setIcon(self.icons_array_white[4])
                                elif self.old_icon_piece_color_3 == 'black':
                                    self.icon_changed_3.setIcon(self.icons_array_white[5])
                            elif self.old_icon_piece_3 == 'bishop':
                                if self.old_icon_piece_color_3 == 'white':
                                    self.icon_changed_3.setIcon(self.icons_array_white[6])
                                elif self.old_icon_piece_color_3 == 'black':
                                    self.icon_changed_3.setIcon(self.icons_array_white[7])
                            elif self.old_icon_piece_3 == 'king':
                                if self.old_icon_piece_color_3 == 'white':
                                    self.icon_changed_3.setIcon(self.icons_array_white[10])
                                elif self.old_icon_piece_color_3 == 'black':
                                    self.icon_changed_3.setIcon(self.icons_array_white[11])
                            elif self.old_icon_piece_3 == 'queen':
                                if self.old_icon_piece_color_3 == 'white':
                                    self.icon_changed_3.setIcon(self.icons_array_white[8])
                                elif self.old_icon_piece_color_3 == 'black':
                                    self.icon_changed_3.setIcon(self.icons_array_white[9])
                        elif self.old_icon_color_3 == 'black':
                            if self.old_icon_piece_3 == 'pawn':
                                if self.old_icon_piece_color_3 == 'white':
                                    self.icon_changed_3.setIcon(self.icons_array_black[0])
                                elif self.old_icon_piece_color_3 == 'black':
                                    self.icon_changed_3.setIcon(self.icons_array_black[1])
                            elif self.old_icon_piece_3 == 'rook':
                                if self.old_icon_piece_color_3 == 'white':
                                    self.icon_changed_3.setIcon(self.icons_array_black[2])
                                elif self.old_icon_piece_color_3 == 'black':
                                    self.icon_changed_3.setIcon(self.icons_array_black[3])
                            elif self.old_icon_piece_3 == 'knight':
                                if self.old_icon_piece_color_3 == 'white':
                                    self.icon_changed_3.setIcon(self.icons_array_black[4])
                                elif self.old_icon_piece_color_3 == 'black':
                                    self.icon_changed_3.setIcon(self.icons_array_black[5])
                            elif self.old_icon_piece_3 == 'bishop':
                                if self.old_icon_piece_color_3 == 'white':
                                    self.icon_changed_3.setIcon(self.icons_array_black[6])
                                elif self.old_icon_piece_color_3 == 'black':
                                    self.icon_changed_3.setIcon(self.icons_array_black[7])
                            elif self.old_icon_piece_3 == 'king':
                                if self.old_icon_piece_color_3 == 'white':
                                    self.icon_changed_3.setIcon(self.icons_array_black[10])
                                elif self.old_icon_piece_color_3 == 'black':
                                    self.icon_changed_3.setIcon(self.icons_array_black[11])
                            elif self.old_icon_piece_3 == 'queen':
                                if self.old_icon_piece_color_3 == 'white':
                                    self.icon_changed_3.setIcon(self.icons_array_black[8])
                                elif self.old_icon_piece_color_3 == 'black':
                                    self.icon_changed_3.setIcon(self.icons_array_black[9])

                    if self.icon_changed_4 != 'none':
                        if self.old_icon_color_4 == 'white':
                            if self.old_icon_piece_4 == 'pawn':
                                if self.old_icon_piece_color_4 == 'white':
                                    self.icon_changed_4.setIcon(self.icons_array_white[0])
                                elif self.old_icon_piece_color_4 == 'black':
                                    self.icon_changed_4.setIcon(self.icons_array_white[1])
                            elif self.old_icon_piece_4 == 'rook':
                                if self.old_icon_piece_color_4 == 'white':
                                    self.icon_changed_4.setIcon(self.icons_array_white[2])
                                elif self.old_icon_piece_color_4 == 'black':
                                    self.icon_changed_4.setIcon(self.icons_array_white[3])
                            elif self.old_icon_piece_4 == 'knight':
                                if self.old_icon_piece_color_4 == 'white':
                                    self.icon_changed_4.setIcon(self.icons_array_white[4])
                                elif self.old_icon_piece_color_4 == 'black':
                                    self.icon_changed_4.setIcon(self.icons_array_white[5])
                            elif self.old_icon_piece_4 == 'bishop':
                                if self.old_icon_piece_color_4 == 'white':
                                    self.icon_changed_4.setIcon(self.icons_array_white[6])
                                elif self.old_icon_piece_color_4 == 'black':
                                    self.icon_changed_4.setIcon(self.icons_array_white[7])
                            elif self.old_icon_piece_4 == 'king':
                                if self.old_icon_piece_color_4 == 'white':
                                    self.icon_changed_4.setIcon(self.icons_array_white[10])
                                elif self.old_icon_piece_color_4 == 'black':
                                    self.icon_changed_4.setIcon(self.icons_array_white[11])
                            elif self.old_icon_piece_4 == 'queen':
                                if self.old_icon_piece_color_4 == 'white':
                                    self.icon_changed_4.setIcon(self.icons_array_white[8])
                                elif self.old_icon_piece_color_4 == 'black':
                                    self.icon_changed_4.setIcon(self.icons_array_white[9])
                        elif self.old_icon_color_4 == 'black':
                            if self.old_icon_piece_4 == 'pawn':
                                if self.old_icon_piece_color_4 == 'white':
                                    self.icon_changed_4.setIcon(self.icons_array_black[0])
                                elif self.old_icon_piece_color_4 == 'black':
                                    self.icon_changed_4.setIcon(self.icons_array_black[1])
                            elif self.old_icon_piece_4 == 'rook':
                                if self.old_icon_piece_color_4 == 'white':
                                    self.icon_changed_4.setIcon(self.icons_array_black[2])
                                elif self.old_icon_piece_color_4 == 'black':
                                    self.icon_changed_4.setIcon(self.icons_array_black[3])
                            elif self.old_icon_piece_4 == 'knight':
                                if self.old_icon_piece_color_4 == 'white':
                                    self.icon_changed_4.setIcon(self.icons_array_black[4])
                                elif self.old_icon_piece_color_4 == 'black':
                                    self.icon_changed_4.setIcon(self.icons_array_black[5])
                            elif self.old_icon_piece_4 == 'bishop':
                                if self.old_icon_piece_color_4 == 'white':
                                    self.icon_changed_4.setIcon(self.icons_array_black[6])
                                elif self.old_icon_piece_color_4 == 'black':
                                    self.icon_changed_4.setIcon(self.icons_array_black[7])
                            elif self.old_icon_piece_4 == 'king':
                                if self.old_icon_piece_color_4 == 'white':
                                    self.icon_changed_4.setIcon(self.icons_array_black[10])
                                elif self.old_icon_piece_color_4 == 'black':
                                    self.icon_changed_4.setIcon(self.icons_array_black[11])
                            elif self.old_icon_piece_4 == 'queen':
                                if self.old_icon_piece_color_4 == 'white':
                                    self.icon_changed_4.setIcon(self.icons_array_black[8])
                                elif self.old_icon_piece_color_4 == 'black':
                                    self.icon_changed_4.setIcon(self.icons_array_black[9])
                #Change the yellow selection flag
                self.yellow_selection_flag = not self.yellow_selection_flag

        elif state == 70: #Manual mode.
            with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                    live_situation = yaml.load(live_file.read())
            if action ==  'delete':
                square_object = live_situation[square][0]
                if square_object == 'none':
                        rospy.logwarn('There is no piece to delete in the selected square')
                else:
                    #Save the color of the square
                    square_color = self.chessboard_colors[square]
                    if square_color == 'white':
                        self.array[self.dict_squares[square]].setIcon(QtGui.QIcon())
                        #Change the live chessboard situation
                        live_situation[square][0] = 'none'
                        live_situation[square][1] = 'none'
                        with open(self.dir_live_chessboard_situation, "w") as t_p:
                            yaml.dump(live_situation, t_p)
                    elif square_color == 'black':
                        self.array[self.dict_squares[square]].setIcon(self.icon_grey)
                        #Change the live chessboard situation
                        live_situation[square][0] = 'none'
                        live_situation[square][1] = 'none'
                        with open(self.dir_live_chessboard_situation, "w") as t_p:
                            yaml.dump(live_situation, t_p)

            else: #If the action is to manually place a piece in the chessboard.
                #Save the color of the square
                square_color = self.chessboard_colors[square]

                #Save the pieces that are currently in the live situation
                present_bishops_w = []
                present_rooks_w = []
                present_knights_w = []
                present_pawns_w = []
                present_bishops_b = []
                present_rooks_b = []
                present_knights_b = []
                present_pawns_b = []

                for casella in live_situation:
                    if casella[0] == 'bishop_c1' or casella[0] == 'bishop_f1':
                        present_bishops_w.append(casella[0])
                    elif casella[0] == 'bishop_c8' or casella[0] == 'bishop_f8':
                        present_bishops_b.append(casella[0])
                    elif casella[0] == 'knight_b1' or casella[0] == 'knight_g1':
                        present_knights_w.append(casella[0])
                    elif casella[0] == 'knight_b8' or casella[0] == 'knight_g8':
                        present_knights_b.append(casella[0])
                    elif casella[0] == 'rook_a1' or casella[0] == 'rook_h1':
                        present_rooks_w.append(casella[0])
                    elif casella[0] == 'rook_a8' or casella[0] == 'rook_h8':
                        present_rooks_b.append(casella[0])
                    elif casella[0] == 'pawn_a2' or casella[0] == 'pawn_b2' or casella[0] == 'pawn_c2' or casella[0] == 'pawn_d2' or casella[0] == 'pawn_e2' or casella[0] == 'pawn_f2' or casella[0] == 'pawn_g2' or casella[0] == 'pawn_h2':
                        present_pawns_w.append(casella[0])
                    elif casella[0] == 'pawn_a7' or casella[0] == 'pawn_b7' or casella[0] == 'pawn_c7' or casella[0] == 'pawn_d7' or casella[0] == 'pawn_e7' or casella[0] == 'pawn_f7' or casella[0] == 'pawn_g7' or casella[0] == 'pawn_h7':
                        present_pawns_b.append(casella[0])

                if square_color == 'white':
                    if action == 'w_pawn':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_white[0])
                        #Change the live chessboard situation
                        if square_object in pawns_w:
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'white'
                        else:
                            if len(present_pawns_w) == 0:
                                live_situation[square][0] = 'pawn_a2'
                                live_situation[square][1] = 'white'
                            elif len(present_pawns_w) == 8:
                                live_situation[square][0] = 'inserted_pawn'
                                live_situation[square][1] = 'white'
                            else:
                                self.white_pawns = cfg_w.white_pawns
                                for pawn in self.white_pawns:
                                    if pawn not in present_pawns_w:
                                        live_situation[square][0] = pawn
                                        live_situation[square][1] = 'white'
                                        break
                    elif action == 'b_pawn':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_white[1])
                        #Change the live chessboard situation
                        if square_object in pawns_b:
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'black'
                        else:
                            if len(present_pawns_b) == 0:
                                live_situation[square][0] = 'pawn_a7'
                                live_situation[square][1] = 'black'
                            elif len(present_pawns_b) == 8:
                                live_situation[square][0] = 'inserted_pawn'
                                live_situation[square][1] = 'black'
                            else:
                                self.black_pawns = cfg_w.black_pawns
                                for pawn in self.black_pawns:
                                    if pawn not in present_pawns_b:
                                        live_situation[square][0] = pawn
                                        live_situation[square][1] = 'black'
                                        break
                    elif action == 'w_knight':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_white[4])
                        #Change the live chessboard situation
                        if square_object == 'knight_g1' or square_object == 'knight_b1':
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'white'
                        else:
                            if len(present_knights_w) == 1:
                                live_situation[square][0] = present_knights_w[0]
                                live_situation[square][1] = 'white'
                            elif len(present_knights_w) == 2:
                                live_situation[square][0] = 'inserted_knight'
                                live_situation[square][1] = 'white'
                            elif len(present_knights_w) == 0:
                                live_situation[square][0] = 'knight_b1'
                                live_situation[square][1] = 'white'                    
                    elif action == 'b_knight':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_white[5])
                        #Change the live chessboard situation
                        if square_object == 'knight_g8' or square_object == 'knight_b8':
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'black'
                        else:
                            if len(present_knights_b) == 1:
                                live_situation[square][0] = present_knights_b[0]
                                live_situation[square][1] = 'black'
                            elif len(present_knights_b) == 2:
                                live_situation[square][0] = 'inserted_knight'
                                live_situation[square][1] = 'black'
                            elif len(present_knights_b) == 0:
                                live_situation[square][0] = 'knight_b8'
                                live_situation[square][1] = 'black'                        
                    elif action == 'w_bishop':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_white[6])
                        #Change the live chessboard situation
                        if square_object == 'bishop_c1' or square_object == 'bishop_f1':
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'white'
                        else:
                            if len(present_bishops_w) == 1:
                                live_situation[square][0] = present_bishops_w[0]
                                live_situation[square][1] = 'white'
                            elif len(present_bishops_w) == 2:
                                live_situation[square][0] = 'inserted_bishop'
                                live_situation[square][1] = 'white'
                            elif len(present_bishops_w) == 0:
                                live_situation[square][0] = 'bishop_c1'
                                live_situation[square][1] = 'white'                        
                    elif action == 'b_bishop':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_white[7])
                        #Change the live chessboard situation
                        if square_object == 'bishop_c8' or square_object == 'bishop_f8':
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'black'
                        else:
                            if len(present_bishops_b) == 1:
                                live_situation[square][0] = present_bishops_b[0]
                                live_situation[square][1] = 'black'
                            elif len(present_bishops_b) == 2:
                                live_situation[square][0] = 'inserted_bishop'
                                live_situation[square][1] = 'black'
                            elif len(present_bishops_b) == 0:
                                live_situation[square][0] = 'bishop_c8'
                                live_situation[square][1] = 'black'                        
                    elif action == 'w_rook':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_white[2])
                        #Change the live chessboard situation
                        if square_object == 'rook_a1' or square_object == 'rook_h1':
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'white'
                        else:
                            if len(present_rooks_w) == 1:
                                live_situation[square][0] = present_rooks_w[0]
                                live_situation[square][1] = 'white'
                            elif len(present_rooks_w) == 2:
                                live_situation[square][0] = 'inserted_rook'
                                live_situation[square][1] = 'white'
                            elif len(present_rooks_w) == 0:
                                live_situation[square][0] = 'rook_a1'
                                live_situation[square][1] = 'white'                        
                    elif action == 'b_rook':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_white[3])
                        #Change the live chessboard situation
                        if square_object == 'rook_a8' or square_object == 'rook_h8':
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'black'
                        else:
                            if len(present_rooks_b) == 1:
                                live_situation[square][0] = present_rooks_b[0]
                                live_situation[square][1] = 'black'
                            elif len(present_rooks_b) == 2:
                                live_situation[square][0] = 'inserted_rook'
                                live_situation[square][1] = 'black'
                            elif len(present_rooks_b) == 0:
                                live_situation[square][0] = 'rook_a8'
                                live_situation[square][1] = 'black'                       
                    elif action == 'w_queen':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_white[8])
                        #Change the live chessboard situation
                        live_situation[square][0] = 'queen_d1'
                        live_situation[square][1] = 'white'                        
                    elif action == 'b_queen':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_white[9])
                        #Change the live chessboard situation
                        live_situation[square][0] = 'queen_d8'
                        live_situation[square][1] = 'black'                        
                    elif action == 'w_king':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_white[10])
                        #Change the live chessboard situation
                        live_situation[square][0] = 'king_e1'
                        live_situation[square][1] = 'white'                        
                    elif action == 'b_king':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_white[11])
                        #Change the live chessboard situation
                        live_situation[square][0] = 'king_e8'
                        live_situation[square][1] = 'black'                        
                elif square_color == 'black':
                    if action == 'w_pawn':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_black[0])
                        #Change the live chessboard situation
                        if square_object in pawns_w:
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'white'
                        else:
                            if len(present_pawns_w) == 0:
                                live_situation[square][0] = 'pawn_a2'
                                live_situation[square][1] = 'white'
                            elif len(present_pawns_w) == 8:
                                live_situation[square][0] = 'inserted_pawn'
                                live_situation[square][1] = 'white'
                            else:
                                self.white_pawns = cfg_w.white_pawns
                                for pawn in self.white_pawns:
                                    if pawn not in present_pawns_w:
                                        live_situation[square][0] = pawn
                                        live_situation[square][1] = 'white'
                                        break
                    elif action == 'b_pawn':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_black[1])
                        #Change the live chessboard situation
                        if square_object in pawns_b:
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'black'
                        else:
                            if len(present_pawns_b) == 0:
                                live_situation[square][0] = 'pawn_a7'
                                live_situation[square][1] = 'black'
                            elif len(present_pawns_b) == 8:
                                live_situation[square][0] = 'inserted_pawn'
                                live_situation[square][1] = 'black'
                            else:
                                self.black_pawns = cfg_w.black_pawns
                                for pawn in self.black_pawns:
                                    if pawn not in present_pawns_b:
                                        live_situation[square][0] = pawn
                                        live_situation[square][1] = 'black'
                                        break
                    elif action == 'w_knight':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_black[4])
                        #Change the live chessboard situation
                        if square_object == 'knight_g1' or square_object == 'knight_b1':
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'white'
                        else:
                            if len(present_knights_w) == 1:
                                live_situation[square][0] = present_knights_w[0]
                                live_situation[square][1] = 'white'
                            elif len(present_knights_w) == 2:
                                live_situation[square][0] = 'inserted_knight'
                                live_situation[square][1] = 'white'
                            elif len(present_knights_w) == 0:
                                live_situation[square][0] = 'knight_b1'
                                live_situation[square][1] = 'white'                    
                    elif action == 'b_knight':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_black[5])
                        #Change the live chessboard situation
                        if square_object == 'knight_g8' or square_object == 'knight_b8':
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'black'
                        else:
                            if len(present_knights_b) == 1:
                                live_situation[square][0] = present_knights_b[0]
                                live_situation[square][1] = 'black'
                            elif len(present_knights_b) == 2:
                                live_situation[square][0] = 'inserted_knight'
                                live_situation[square][1] = 'black'
                            elif len(present_knights_b) == 0:
                                live_situation[square][0] = 'knight_b8'
                                live_situation[square][1] = 'black'                        
                    elif action == 'w_bishop':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_black[6])
                        #Change the live chessboard situation
                        if square_object == 'bishop_c1' or square_object == 'bishop_f1':
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'white'
                        else:
                            if len(present_bishops_w) == 1:
                                live_situation[square][0] = present_bishops_w[0]
                                live_situation[square][1] = 'white'
                            elif len(present_bishops_w) == 2:
                                live_situation[square][0] = 'inserted_bishop'
                                live_situation[square][1] = 'white'
                            elif len(present_bishops_w) == 0:
                                live_situation[square][0] = 'bishop_c1'
                                live_situation[square][1] = 'white'                        
                    elif action == 'b_bishop':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_black[7])
                        #Change the live chessboard situation
                        if square_object == 'bishop_c8' or square_object == 'bishop_f8':
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'black'
                        else:
                            if len(present_bishops_b) == 1:
                                live_situation[square][0] = present_bishops_b[0]
                                live_situation[square][1] = 'black'
                            elif len(present_bishops_b) == 2:
                                live_situation[square][0] = 'inserted_bishop'
                                live_situation[square][1] = 'black'
                            elif len(present_bishops_b) == 0:
                                live_situation[square][0] = 'bishop_c8'
                                live_situation[square][1] = 'black'                        
                    elif action == 'w_rook':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_black[2])
                        #Change the live chessboard situation
                        if square_object == 'rook_a1' or square_object == 'rook_h1':
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'white'
                        else:
                            if len(present_rooks_w) == 1:
                                live_situation[square][0] = present_rooks_w[0]
                                live_situation[square][1] = 'white'
                            elif len(present_rooks_w) == 2:
                                live_situation[square][0] = 'inserted_rook'
                                live_situation[square][1] = 'white'
                            elif len(present_rooks_w) == 0:
                                live_situation[square][0] = 'rook_a1'
                                live_situation[square][1] = 'white'                        
                    elif action == 'b_rook':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_black[3])
                        #Change the live chessboard situation
                        if square_object == 'rook_a8' or square_object == 'rook_h8':
                            live_situation[square][0] = square_object
                            live_situation[square][1] = 'black'
                        else:
                            if len(present_rooks_b) == 1:
                                live_situation[square][0] = present_rooks_b[0]
                                live_situation[square][1] = 'black'
                            elif len(present_rooks_b) == 2:
                                live_situation[square][0] = 'inserted_rook'
                                live_situation[square][1] = 'black'
                            elif len(present_rooks_b) == 0:
                                live_situation[square][0] = 'rook_a8'
                                live_situation[square][1] = 'black'                       
                    elif action == 'w_queen':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_black[10])
                        #Change the live chessboard situation
                        live_situation[square][0] = 'queen_d1'
                        live_situation[square][1] = 'white'                        
                    elif action == 'b_queen':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_black[11])
                        #Change the live chessboard situation
                        live_situation[square][0] = 'queen_d8'
                        live_situation[square][1] = 'black'                        
                    elif action == 'w_king':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_black[8])
                        #Change the live chessboard situation
                        live_situation[square][0] = 'king_e1'
                        live_situation[square][1] = 'white'                        
                    elif action == 'b_king':
                        self.array[self.dict_squares[square]].setIcon(self.icons_array_black[9])
                        #Change the live chessboard situation
                        live_situation[square][0] = 'king_e8'
                        live_situation[square][1] = 'black'
                with open(self.dir_live_chessboard_situation, "w") as t_p:
                        yaml.dump(live_situation, t_p)

    def Promotion(self, data):
        global casella_end
        global casella_start
        self.promoted_piece_msg = data.data
        self.promotion_happened = True

        #destination_color_promotion = self.chessboard_colors[self.opponent_move_end_square]
        with open(self.dir_live_chessboard_situation, 'rb') as live_file:
            live_situation = yaml.load(live_file.read())

        if state == 16 or state == 17 or state == 11: #Promotion is executed by the opponent
            casella_end = self.opponent_move_end_square
            casella_start = self.opponent_move_start_square
            destination_color_promotion = self.chessboard_colors[self.opponent_move_end_square]
        elif state == 12: #Promotion is executed by TIAGo.
            casella_end = self.current_focus
            casella_start = self.current_red_icon
            destination_color_promotion = self.chessboard_colors[casella_end]

        if self.color_to_move == 'white' and destination_color_promotion == 'white':
            if self.promoted_piece_msg == 'rook':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_white[2])
                    live_situation[casella_end][0] = 'rook_a1'
            elif self.promoted_piece_msg == 'knight':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_white[4])
                    live_situation[casella_end][0] = 'knight_b1'
            elif self.promoted_piece_msg == 'bishop':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_white[6])
                    live_situation[casella_end][0] = 'bishop_c1'
            elif self.promoted_piece_msg == 'queen':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_white[8])
                    live_situation[casella_end][0] = 'queen_d1'
            self.close_promotion.emit(1)

        elif self.color_to_move == 'white' and destination_color_promotion == 'black':
            if self.promoted_piece_msg == 'rook':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_black[2])
                    live_situation[casella_end][0] = 'rook_a1'
            elif self.promoted_piece_msg == 'knight':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_black[4])
                    live_situation[casella_end][0] = 'knight_b1'
            elif self.promoted_piece_msg == 'bishop':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_black[6])
                    live_situation[casella_end][0] = 'bishop_c1'
            elif self.promoted_piece_msg == 'queen':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_black[8])
                    live_situation[casella_end][0] = 'queen_d1'
            self.close_promotion.emit(1)

        elif self.color_to_move == 'black' and destination_color_promotion == 'white':
            if self.promoted_piece_msg == 'rook':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_white[3])
                    live_situation[casella_end][0] = 'rook_a8'
            elif self.promoted_piece_msg == 'knight':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_white[5])
                    live_situation[casella_end][0] = 'knight_b8'
            elif self.promoted_piece_msg == 'bishop':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_white[7])
                    live_situation[casella_end][0] = 'bishop_c8'
            elif self.promoted_piece_msg == 'queen':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_white[9])
                    live_situation[casella_end][0] = 'queen_d8'
            self.close_promotion.emit(2)

        elif self.color_to_move == 'black' and destination_color_promotion == 'black':
            if self.promoted_piece_msg == 'rook':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_black[3])
                    live_situation[casella_end][0] = 'rook_a8'
            elif self.promoted_piece_msg == 'knight':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_black[5])
                    live_situation[casella_end][0] = 'knight_b8'
            elif self.promoted_piece_msg == 'bishop':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_black[7])
                    live_situation[casella_end][0] = 'bishop_c8'
            elif self.promoted_piece_msg == 'queen':
                    self.array[self.dict_squares[casella_end]].setIcon(self.icons_array_black[9])
                    live_situation[casella_end][0] = 'queen_d8'
            self.close_promotion.emit(2)

        live_situation[casella_end][1] = self.color_to_move

        with open(self.dir_live_chessboard_situation, "w") as t_p:
                    yaml.dump(live_situation, t_p)
        #######
        #Update the planning scene.
        with open(simul_config) as file:
            square_centers = yaml.load(file)
        self.populate_pieces(square_centers, live_situation)
        ###########

        self.promotion_happened = False

    def PromotionHappened(self, data):
        #Change the promotion flag
        self.opponent_promotion_happened = True
        self.color_to_move = data.data
        #Make the user choose the promoted piece
        if self.color_to_move == 'white':
            self.promotion.emit(1)
        elif self.color_to_move == 'black':
            self.promotion.emit(2)

    def OpponentMoveStartSquare(self, data):
        self.opponent_move_start_square = data.data
        state_publisher.publish(17) #Pass to the state of GUI updating.

        if self.opponent_move_start_square == 'castle':
            pass

        else:
            #Save the color of the starting square
            starting_color = self.chessboard_colors[self.opponent_move_start_square]

            #Change the starting square icon
            if starting_color == 'white':
                self.array[self.dict_squares[self.opponent_move_start_square]].setIcon(QtGui.QIcon())
            elif starting_color == 'black':
                self.array[self.dict_squares[self.opponent_move_start_square]].setIcon(self.icon_grey)

    def OpponentMoveEndSquare(self, data):
        self.opponent_move_end_square = data.data

        rospy.sleep(1)

        if self.opponent_move_end_square == 'castle':
            #Change the GUI icons.
            self.array[self.dict_squares[self.castle_squares[0]]].setIcon(self.icons_array_black[10])
            self.array[self.dict_squares[self.castle_squares[1]]].setIcon(self.icons_array_white[2])
            self.array[self.dict_squares['e1']].setIcon(self.icon_grey)
            if self.castle_squares[1] == 'f1':
                self.array[self.dict_squares['h1']].setIcon(QtGui.QIcon())
            elif self.castle_squares[1] == 'd1':
                self.array[self.dict_squares['a1']].setIcon(QtGui.QIcon())

            #Update the live chessboard situation
            with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                    live_situation = yaml.load(live_file.read())
            if self.castle_squares[0] == 'c1':
                self.array[self.dict_squares['a1']].setIcon(self.icon_grey)
                live_situation['c1'][0] = 'king_e1'
                live_situation['c1'][1] = 'white'
                live_situation['d1'][0] = 'rook_a1'
                live_situation['d1'][1] = 'white'
                live_situation['a1'][0] = 'none'
                live_situation['a1'][1] = 'none'
                live_situation['e1'][0] = 'none'
                live_situation['e1'][1] = 'none'
            elif self.castle_squares[0] == 'g1':
                self.array[self.dict_squares['h1']].setIcon(QtGui.QIcon())
                live_situation['g1'][0] = 'king_e1'
                live_situation['g1'][1] = 'white'
                live_situation['f1'][0] = 'rook_h1'
                live_situation['f1'][1] = 'white'
                live_situation['h1'][0] = 'none'
                live_situation['h1'][1] = 'none'
                live_situation['e1'][0] = 'none'
                live_situation['e1'][1] = 'none'

            with open(self.dir_live_chessboard_situation, "w") as t_p:
                    yaml.dump(live_situation, t_p)
            #######
            #Update the planning scene.
            with open(simul_config) as file:
                square_centers = yaml.load(file)
            self.populate_pieces(square_centers, live_situation)
            ###########
            if state != 16:
                state_publisher.publish(11) #Go back to the state of waiting for the move to perform with TIAGo.

        else:
            #Read which piece is in the currently highlighted square
            with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                    live_situation = yaml.load(live_file.read())
            self.piece_to_move = self.pieces_coordinates[live_situation[self.opponent_move_start_square][0]][1]['name']
            piece_to_capture = live_situation[self.opponent_move_end_square][0]
            self.color_to_move = live_situation[self.opponent_move_start_square][1]
            #Save the color of the destination square and of the starting square
            destination_color = self.chessboard_colors[self.opponent_move_end_square]
            starting_color = self.chessboard_colors[self.opponent_move_start_square]

            #Change the destination icon and the starting square icon
            if destination_color == 'black':
                    if self.piece_to_move == 'pawn':
                        if self.color_to_move == 'white':
                                self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_black[0])
                        elif self.color_to_move == 'black':
                                self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_black[1])
                    elif self.piece_to_move == 'rook':
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_black[2])
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_black[3])
                    elif self.piece_to_move == 'knight':
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_black[4])
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_black[5])
                    elif self.piece_to_move == 'bishop':
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_black[6])
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_black[7])
                    elif self.piece_to_move == 'queen':
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_black[8])
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_black[9])
                    elif self.piece_to_move == 'king':
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_black[10])
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_black[11])

            elif destination_color == 'white':
                    if self.piece_to_move == 'pawn':
                        if not self.opponent_promotion_happened:
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_white[0])
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_white[1])
                        elif self.opponent_promotion_happened:
                            self.opponent_promotion_happened = False #Set the flag back to false
                    elif self.piece_to_move == 'rook':
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_white[2])
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_white[3])
                    elif self.piece_to_move == 'knight':
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_white[4])
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_white[5])
                    elif self.piece_to_move == 'bishop':
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_white[6])
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_white[7])
                    elif self.piece_to_move == 'queen':
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_white[8])
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_white[9])
                    elif self.piece_to_move == 'king':
                            if self.color_to_move == 'white':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_white[10])
                            elif self.color_to_move == 'black':
                                    self.array[self.dict_squares[self.opponent_move_end_square]].setIcon(self.icons_array_white[11])

            #Update the live chessboard situation
            live_situation[self.opponent_move_end_square][0] = live_situation[self.opponent_move_start_square][0]
            if self.color_to_move == 'white':
                live_situation[self.opponent_move_end_square][1] = 'white'
            elif self.color_to_move == 'black':
                live_situation[self.opponent_move_end_square][1] = 'black'

            live_situation[self.opponent_move_start_square][0] = 'none'
            live_situation[self.opponent_move_start_square][1] = 'none'

            if self.en_passant_happened:
                #Save the color of the starting square
                square_color = self.chessboard_colors[self.en_passant_square]
                #Change the starting square icon
                if square_color == 'white':
                    self.array[self.dict_squares[self.en_passant_square]].setIcon(QtGui.QIcon())
                elif square_color == 'black':
                    self.array[self.dict_squares[self.en_passant_square]].setIcon(self.icon_grey)
                #Update the live chessboard situation
                live_situation[self.en_passant_square][0] = 'none'
                live_situation[self.en_passant_square][1] = 'none'
                self.en_passant_happened = False #Bring back the flag to False

            with open(self.dir_live_chessboard_situation, "w") as t_p:
                    yaml.dump(live_situation, t_p)

            #######
            #Update the planning scene.
            with open(simul_config) as file:
                square_centers = yaml.load(file)
            self.populate_pieces(square_centers, live_situation)
            ###########
            if state != 16:
                state_publisher.publish(11) #Go back to the state of waiting for the move to perform with TIAGo.

    def populate_pieces(self, base_centers, live_chessboard_situation):
        #Function to populate the planning scene with the pieces in their current position.
        #It receives as input the coordinates of the squares' centeres in the base_footprint reference frame and live_chessboard_situation that keeps track of the pieces positions thoughout the game.
        #This function needs to be used whenever TIAGo sees that a piece has been moved from the opponent.

        #In case there is something in the world form previous executions (maybe the robot was closer or further from the chessboard)
        for name in self.pieces_coordinates:
            self.scene.remove_world_object(name)

        key_list = list(live_chessboard_situation.keys())
        val = []
        for elem in key_list:
            val.append(live_chessboard_situation[elem][0])
        val_list = list(val)

        #Add cylinders corresponding to the pieces to the planning scene.
        for name in self.pieces_coordinates:
            if name in val_list:
                position = val_list.index(name) #Get the position (index) of the current piece in the chessboard
                square = key_list[position] #Get the square where the current piece is.
                pose = PoseStamped()
                pose.header.frame_id = 'base_footprint'
                pose.header.stamp = rospy.Time.now()
                pose.pose = Pose(Point(base_centers[self.squares_to_index[square]].point.x, base_centers[self.squares_to_index[square]].point.y, base_centers[self.squares_to_index[square]].point.z + self.pieces_coordinates[name][1]['height']/2), Quaternion(0, 0, 0, 1))
                self.scene.add_cylinder(name, pose, height = self.pieces_coordinates[name][1]['height'], radius = self.pieces_coordinates[name][1]['diameter']/2)

    def EnPassant(self, data):
        self.en_passant_square = data.data
        self.en_passant_happened = True

    def Castle(self, data):
        self.castle_square = data.data
        if self.castle_square == 'c1' or self.castle_square == 'd1':
            self.castle_squares = ['c1', 'd1']
        elif self.castle_square == 'g1' or self.castle_square == 'f1':
            self.castle_squares = ['g1', 'f1']

    def UpdateLiveSituation(self, data):
        if data.data == 14 and update_flag: #state of waiting for the opponent move.
            #Update the live chessboard situation with the move just executed by TIAGo.
            print('UPDATING............')
            #Update the live chessboard situation with the move just executed by TIAGo.
            with open(self.dir_live_chessboard_situation, 'rb') as live_file:
                live_situation = yaml.load(live_file.read())

            if not self.promotion_happened:
                live_situation[self.current_focus][0] = live_situation[self.current_red_icon][0]
                if self.color_to_move == 'white':
                    live_situation[self.current_focus][1] = 'white'
                elif self.color_to_move == 'black':
                    live_situation[self.current_focus][1] = 'black'
                if self.en_passant:
                        live_situation[square_captured_piece][0] = 'none'
                        live_situation[square_captured_piece][1] = 'none'
                if self.short_castle:
                        live_situation['f8'][0] = 'rook_h8'
                        live_situation['f8'][1] = 'black'
                        live_situation['h8'][0] = 'none'
                        live_situation['h8'][1] = 'none'
                if self.long_castle:
                        live_situation['d8'][0] = 'rook_a8'
                        live_situation['d8'][1] = 'black'
                        live_situation['a8'][0] = 'none'
                        live_situation['a8'][1] = 'none'
                live_situation[self.current_red_icon][0] = 'none'
                live_situation[self.current_red_icon][1] = 'none'
            '''
            elif self.promotion_happened:
                if self.promoted_piece_msg == 'rook':
                        live_situation[casella_end][0] = 'rook_a1'
                        live_situation[casella_end][1] = 'white'
                        live_situation[casella_start][0] = 'none'
                        live_situation[casella_start][1] = 'none'
                elif self.promoted_piece_msg == 'knight':
                        live_situation[casella_end][0] = 'knight_b1'
                        live_situation[casella_end][1] = 'white'
                        live_situation[casella_start][0] = 'none'
                        live_situation[casella_start][1] = 'none'
                elif self.promoted_piece_msg == 'bishop':
                        live_situation[casella_end][0] = 'bishop_c1'
                        live_situation[casella_end][1] = 'white'
                        live_situation[casella_start][0] = 'none'
                        live_situation[casella_start][1] = 'none'
                elif self.promoted_piece_msg == 'queen':
                        live_situation[casella_end][0] = 'queen_d1'
                        live_situation[casella_end][1] = 'white'
                        live_situation[casella_start][0] = 'none'
                        live_situation[casella_start][1] = 'none'
                
                self.promotion_happened = False #Change back the flag
            '''

            self.promotion_happened = False #Change back the flag

            with open(self.dir_live_chessboard_situation, "w") as t_p:
                yaml.dump(live_situation, t_p)

            #######
            #Update the planning scene.
            with open(simul_config) as file:
                square_centers = yaml.load(file)
            self.populate_pieces(square_centers, live_situation)
            ###########

    def ManualMode(self):
        self.PlayGameModePushbutton.setEnabled(True)
        self.ManualModePushbutton.setEnabled(False)
        self.manual_mode.emit(1)
        state_publisher.publish(70)

    def BackToGame(self):
        self.PlayGameModePushbutton.setEnabled(False)
        self.ManualModePushbutton.setEnabled(True)
        self.back_to_game.emit(1)
        state_publisher.publish(11)


###################################################################


class WhitePromotionWindow(QtWidgets.QWidget, Ui_WhitePromotion):
    def __init__(self):
        super(WhitePromotionWindow, self).__init__()
        self.setupUi(self)
        #Subscribers initialization
        rospy.Subscriber("/cursor_step_promotion", Point, self.MoveHighlight)
        rospy.Subscriber("/button_promotion", Bool, self.SelectButton)

        #Publishers initialization
        self.promotion_publisher = rospy.Publisher('/promoted_piece', String, queue_size = 10)

        #variables initialization
        self.current_focus_promotion = 'queenPushButton'

        #Connect signals and slots
        self.rookPushButton.clicked.connect(self.rook)
        self.bishopPushButton.clicked.connect(self.bishop)
        self.knightPushButton.clicked.connect(self.knight)
        self.queenPushButton.clicked.connect(self.queen)

    def rook(self):
        self.promotion_publisher.publish('rook')

    def knight(self):
        self.promotion_publisher.publish('knight')

    def bishop(self):
        self.promotion_publisher.publish('bishop')

    def queen(self):
        self.promotion_publisher.publish('queen')

    def MoveHighlight(self, data):
        #If a movement in the x direction is detected, perform the corresponding movement on the GUI.
        if data.x == 1:
            #Change the highlighted button with the one on the right.
            if self.current_focus_promotion == 'queenPushButton':
                self.current_focus_promotion = 'rookPushButton'
                self.rookPushButton.setIcon(self.icon_rook_selected)
                self.queenPushButton.setIcon(self.icon_queen)
                self.rookPushButton.setFocus()
            elif self.current_focus_promotion == 'rookPushButton':
                self.current_focus_promotion = 'knightPushButton'
                self.knightPushButton.setIcon(self.icon_knight_selected)
                self.rookPushButton.setIcon(self.icon_rook)
                self.knightPushButton.setFocus()
            elif self.current_focus_promotion == 'knightPushButton':
                self.current_focus_promotion = 'bishopPushButton'
                self.bishopPushButton.setIcon(self.icon_bishop_selected)
                self.knightPushButton.setIcon(self.icon_knight)
                self.bishopPushButton.setFocus()
            elif self.current_focus_promotion == 'bishopPushButton':
                self.current_focus_promotion = 'queenPushButton'
                self.queenPushButton.setIcon(self.icon_queen_selected)
                self.bishopPushButton.setIcon(self.icon_bishop)
                self.queenPushButton.setFocus()

            print('CURRENT HIGHLIGHT: ' + str(self.current_focus_promotion))

        elif data.x == 2:
            #Change the highlighted button with the one on the left.
            if self.current_focus_promotion == 'queenPushButton':
                self.current_focus_promotion = 'bishopPushButton'
                self.bishopPushButton.setIcon(self.icon_bishop_selected)
                self.queenPushButton.setIcon(self.icon_queen)
                self.bishopPushButton.setFocus()
            elif self.current_focus_promotion == 'rookPushButton':
                self.current_focus_promotion = 'queenPushButton'
                self.queenPushButton.setIcon(self.icon_queen_selected)
                self.rookPushButton.setIcon(self.icon_rook)
                self.queenPushButton.setFocus()
            elif self.current_focus_promotion == 'knightPushButton':
                self.current_focus_promotion = 'rookPushButton'
                self.rookPushButton.setIcon(self.icon_rook_selected)
                self.knightPushButton.setIcon(self.icon_knight)
                self.rookPushButton.setFocus()
            elif self.current_focus_promotion == 'bishopPushButton':
                self.current_focus_promotion = 'knightPushButton'
                self.knightPushButton.setIcon(self.icon_knight_selected)
                self.bishopPushButton.setIcon(self.icon_bishop)
                self.knightPushButton.setFocus()

            print('CURRENT HIGHLIGHT: ' + str(self.current_focus_promotion))

    def SelectButton(self, data):
        if data:
            if self.current_focus_promotion == 'queenPushButton':
                self.queen()
            elif self.current_focus_promotion == 'rookPushButton':
                self.rook()
            elif self.current_focus_promotion == 'bishopPushButton':
                self.bishop()
            elif self.current_focus_promotion == 'knightPushButton':
                self.knight()


###################################################################


class BlackPromotionWindow(QtWidgets.QWidget, Ui_BlackPromotion):
    def __init__(self):
        super(BlackPromotionWindow, self).__init__()
        self.setupUi(self)

        #Subscribers initialization
        rospy.Subscriber("/cursor_step_promotion", Point, self.MoveHighlight)
        rospy.Subscriber("/button_promotion", Bool, self.SelectButton)

        #Publishers initialization
        self.promotion_publisher = rospy.Publisher('/promoted_piece', String, queue_size = 10)

        #Variables initialization
        self.current_focus_promotion = 'queenPushButton'

        #Connect signals and slots
        self.rookPushButton.clicked.connect(self.rook)
        self.bishopPushButton.clicked.connect(self.bishop)
        self.knightPushButton.clicked.connect(self.knight)
        self.queenPushButton.clicked.connect(self.queen)

    def rook(self):
        self.promotion_publisher.publish('rook')

    def knight(self):
        self.promotion_publisher.publish('knight')

    def bishop(self):
        self.promotion_publisher.publish('bishop')

    def queen(self):
        self.promotion_publisher.publish('queen')

    def MoveHighlight(self, data):
        #If a movement in the x direction is detected, perform the corresponding movement on the GUI.
        if data.x == 1:
            #Change the highlighted button with the one on the right.
            if self.current_focus_promotion == 'queenPushButton':
                self.current_focus_promotion = 'rookPushButton'
                self.rookPushButton.setIcon(self.icon_rook_selected)
                self.queenPushButton.setIcon(self.icon_queen)
                self.rookPushButton.setFocus()
            elif self.current_focus_promotion == 'rookPushButton':
                self.current_focus_promotion = 'knightPushButton'
                self.knightPushButton.setIcon(self.icon_knight_selected)
                self.rookPushButton.setIcon(self.icon_rook)
                self.knightPushButton.setFocus()
            elif self.current_focus_promotion == 'knightPushButton':
                self.current_focus_promotion = 'bishopPushButton'
                self.bishopPushButton.setIcon(self.icon_bishop_selected)
                self.knightPushButton.setIcon(self.icon_knight)
                self.bishopPushButton.setFocus()
            elif self.current_focus_promotion == 'bishopPushButton':
                self.current_focus_promotion = 'queenPushButton'
                self.queenPushButton.setIcon(self.icon_queen_selected)
                self.bishopPushButton.setIcon(self.icon_bishop)
                self.queenPushButton.setFocus()

            print('CURRENT HIGHLIGHT: ' + str(self.current_focus_promotion))

        elif data.x == 2:
            #Change the highlighted button with the one on the left.
            if self.current_focus_promotion == 'queenPushButton':
                self.current_focus_promotion = 'bishopPushButton'
                self.bishopPushButton.setIcon(self.icon_bishop_selected)
                self.queenPushButton.setIcon(self.icon_queen)
                self.bishopPushButton.setFocus()
            elif self.current_focus_promotion == 'rookPushButton':
                self.current_focus_promotion = 'queenPushButton'
                self.queenPushButton.setIcon(self.icon_queen_selected)
                self.rookPushButton.setIcon(self.icon_rook)
                self.queenPushButton.setFocus()
            elif self.current_focus_promotion == 'knightPushButton':
                self.current_focus_promotion = 'rookPushButton'
                self.rookPushButton.setIcon(self.icon_rook_selected)
                self.knightPushButton.setIcon(self.icon_knight)
                self.rookPushButton.setFocus()
            elif self.current_focus_promotion == 'bishopPushButton':
                self.current_focus_promotion = 'knightPushButton'
                self.knightPushButton.setIcon(self.icon_knight_selected)
                self.bishopPushButton.setIcon(self.icon_bishop)
                self.knightPushButton.setFocus()

            print('CURRENT HIGHLIGHT: ' + str(self.current_focus_promotion))

    def SelectButton(self, data):
        if data:
            if self.current_focus_promotion == 'queenPushButton':
                self.queen()
            elif self.current_focus_promotion == 'rookPushButton':
                self.rook()
            elif self.current_focus_promotion == 'bishopPushButton':
                self.bishop()
            elif self.current_focus_promotion == 'knightPushButton':
                self.knight()


###################################################################


class ManualMode(QtWidgets.QWidget, Ui_ManualModeWindow):
    clicked = QtCore.pyqtSignal(int)
    def __init__(self):
        super(ManualMode, self).__init__()
        self.setupUi(self)

        #Connect signals and slots.
        self.BlackKnightPushbutton.clicked.connect(self.BKnightBtnClicked)
        self.WhiteKnightPushbutton.clicked.connect(self.WKnightBtnClicked)
        self.BlackBishopPushbutton.clicked.connect(self.BBishopBtnClicked)
        self.WhiteBishopPushbutton.clicked.connect(self.WBishopBtnClicked)
        self.BlackRookPushbutton.clicked.connect(self.BRookBtnClicked)
        self.WhiteRookPushbutton.clicked.connect(self.WRookBtnClicked)
        self.BlackQueenPushbutton.clicked.connect(self.BQueenBtnClicked)
        self.WhiteQueenPushbutton.clicked.connect(self.WQueenBtnClicked)
        self.BlackKingPushbutton.clicked.connect(self.BKingBtnClicked)
        self.WhiteKingPushbutton.clicked.connect(self.WKingBtnClicked)
        self.BlackPawnPushbutton.clicked.connect(self.BPawnBtnClicked)
        self.WhitePawnPushbutton.clicked.connect(self.WPawnBtnClicked)
        self.EmptySquarePushbutton.clicked.connect(self.EmptySquare)

    def BKnightBtnClicked(self):
        rospy.loginfo('Black knight')
        self.clicked.emit(4)

    def WKnightBtnClicked(self):
        rospy.loginfo('White knight')
        self.clicked.emit(3)

    def BRookBtnClicked(self):
        rospy.loginfo('Black rook')
        self.clicked.emit(8)

    def WRookBtnClicked(self):
        rospy.loginfo('White rook')
        self.clicked.emit(7)

    def BBishopBtnClicked(self):
        rospy.loginfo('Black bishop')
        self.clicked.emit(6)

    def WBishopBtnClicked(self):
        rospy.loginfo('White bishop')
        self.clicked.emit(5)

    def BQueenBtnClicked(self):
        rospy.loginfo('Black queen')
        self.clicked.emit(10)

    def WQueenBtnClicked(self):
        rospy.loginfo('White queen')
        self.clicked.emit(9)

    def BKingBtnClicked(self):
        rospy.loginfo('Black king')
        self.clicked.emit(12)

    def WKingBtnClicked(self):
        rospy.loginfo('White king')
        self.clicked.emit(11)

    def BPawnBtnClicked(self):
        rospy.loginfo('Black pawn')
        self.clicked.emit(2)

    def WPawnBtnClicked(self):
        rospy.loginfo('White pawn')
        self.clicked.emit(1)

    def EmptySquare(self):
        rospy.loginfo('empty square')
        self.clicked.emit(13)



###################################################################


class WizardWindow(QtWidgets.QWidget, Ui_WizardWindow):
    close = QtCore.pyqtSignal(int)
    def __init__(self):
        super(WizardWindow, self).__init__()
        self.setupUi(self)

        #Connect signals and slots.
        self.NextPushButton.clicked.connect(self.GoToNext)
        self.ClosePushButton.clicked.connect(self.CloseWizard)
        self.PrevPushButton.clicked.connect(self.GoToPrev)

    def GoToNext(self):
        global current_window
        if current_window == 1:
            #Change Directions to the selection Directions.
            self.DirectionsLabel.setText("To move, select the desired piece with the joystick button or directly left-clicking with the mouse.\n"
"\n"
"The selected piece will turn red.")
            self.movie = QtGui.QMovie(GUI_PKG_DIR + "/images/GIFs/Selection_gif.gif")
            self.ImageLabel.setMinimumSize(QtCore.QSize(335, 300))
            self.ImageLabel.setMaximumSize(QtCore.QSize(335, 300))
            self.ImageLabel.setMovie(self.movie)
            self.movie.start()
            self.PrevPushButton.setEnabled(True)
            #Change the current window variable
            current_window = 2

        elif current_window == 2:
            #Change Directions to the confirm Directions.
            self.DirectionsLabel.setText("Click on the square intended as the selected piece destination.\n"
"The move will be highlightd in yellow.\n"
"\n"
"Click again on the destination square to confirm the move.\n"
"\n"
"If you want to change move, click on another square.")
            self.movie = QtGui.QMovie(GUI_PKG_DIR + "/images/GIFs/Confirm_move.gif")
            self.ImageLabel.setMinimumSize(QtCore.QSize(335, 300))
            self.ImageLabel.setMaximumSize(QtCore.QSize(335, 300))
            self.ImageLabel.setMovie(self.movie)
            self.movie.start()
            #Change the current window variable
            current_window = 3

        elif current_window == 3:
            #Change Directions to the promotion Directions.
            self.DirectionsLabel.setText("Whenever you or the opponent promote a pawn, a window will appear to choose the piece. Select the desired piece for your promotion or the piece chosen by the opponent for his promotion.\n"
"\n"
"Ask the opponent to place the piece chosen by you on the regular chessboard, eventually.")
            self.movie = QtGui.QMovie(GUI_PKG_DIR + "/images/GIFs/Promotion_gif.gif")
            self.ImageLabel.setMinimumSize(QtCore.QSize(335, 300))
            self.ImageLabel.setMaximumSize(QtCore.QSize(335, 300))
            self.ImageLabel.setMovie(self.movie)
            self.movie.start()
            #Change the current window variable
            current_window = 4

        elif current_window == 4:
            #Change Directions to the manual mode insert piece Directions.
            self.DirectionsLabel.setText("If TIAGo is wrong in identifiyng the executed opponent move, you can switch to manual mode to manually change the GUI chessboard.\n"
"\n"
"A window will appear: select the piece that you want to place and then click on the destination square.")
            self.movie = QtGui.QMovie(GUI_PKG_DIR +  "/images/GIFs/Manual_mode_add.gif")
            self.ImageLabel.setMinimumSize(QtCore.QSize(490, 300))
            self.ImageLabel.setMaximumSize(QtCore.QSize(490, 300))
            self.ImageLabel.setMovie(self.movie)
            self.movie.start()
            #Change the current window variable
            current_window = 5

        elif current_window == 5:
            #Change Directions to the manual mode delete piece Directions.
            self.DirectionsLabel.setText("If you want to delete a piece, click on the \"Empty square\" button and then on the destination square on the chessboard.\n"
"\n"
"To go back to the game, click on the \"Back to game\" button or close the manual mode window.")
            self.movie = QtGui.QMovie(GUI_PKG_DIR +  "/images/GIFs/Manual_mode_back.gif")
            self.ImageLabel.setMinimumSize(QtCore.QSize(422, 300))
            self.ImageLabel.setMaximumSize(QtCore.QSize(422, 300))
            self.ImageLabel.setMovie(self.movie)
            self.movie.start()
            self.NextPushButton.setEnabled(False)
            #Change the current window variable
            current_window = 6


    def GoToPrev(self):
        global current_window
        if current_window == 2:
            #Change Directions to the highlight Directions.
            self.PrevPushButton.setEnabled(False)
            self.DirectionsLabel.setText("When it\'s your turn, move around the chessboard squares with the joystick.\n"
"\n"
"The square with the focus on it will be highlighted in light red.")
            self.movie = QtGui.QMovie(GUI_PKG_DIR + "/images/GIFs/Highlight.gif")
            self.ImageLabel.setMinimumSize(QtCore.QSize(335, 300))
            self.ImageLabel.setMaximumSize(QtCore.QSize(335, 300))
            self.ImageLabel.setMovie(self.movie)
            self.movie.start()
            #Change the current window variable
            current_window = 1

        elif current_window == 3:
            #Change Directions to the selection Directions.
            self.DirectionsLabel.setText("To move, select the desired piece with the joystick button or directly left-clicking with the mouse.\n"
"\n"
"The selected piece will turn red.")
            self.movie = QtGui.QMovie(GUI_PKG_DIR + "/images/GIFs/Selection_gif.gif")
            self.ImageLabel.setMinimumSize(QtCore.QSize(335, 300))
            self.ImageLabel.setMaximumSize(QtCore.QSize(335, 300))
            self.ImageLabel.setMovie(self.movie)
            self.movie.start()
            #Change the current window variable
            current_window = 2

        elif current_window == 4:
            #Change Directions to the confirm Directions.
            self.DirectionsLabel.setText("Click on the square intended as the selected piece destination.\n"
"The move will be highlightd in yellow.\n"
"\n"
"Click again on the destination square to confirm the move.\n"
"\n"
"If you want to change move, click on another square.")
            self.movie = QtGui.QMovie(GUI_PKG_DIR + "/images/GIFs/Confirm_move.gif")
            self.ImageLabel.setMinimumSize(QtCore.QSize(335, 300))
            self.ImageLabel.setMaximumSize(QtCore.QSize(335, 300))
            self.ImageLabel.setMovie(self.movie)
            self.movie.start()
            #Change the current window variable
            current_window = 3

        elif current_window == 5:
            #Change Directions to the promotion.
            self.DirectionsLabel.setText("Whenever you or the opponent promote a pawn, a window will appear to choose the piece. Select the desired piece for your promotion or the piece chosen by the opponent for his promotion.\n"
"\n"
"Ask the opponent to place the piece chosen by you on the regular chessboard, eventually.")
            self.movie = QtGui.QMovie(GUI_PKG_DIR + "/images/GIFs/Promotion_gif.gif")
            self.ImageLabel.setMinimumSize(QtCore.QSize(335, 300))
            self.ImageLabel.setMaximumSize(QtCore.QSize(335, 300))
            self.ImageLabel.setMovie(self.movie)
            self.movie.start()
            #Change the current window variable
            current_window = 4

        elif current_window == 6:
            #Change Directions to the manual mode insert piece Directions.
            self.DirectionsLabel.setText("If TIAGo is wrong in identifiyng the executed opponent move, you can switch to manual mode to manually change the GUI chessboard.\n"
"\n"
"A window will appear: select the piece that you want to place and then click on the destination square.")
            self.NextPushButton.setEnabled(True)
            self.movie = QtGui.QMovie(GUI_PKG_DIR + "/images/GIFs/Manual_mode_add.gif")
            self.ImageLabel.setMinimumSize(QtCore.QSize(490, 300))
            self.ImageLabel.setMaximumSize(QtCore.QSize(490, 300))
            self.ImageLabel.setMovie(self.movie)
            self.movie.start()
            #Change the current window variable
            current_window = 5

    def CloseWizard(self):
        self.close.emit(1)


###################################################################



###From: https://stackoverflow.com/questions/46706650/determine-qwidget-that-had-last-focus-before-button-press

class inputFocusFilter(QObject):
    focusIn = pyqtSignal(object)

    def eventFilter(self, widget, event):
        if event.type() == QEvent.FocusIn: #and isinstance(widget, QPushButton):
            # emit a `focusIn` signal, with the widget as its argument:
            self.focusIn.emit(widget)
        return super(inputFocusFilter, self).eventFilter(widget, event)

class MyApplication(QApplication):
    def __init__(self, *arg, **kwarg):
        super(MyApplication, self).__init__(*arg, **kwarg)

        self._input_focus_widget = None

        self.event_filter = inputFocusFilter()
        self.event_filter.focusIn.connect(self.setInputFocusWidget)
        self.installEventFilter(self.event_filter)

    def setInputFocusWidget(self, widget):
        if isinstance(widget, QtWidgets.QPushButton):
                self._input_focus_widget = widget

    def inputFocusWidget(self):
        return self._input_focus_widget


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())

