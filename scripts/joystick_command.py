#!/usr/bin/env python3
#This script defines a subscriber node that reads different messages and governates the different phases of the play_chess action.

#Python libs

#ROS libs
import rospy
import pygame
import time

#ROS messages
from sensor_msgs.msg import PointCloud2, CompressedImage, Image
from std_msgs.msg import Bool, Int16
from geometry_msgs.msg import Point

#Callbacks definition
def Callback_State(data):
    state.data = data.data
    print('Current state: ' + str(state.data)) 

#Starts a new node
rospy.init_node('joystick_commander', anonymous = True)

#Variables initialization
#Joystick offset experimentally defined
offset_h = -0.196 #TODO
offset_v = -0.175 #TODO

#Publishers initialization
#The number in the topic name refers to the current state of the elaboration
button_initialization_publisher = rospy.Publisher('/button_zero', Bool, queue_size = 10)
button_proceed_publisher = rospy.Publisher('/button_one_point_one', Bool, queue_size = 10)
button_color_publisher = rospy.Publisher('/button_two', Bool, queue_size = 10)
button_start_segmentation_publisher = rospy.Publisher('/button_three', Bool, queue_size = 10)
button_segmentation_publisher = rospy.Publisher('/button_four', Bool, queue_size = 10)
button_segmentation_confirm_publisher = rospy.Publisher('/button_five', Bool, queue_size = 10)
button_start_search_publisher = rospy.Publisher('/button_six', Bool, queue_size = 10)
button_search_publisher = rospy.Publisher('/button_seven', Bool, queue_size = 10)
button_search_confirm_publisher = rospy.Publisher('/button_eight', Bool, queue_size = 10)
button_tiago_preparation_publisher = rospy.Publisher('/button_ten', Bool, queue_size = 10)
button_chessboard_publisher = rospy.Publisher('/button_eleven', Bool, queue_size = 10)
button_promotion_publisher = rospy.Publisher('/button_promotion', Bool, queue_size = 10)
button_end_segmentation_publisher = rospy.Publisher('/button_end_segmentation', Bool, queue_size = 10)
button_end_search_publisher = rospy.Publisher('/button_end_search', Bool, queue_size = 10)

cursor_step_proceed_publisher = rospy.Publisher('/cursor_step_one_point_one', Point, queue_size = 10)
cursor_step_color_publisher = rospy.Publisher('/cursor_step_two', Point, queue_size = 10)
cursor_step_start_segmentation_publisher = rospy.Publisher('/cursor_step_three', Point, queue_size = 10)
cursor_step_segmentation_confirm_publisher = rospy.Publisher('/cursor_step_five', Point, queue_size = 10)
cursor_step_search_confirm_publisher = rospy.Publisher('/cursor_step_eight', Point, queue_size = 10)
cursor_step_chessboard_publisher = rospy.Publisher('/cursor_step_eleven', Point, queue_size = 10)
cursor_step_promotion_publisher = rospy.Publisher('/cursor_step_promotion', Point, queue_size = 10)

segmented_chessboard_publisher = rospy.Publisher('/segmented_chessboard', CompressedImage, queue_size = 1)
found_markers_publisher = rospy.Publisher('/found_markers', CompressedImage, queue_size = 1)
opponent_chess_move_publisher = rospy.Publisher('/chess_move', CompressedImage, queue_size = 1)
init_state_publisher = rospy.Publisher('/init_state', Bool, queue_size = 10)
state_publisher = rospy.Publisher('/state', Int16, queue_size = 10)

#Subscribers initialization
rospy.Subscriber("/state", Int16, Callback_State)

rate = rospy.Rate(2)

#Messages initialization
cursor_step = Point()
cursor_step.x = 0
cursor_step.y = 0
cursor_step.z = 0 #FORSE NON SERVE

#button_state = joystick.get_button(1) #The button is not pressed in initialization

init_state = Bool()
init_state.data = False

state = Int16()
state.data = 0

button_zero = Bool()
button_zero.data = False
        
		
if __name__ == '__main__':
    try:
        pygame.init()
        pygame.joystick.init()
        JoystickCnt = pygame.joystick.get_count()

        if JoystickCnt != 0:
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
            init_state.data = True
            init_state_publisher.publish(init_state)

        else:
            print('No Joystick found.')
            init_state.data = False
            init_state_publisher.publish(init_state)

    except Exception as e:#rospy.ROSInterruptException: #KeyboardInterrupt:
        print('Error: Pygame initialization failed | ' + str(e))
        init_state.data = False
        init_state_publisher.publish(init_state)

        #rospy.logwarn('Shutting down the master node')
        #pass

button_state = joystick.get_button(0) #The button is not pressed in initialization
'''
n = joystick.get_numbuttons()
old_states = [joystick.get_button(b) for b in range(n)]
while not rospy.is_shutdown():
    states = [joystick.get_button(b) for b in range(n)]
    diff = [b for b in range(n) if old_states[b] != states[b]]
    print(states)
    pygame.event.pump()
'''

while not rospy.is_shutdown():
    pygame.event.pump()
    if state.data == 0: #Initialization state.
        events = pygame.event.get()
        if joystick.get_button(0) != button_state:
            button_state = joystick.get_button(0)
            if button_state == 1: #If the button is pressed down, the operational state is changed to 1 to start TIAGo preparation.
                state_publisher.publish(1)
                rospy.loginfo('Button pressed in initialization to start TIAGo preparation (looking down and looking around).')
                time.sleep(1)

    elif state.data == 1: #TIAGo initialization state.
        events = pygame.event.get()
        if joystick.get_button(0) != button_state:
            button_state = joystick.get_button(0)
            if button_state == 1: #If the button is pressed down, the color asking window is enabled.
                button_zero.data = True
                button_initialization_publisher.publish(button_zero)
                rospy.loginfo('Button pressed during TIAGo preparation. Wait till the end to proceed.')

    elif state.data == 20: #State of initialization in which the GUI pushbutton is enabled to go on (after TIAGO's preparation ended).
        try:
            #Verify the event on the joystick
            events = pygame.event.get()
            if - (joystick.get_axis(1) - offset_v) > 0: #If the joystick is moved to the right, one step of the GUI's cursor is performed in the right direction.
                if abs(joystick.get_axis(1) - offset_v) > 0.300: #Dead band control
                    cursor_step.x = 1 #* Gain_x_p # Filling of the cursor_step message.
                else:
                    cursor_step.x = 0
            elif - (joystick.get_axis(1) - offset_v) < 0: #If the joystick is moved to the left, one step of the GUI's cursor is performed in the left direction.
                if abs(joystick.get_axis(1) - offset_v) > 0.300: #Dead band control
                    cursor_step.x = 2 #* Gain_x_n
                else:
                    cursor_step.x = 0
            
            cursor_step.y = 0
            cursor_step.z = 0

            cursor_step_proceed_publisher.publish(cursor_step)

            if joystick.get_button(0) != button_state:
                button_state = joystick.get_button(0)
                if button_state == 1: #If the button is pressed down, the color asking window is enabled.
                    button_zero.data = True
                    button_proceed_publisher.publish(button_zero)
                    rospy.loginfo('Button pressed to proceed to color choice.')

            rate.sleep()

        except (KeyboardInterrupt, SystemExit):
            print('\n# Ctrl-C detected\n')   

    elif state.data == 2: #State of TIAGo's color question.
        try:
            #Verify the event on the joystick
            events = pygame.event.get()
            if - (joystick.get_axis(1) - offset_v) > 0: #If the joystick is moved to the right, one step of the GUI's cursor is performed in the right direction.
                if abs(joystick.get_axis(1) - offset_v) > 0.300: #Dead band control
                    cursor_step.x = 1 #* Gain_x_p # Filling of the cursor_step message.
                else:
                    cursor_step.x = 0
            elif - (joystick.get_axis(1) - offset_v) < 0: #If the joystick is moved to the left, one step of the GUI's cursor is performed in the left direction.
                if abs(joystick.get_axis(1) - offset_v) > 0.300: #Dead band control
                    cursor_step.x = 2 #* Gain_x_n
                else:
                    cursor_step.x = 0
            
            cursor_step.y = 0
            cursor_step.z = 0

            cursor_step_color_publisher.publish(cursor_step)

            if joystick.get_button(0) != button_state: #If the button is pressed, send a message to the GUI that will select the choice.
                button_state = joystick.get_button(0)
                if button_state == 1: #If the button is pressed down, not if the button is coming back up
                    button_zero.data = True
                    button_color_publisher.publish(button_zero)
                    rospy.loginfo('Button pressed to start segmentation.')

            rate.sleep()

        except (KeyboardInterrupt, SystemExit):
            print('\n# Ctrl-C detected\n')

    elif state.data == 3: #State of waiting to start the chessboard segmentation.
        try:
            #Verify the event on the joystick
            events = pygame.event.get()
            if - (joystick.get_axis(1) - offset_v) > 0: #If the joystick is moved to the right, one step of the GUI's cursor is performed in the right direction.
                if abs(joystick.get_axis(1) - offset_v) > 0.300: #Dead band control
                    cursor_step.x = 1 #* Gain_x_p # Filling of the cursor_step message.
                else:
                    cursor_step.x = 0
            elif - (joystick.get_axis(1) - offset_v) < 0: #If the joystick is moved to the left, one step of the GUI's cursor is performed in the left direction.
                if abs(joystick.get_axis(1) - offset_v) > 0.300: #Dead band control
                    cursor_step.x = 2 #* Gain_x_n
                else:
                    cursor_step.x = 0

            if joystick.get_axis(0) - offset_h > 0: #If the joystick is moved up, one step up of the GUI's cursor is performed.
                if abs(joystick.get_axis(0) - offset_h) > 0.300: #Dead band control
                    cursor_step.y = 1 #* Gain_y_p
                else:
                    cursor_step.y = 0
            elif joystick.get_axis(0) - offset_h < 0: #If the joystick is moved down, one step down of the GUI's cursor is performed.
                if abs(joystick.get_axis(0) - offset_h) > 0.300: #Dead band control
                    cursor_step.y = 2 #* Gain_y_n
                else:
                    cursor_step.y = 0
            cursor_step.z = 0

            cursor_step_start_segmentation_publisher.publish(cursor_step)

            if joystick.get_button(0) != button_state: #If the button is pressed, send a message to the GUI that will select the choice.
                button_state = joystick.get_button(0)
                if button_state == 1: #If the button is pressed down, not if the button is coming back up
                    button_zero.data = True
                    button_start_segmentation_publisher.publish(button_zero)
                    rospy.loginfo('Button pressed to start segmentation')

            rate.sleep()

        except (KeyboardInterrupt, SystemExit):
            print('\n# Ctrl-C detected\n')

    elif state == 4: #State of chessboard segmentation.
        events = pygame.event.get()
        #print(joystick.get_button(0))
        if joystick.get_button(0) != button_state:
            button_state = joystick.get_button(0)
            if button_state == 1: #If the button is pressed down, the color asking window is enabled.
                button_zero.data = True
                button_segmentation_publisher.publish(button_zero)
                rospy.loginfo('Button pressed during segmentation. Wait for it to end before proceeding.')

    elif state.data == 30: #State of segmentation confirmation in which the GUI pushbutton is enabled to go on (after chessboard segmentation ended).
        events = pygame.event.get()
        if joystick.get_button(0) != button_state:
            button_state = joystick.get_button(0)
            if button_state == 1: #If the button is pressed down, the color asking window is enabled.
                button_zero.data = True
                button_end_segmentation_publisher.publish(button_zero)
                rospy.loginfo('Button pressed to confirm segmentation.') 

    elif state.data == 5: #State of chessboard segmentation confirmation.
        try:
            #Verify the event on the joystick
            events = pygame.event.get()
            if - (joystick.get_axis(1) - offset_v) > 0: #If the joystick is moved to the right, one step of the GUI's cursor is performed in the right direction.
                if abs(joystick.get_axis(1) - offset_v) > 0.300: #Dead band control
                    cursor_step.x = 1 #* Gain_x_p # Filling of the cursor_step message.
                else:
                    cursor_step.x = 0
            elif - (joystick.get_axis(1) - offset_v) < 0: #If the joystick is moved to the left, one step of the GUI's cursor is performed in the left direction.
                if abs(joystick.get_axis(1) - offset_v) > 0.300: #Dead band control
                    cursor_step.x = 2 #* Gain_x_n
                else:
                    cursor_step.x = 0
            
            cursor_step.y = 0
            cursor_step.z = 0

            cursor_step_segmentation_confirm_publisher.publish(cursor_step)

            if joystick.get_button(0) != button_state: #If the button is pressed, send a message to the GUI that will select the choice.
                button_state = joystick.get_button(0)
                if button_state == 1: #If the button is pressed down, not if the button is coming back up
                    button_zero.data = True
                    button_segmentation_confirm_publisher.publish(button_zero)
                    rospy.loginfo('Button pressed to confirm segmentation')

            rate.sleep()

        except (KeyboardInterrupt, SystemExit):
            print('\n# Ctrl-C detected\n')

    elif state.data == 6: #State of waiting to search ARUCO markers.
        events = pygame.event.get()
        #print(joystick.get_button(0))
        if joystick.get_button(0) != button_state:
            button_state = joystick.get_button(0)
            if button_state == 1: #If the button is pressed down, the color asking window is enabled.
                button_zero.data = True
                button_start_search_publisher.publish(button_zero)
                rospy.loginfo('Button pressed to start searching for ARUCO markers.')

    elif state.data == 7: #State of ARUCO markers search.
        events = pygame.event.get()
        #print(joystick.get_button(0))
        if joystick.get_button(0) != button_state:
            button_state = joystick.get_button(0)
            if button_state == 1: #If the button is pressed down, the color asking window is enabled.
                button_zero.data = True
                button_search_publisher.publish(button_zero)
                rospy.loginfo('Button pressed during markers search. Wait for it to end before proceeding to prepare TIAGo.')

    elif state.data == 40: #State of markers localization confirmation in which the GUI pushbutton is enabled to go on (after the search ended).
        events = pygame.event.get()
        if joystick.get_button(0) != button_state:
            button_state = joystick.get_button(0)
            if button_state == 1: #If the button is pressed down, the color asking window is enabled.
                button_zero.data = True
                button_end_search_publisher.publish(button_zero)
                rospy.loginfo('Button pressed to confirm markers search.')

    elif state.data == 8: #State of clock and box positioning confirmation.
        try:
            #Verify the event on the joystick
            events = pygame.event.get()
            if - (joystick.get_axis(1) - offset_v) > 0: #If the joystick is moved to the right, one step of the GUI's cursor is performed in the right direction.
                if abs(joystick.get_axis(1) - offset_v) > 0.300: #Dead band control
                    cursor_step.x = 1 #* Gain_x_p # Filling of the cursor_step message.
                else:
                    cursor_step.x = 0
            elif - (joystick.get_axis(1) - offset_v) < 0: #If the joystick is moved to the left, one step of the GUI's cursor is performed in the left direction.
                if abs(joystick.get_axis(1) - offset_v) > 0.300: #Dead band control
                    cursor_step.x = 2 #* Gain_x_n
                else:
                    cursor_step.x = 0
            
            cursor_step.y = 0
            cursor_step.z = 0

            cursor_step_search_confirm_publisher.publish(cursor_step)

            if joystick.get_button(0) != button_state: #If the button is pressed, send a message to the GUI that will select the choice.
                button_state = joystick.get_button(0)
                if button_state == 1: #If the button is pressed down, not if the button is coming back up
                    button_zero.data = True
                    button_search_confirm_publisher.publish(button_zero)
                    rospy.loginfo('Button pressed to confirm ARUCO markers search.')

            rate.sleep()

        except (KeyboardInterrupt, SystemExit):
            print('\nCtrl-C detected\n')

    elif state.data == 10: #State of wait to start the game.
        events = pygame.event.get()
        #print(joystick.get_button(0))
        if joystick.get_button(0) != button_state:
            button_state = joystick.get_button(0)
            if button_state == 1: #If the button is pressed down, the color asking window is enabled.
                button_zero.data = True
                button_tiago_preparation_publisher.publish(button_zero)
                rospy.loginfo('Button pressed after TIAGO preparation to start the game.')

    elif state.data == 11: #State of waiting the move to play.
        #In this state, the joystick has to control the movement of the 'cursor' on the GUI interface chessboard.
        #To select the desired piece, the button has to be pressed once.
        #To de-select the piece, push the button again on the wrongly selected piece.
        #If a pawn is promoted, a window will appear to choose the desired piece to promote the pawn to. The joystick will move the cursor up and down.

        try:
            #Verify the event on the joystick
            events = pygame.event.get()

            #Joystick input calculation.
            #Gain: calibration.
            #Offset: joystick's dead band while nobody is touching it.

            if - (joystick.get_axis(1) - offset_v) > 0: #If the joystick is moved to the right, one step of the GUI's cursor is performed in the right direction.
                #x_sens = - (joystick.get_axis(1) - offset_v)###
                #print('-----')
                #print('x_sens: '+str(x_sens))###
                if abs(joystick.get_axis(1) - offset_v) > 0.300: #Dead band control
                    cursor_step.x = 1 #* Gain_x_p # Filling of the cursor_step message.
                else:
                    cursor_step.x = 0
            elif - (joystick.get_axis(1) - offset_v) < 0: #If the joystick is moved to the left, one step of the GUI's cursor is performed in the left direction.
                #x_sens = - (joystick.get_axis(1) - offset_v)###
                #print('-----')
                #print('x_sens: '+str(x_sens))###
                if abs(joystick.get_axis(1) - offset_v) > 0.300: #Dead band control
                    cursor_step.x = 2 #* Gain_x_n
                else:
                    cursor_step.x = 0
            
            if joystick.get_axis(0) - offset_h > 0: #If the joystick is moved up, one step up of the GUI's cursor is performed.
                #y_sens = joystick.get_axis(0) - offset_h ###
                #print('y_sens: '+str(y_sens))
                #print('-----')
                if abs(joystick.get_axis(0) - offset_h) > 0.300: #Dead band control
                    cursor_step.y = 1 #* Gain_y_p
                else:
                    cursor_step.y = 0
            elif joystick.get_axis(0) - offset_h < 0: #If the joystick is moved down, one step down of the GUI's cursor is performed.
                #y_sens = joystick.get_axis(0) - offset_h ###
                #print('y_sens: '+str(y_sens))
                #print('-----')
                if abs(joystick.get_axis(0) - offset_h) > 0.300: #Dead band control
                    cursor_step.y = 2 #* Gain_y_n
                else:
                    cursor_step.y = 0
            cursor_step.z = 0
            '''
            #Control on the case in which both the x and the y direction steps are identified. MAYBE NOT NEEDED.
            if cursor_step.x != 0 and cursor_step.y != 0: #If a movement in both the x and the y directions is detected, verify if one is predominant over the other
                if abs(y_sens - x_sens) > 0.2: #If the difference between the two sensed signal is over a certain threshold, bring back to zero the cursor step in the direction where less force on the joystick is sensed.
                    if y_sens > x_sens:
                        cursor_step.x = 0
                    else:
                        cursor_step.y = 0
            '''
            cursor_step_chessboard_publisher.publish(cursor_step)
            if joystick.get_button(0) != button_state: #If the button is pressed, send a message to the GUI that will select the choice.
                button_state = joystick.get_button(0)
                if button_state == 1: #If the button is pressed down, not if the button is coming back up
                    button_zero.data = True
                    button_chessboard_publisher.publish(button_zero)
                    rospy.loginfo('Button pressed to choose the move to perform')
                    #time.sleep(1)

            rate.sleep()

        except (KeyboardInterrupt, SystemExit):
            print('\nCtrl-C detected\n')

    elif state.data == 12: #State of asking the promoted piece by the player.
        #In this state, the joystick has to control the movement of the 'cursor' on the GUI interface chessboard only in the horzontal direction (the promotion piece option will be displayed on one row)
        #To select the desired piece, the button has to be pressed once.

        try:
            #Verify the event on the joystick
            events = pygame.event.get()

            #Joystick input calculation.
            #Gain: calibration.
            #Offset: joystick's dead band while nobody is touching it.
            #Normalization between [-1 1].
            if - (joystick.get_axis(1) - offset_v) > 0: #If the joystick is moved to the left, one step of the GUI's cursor is performed in the left direction.
                if abs(joystick.get_axis(1) - offset_v) > 0.300: #Dead band control
                    cursor_step.x = 1 #* Gain_x_p # Filling of the cursor_step message.
                else:
                    cursor_step.x = 0
            elif - (joystick.get_axis(1) - offset_v) < 0: #If the joystick is moved to the right, one step of the GUI's cursor is performed in the right direction.
                if abs(joystick.get_axis(1) - offset_v) > 0.300: #Dead band control
                    cursor_step.x = 2 #* Gain_x_n
                else:
                    cursor_step.x = 0
            cursor_step.y = 0
            cursor_step.z = 0
            cursor_step_promotion_publisher.publish(cursor_step)

            if joystick.get_button(0) != button_state:
                button_state = joystick.get_button(0)
                if button_state == 1: #If the button is pressed down, not if the button is coming back up
                    button_zero.data = True
                    button_promotion_publisher.publish(button_zero)
                    rospy.loginfo('Button pressed to choose the promoted piece by the player.')
                    #time.sleep(1)

        except (KeyboardInterrupt, SystemExit):
            print('\nCtrl-C detected\n')

    elif state.data == 16: #State of asking the promoted piece by the opponent.
        #In this state, the joystick has to control the movement of the 'cursor' on the GUI interface chessboard only in the horzontal direction (the promotion piece option will be displayed on one row)
        #To select the desired piece, the button has to be pressed once.

        try:
            #Verify the event on the joystick
            events = pygame.event.get()

            #Joystick input calculation.
            #Gain: calibration.
            #Offset: joystick's dead band while nobody is touching it.
            #Normalization between [-1 1].
            if - (joystick.get_axis(1) - offset_v) > 0: #If the joystick is moved to the left, one step of the GUI's cursor is performed in the left direction.
                if abs(joystick.get_axis(1) - offset_v) > 0.300: #Dead band control
                    cursor_step.x = 1 #* Gain_x_p # Filling of the cursor_step message.
                else:
                    cursor_step.x = 0
            elif - (joystick.get_axis(1) - offset_v) < 0: #If the joystick is moved to the right, one step of the GUI's cursor is performed in the right direction.
                if abs(joystick.get_axis(1) - offset_v) > 0.300: #Dead band control
                    cursor_step.x = 2 #* Gain_x_n
                else:
                    cursor_step.x = 0
            cursor_step.y = 0
            cursor_step.z = 0
            cursor_step_promotion_publisher.publish(cursor_step)

            if joystick.get_button(0) != button_state:
                button_state = joystick.get_button(0)
                if button_state == 1: #If the button is pressed down, not if the button is coming back up
                    button_zero.data = True
                    button_promotion_publisher.publish(button_zero)
                    rospy.loginfo('Button pressed to choose the promoted piece by the opponent.')
                    #time.sleep(1)

        except (KeyboardInterrupt, SystemExit):
            print('\nCtrl-C detected\n')

    