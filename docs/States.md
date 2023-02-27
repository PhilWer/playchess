# Playchess state machine [WORK IN PROGRESS]

The following is a list of the possible states of the `playchess` state machine with a brief explanation of the corresponding behavior of the application:
1. TIAGo initialization.
    **Trigger.** First joystick button press when the GUI is launched.
    **Behavior.** The robot lifts the torso and lowers the head to ensure an 'optimal' view of the chessboard.
2. TODO.
3. TODO.
4. Chessboard segmentation.
    **Trigger.** Button press in the GUI.
    **Behavior.** The 'processing' GIF is shown until the completion of the segmentation, then the button to move to the following window is activated.
5. Confirm chessboard segmentation.
    **Trigger.** Button press in the GUI.
    **Behavior.** The outcome of the segmentation is shown, the user can accept or reject it.
6. TODO.
7. ArUCo markers search.
    **Trigger.** Button press in the GUI.
    **Behavior.**The 'processing' GIF is shown until the completion of the search, then the button to move to the following window is activated.
8. TODO.
9. TIAGo's arm preparation.
    **Trigger.** 'Start Game' button pressed in the GUI.
    **Behavior.** TIAGo moves the arm to a pre-defined configuration (right side of the chessboard).
13. Move execution (by TIAGo).
    **Trigger.** The move is selected and confirmed in the GUI.
    **Behavior.** The robot's head is lifted, the robot arm executes the movements to complete the given move.
14. Opponent's turn.
    **Trigger.** TIAGo completes the 'click clock' operation.
    **Behavior.** The GUI status is updated to 'Waiting'.
15. Opponent's move recognition.
    **Trigger.** Clock button pressed by the opponent.
    **Behavior.** The pipeline for the opponent move detection is run and the GUI updated accordingly. 
16. TODO.
40. **Sent upon.** End of ArUCo search and planning scene setup.
    **Trigger.** End of the planning scene setup, issued by the `aruco_detection.py` node.
    **Behavior.** The GUI button to move to the ArUCo confirmation window is enabled.