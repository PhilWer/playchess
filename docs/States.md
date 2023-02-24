# Playchess state machine [WORK IN PROGRESS]

The following is a list of the possible states of the `playchess` state machine with a brief explanation of the corresponding behavior of the application:
1. TODO
2. TODO
...
14. **Sent upon.** End of the user turn.
    **Trigger.** TIAGo completes the 'click clock' operation.
    **Behavior.** The GUI status is updated to 'Waiting'.
15. **Sent upon.** End of the opponent turn.
    **Trigger.** Clock button pressed by the opponent.
    **Behavior.** The pipeline for the opponent move detection is run and the GUI updated accordingly. 
16. TODO
40. **Sent upon.** End of ArUCo search and planning scene setup.
    **Trigger.** End of the planning scene setup, issued by the `aruco_detection.py` node.
    **Behavior.** The GUI button to move to the ArUCo confirmation window is enabled.