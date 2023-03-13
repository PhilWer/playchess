#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets

from WizardWindow import Ui_WizardWindow

#Initialize global variables
current_window = 1 #Highlight Wizard Window.


class MyWizard(QtWidgets.QWidget, Ui_WizardWindow):
    def __init__(self):
        super(MyWizard, self).__init__()
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
            #self.ImageLabel #CHANGE GIF
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
            #self.ImageLabel #CHANGE GIF

            #Change the current window variable
            current_window = 3

        elif current_window == 3:
            #Change Directions to the promotion Directions.
            self.DirectionsLabel.setText("Whenever you or the opponent promote a pawn, a window will appear to choose the piece. Select the desired piece for your promotion or the piece chosen by the opponent for his promotion.\n"
"\n"
"Ask the opponent to place the piece chosen by you on the regular chessboard, eventually.")
            #self.ImageLabel #CHANGE GIF

            #Change the current window variable
            current_window = 4

        elif current_window == 4:
            #Change Directions to the manual mode insert piece Directions.
            self.DirectionsLabel.setText("If TIAGo is wrong in identifiyng the executed opponent move, you can switch to manual mode to manually change the GUI chessboard.\n"
"\n"
"A window will appear: select the piece that you want to place and then click on the destination square.")
            #self.ImageLabel #CHANGE GIF

            #Change the current window variable
            current_window = 5


        elif current_window == 5:
            #Change Directions to the manual mode delete piece Directions.
            self.DirectionsLabel.setText("If you want to delete a piece, click on the \"Empty square\" button and then on the destination square on the chessboard.\n"
"\n"
"To go back to the game, click on the \"Back to game\" button or close the manual mode window.")
            #self.ImageLabel #CHANGE GIF
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

            #Change the current window variable
            current_window = 1

        elif current_window == 3:
            #Change Directions to the selection Directions.
            self.DirectionsLabel.setText("To move, select the desired piece with the joystick button or directly left-clicking with the mouse.\n"
"\n"
"The selected piece will turn red.")

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

            #Change the current window variable
            current_window = 3

        elif current_window == 5:
            #Change Directions to the promotion.
            self.DirectionsLabel.setText("Whenever you or the opponent promote a pawn, a window will appear to choose the piece. Select the desired piece for your promotion or the piece chosen by the opponent for his promotion.\n"
"\n"
"Ask the opponent to place the piece chosen by you on the regular chessboard, eventually.")
            #Change the current window variable
            current_window = 4

        elif current_window == 6:
            #Change Directions to the manual mode insert piece Directions.
            self.DirectionsLabel.setText("If TIAGo is wrong in identifiyng the executed opponent move, you can switch to manual mode to manually change the GUI chessboard.\n"
"\n"
"A window will appear: select the piece that you want to place and then click on the destination square.")
            #Change the current window variable
            current_window = 5
            



    def CloseWizard(self):
        pass
        

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    wizard = MyWizard()
    wizard.show()
    sys.exit(app.exec_())

