# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'WizardWindow.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

PLAYCHESS_PKG_DIR = "/home/pal/tiago_public_ws/src/playchess"
GUI_SCRIPTS_DIR   = PLAYCHESS_PKG_DIR + "/scripts/gui"

class Ui_WizardWindow(object):
    def setupUi(self, WizardWindow):
        WizardWindow.setObjectName("WizardWindow")
        WizardWindow.resize(666, 398)
        self.verticalLayout = QtWidgets.QVBoxLayout(WizardWindow)
        self.verticalLayout.setObjectName("verticalLayout")
        self.InstructionsLabel = QtWidgets.QLabel(WizardWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.InstructionsLabel.sizePolicy().hasHeightForWidth())
        self.InstructionsLabel.setSizePolicy(sizePolicy)
        self.InstructionsLabel.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.InstructionsLabel.setFont(font)
        self.InstructionsLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.InstructionsLabel.setObjectName("InstructionsLabel")
        self.verticalLayout.addWidget(self.InstructionsLabel, 0, QtCore.Qt.AlignHCenter)
        self.line = QtWidgets.QFrame(WizardWindow)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.DirectionsLabel = QtWidgets.QLabel(WizardWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.DirectionsLabel.sizePolicy().hasHeightForWidth())
        self.DirectionsLabel.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.DirectionsLabel.setFont(font)
        self.DirectionsLabel.setMinimumSize(QtCore.QSize(150, 250))
        self.DirectionsLabel.setMaximumSize(QtCore.QSize(250, 250))
        self.DirectionsLabel.setScaledContents(False)
        self.DirectionsLabel.setAlignment(QtCore.Qt.AlignJustify|QtCore.Qt.AlignVCenter)
        self.DirectionsLabel.setWordWrap(True)
        self.DirectionsLabel.setObjectName("DirectionsLabel")
        self.horizontalLayout.addWidget(self.DirectionsLabel, 0, QtCore.Qt.AlignVCenter)
        self.ImageLabel = QtWidgets.QLabel(WizardWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ImageLabel.sizePolicy().hasHeightForWidth())
        self.ImageLabel.setSizePolicy(sizePolicy)
        self.ImageLabel.setMinimumSize(QtCore.QSize(335, 300))
        self.ImageLabel.setMaximumSize(QtCore.QSize(335, 300))
        self.ImageLabel.setText("")
        self.movie = QtGui.QMovie(GUI_SCRIPTS_DIR + "/images/GIFs/Highlight.gif")
        self.ImageLabel.setMovie(self.movie)
        self.movie.start()
        self.ImageLabel.setScaledContents(True)
        self.ImageLabel.setObjectName("ImageLabel")
        self.horizontalLayout.addWidget(self.ImageLabel)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.ClosePushButton = QtWidgets.QPushButton(WizardWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ClosePushButton.sizePolicy().hasHeightForWidth())
        self.ClosePushButton.setSizePolicy(sizePolicy)
        self.ClosePushButton.setMaximumSize(QtCore.QSize(80, 16777215))
        self.ClosePushButton.setObjectName("ClosePushButton")
        self.horizontalLayout_2.addWidget(self.ClosePushButton)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.PrevPushButton = QtWidgets.QPushButton(WizardWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PrevPushButton.sizePolicy().hasHeightForWidth())
        self.PrevPushButton.setSizePolicy(sizePolicy)
        self.PrevPushButton.setMaximumSize(QtCore.QSize(80, 16777215))
        self.PrevPushButton.setObjectName("PrevPushButton")
        self.PrevPushButton.setEnabled(False)
        self.horizontalLayout_2.addWidget(self.PrevPushButton)
        self.NextPushButton = QtWidgets.QPushButton(WizardWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.NextPushButton.sizePolicy().hasHeightForWidth())
        self.NextPushButton.setSizePolicy(sizePolicy)
        self.NextPushButton.setMaximumSize(QtCore.QSize(80, 16777215))
        self.NextPushButton.setObjectName("NextPushButton")
        self.horizontalLayout_2.addWidget(self.NextPushButton)
        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.retranslateUi(WizardWindow)
        QtCore.QMetaObject.connectSlotsByName(WizardWindow)

    def retranslateUi(self, WizardWindow):
        _translate = QtCore.QCoreApplication.translate
        WizardWindow.setWindowTitle(_translate("WizardWindow", "Form"))
        self.InstructionsLabel.setText(_translate("WizardWindow", "Instructions"))
        self.DirectionsLabel.setText(_translate("WizardWindow", "When it\'s your turn, move around the chessboard squares with the joystick.\n"
"\n"
"The square with the focus on it will be highlighted in light red."))
        self.ClosePushButton.setText(_translate("WizardWindow", "Close"))
        self.PrevPushButton.setText(_translate("SelectWizardWindow", "< Prev"))
        self.NextPushButton.setText(_translate("WizardWindow", "Next >"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    WizardWindow = QtWidgets.QWidget()
    ui = Ui_WizardWindow()
    ui.setupUi(WizardWindow)
    WizardWindow.show()
    sys.exit(app.exec_())

