# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_ColorAskingWindow(object):
    def setupUi(self, ColorAskingWindow):
        ColorAskingWindow.setObjectName("ColorAskingWindow")
        ColorAskingWindow.resize(615, 478)
        self.verticalLayout = QtWidgets.QVBoxLayout(ColorAskingWindow) #(ColorAskingWindow)
        self.verticalLayout.setObjectName("verticalLayout")
        self.ColorAskingLabel = QtWidgets.QLabel(ColorAskingWindow)
        self.ColorAskingLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.ColorAskingLabel.setObjectName("ColorAskingLabel")
        self.verticalLayout.addWidget(self.ColorAskingLabel)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.WhiteBtn = QtWidgets.QPushButton(ColorAskingWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.WhiteBtn.sizePolicy().hasHeightForWidth())
        self.WhiteBtn.setSizePolicy(sizePolicy)
        self.WhiteBtn.setMaximumSize(QtCore.QSize(1000000, 100))
        self.WhiteBtn.setObjectName("WhiteBtn")
        self.WhiteBtn.setFocus() #Set the initial focus of the GUI.
        self.WhiteBtn.setStyleSheet("background-color: rgb(176, 202, 220);")
        self.horizontalLayout.addWidget(self.WhiteBtn)
        self.BlackBtn = QtWidgets.QPushButton(ColorAskingWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.BlackBtn.sizePolicy().hasHeightForWidth())
        self.BlackBtn.setSizePolicy(sizePolicy)
        self.BlackBtn.setMaximumSize(QtCore.QSize(1000000, 100))
        self.BlackBtn.setObjectName("BlackBtn")
        self.horizontalLayout.addWidget(self.BlackBtn)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(ColorAskingWindow)
        QtCore.QMetaObject.connectSlotsByName(ColorAskingWindow)

    def retranslateUi(self, ColorAskingWindow):
        _translate = QtCore.QCoreApplication.translate
        ColorAskingWindow.setWindowTitle(_translate("ColorAskingWindow", "Form"))
        self.ColorAskingLabel.setText(_translate("ColorAskingWindow", "Which color is TIAGo playing with?"))
        self.WhiteBtn.setText(_translate("ColorAskingWindow", "White"))
        self.BlackBtn.setText(_translate("ColorAskingWindow", "Black"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ColorAskingWindow = QtWidgets.QWidget()
    ui = Ui_ColorAskingWindow()
    ui.setupUi(ColorAskingWindow, app_focus)
    ColorAskingWindow.show()
    sys.exit(app.exec_())

