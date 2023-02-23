# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets

PLAYCHESS_PKG_DIR = "/home/pal/tiago_public_ws/src/playchess"
GUI_SCRIPTS_DIR   = PLAYCHESS_PKG_DIR + "/scripts/gui"

class Ui_WhitePromotion(object):
    def setupUi(self, WhitePromotion):
        #Icons of the selected buttons
        self.icon_bishop_selected = QtGui.QIcon()
        self.icon_bishop_selected.addPixmap(QtGui.QPixmap(GUI_SCRIPTS_DIR + "/images/W_Bishop_w_frame.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.icon_knight_selected = QtGui.QIcon()
        self.icon_knight_selected.addPixmap(QtGui.QPixmap(GUI_SCRIPTS_DIR + "/images/W_Knight_w_frame.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.icon_queen_selected = QtGui.QIcon()
        self.icon_queen_selected.addPixmap(QtGui.QPixmap(GUI_SCRIPTS_DIR + "/images/W_Queen_w_frame.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.icon_rook_selected = QtGui.QIcon()
        self.icon_rook_selected.addPixmap(QtGui.QPixmap(GUI_SCRIPTS_DIR + "/images/W_Rook_w_frame.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)

        WhitePromotion.setObjectName("WhitePromotion")
        WhitePromotion.resize(400, 134)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(WhitePromotion)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.QuestionLabel = QtWidgets.QLabel(WhitePromotion)
        self.QuestionLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.QuestionLabel.setObjectName("QuestionLabel")
        self.verticalLayout_2.addWidget(self.QuestionLabel)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.queenPushButton = QtWidgets.QPushButton(WhitePromotion)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.queenPushButton.sizePolicy().hasHeightForWidth())
        self.queenPushButton.setSizePolicy(sizePolicy)
        self.queenPushButton.setText("")
        self.icon_queen = QtGui.QIcon()
        self.icon_queen.addPixmap(QtGui.QPixmap(GUI_SCRIPTS_DIR + "/images/W_Queen_w.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.queenPushButton.setIcon(self.icon_queen_selected)
        self.queenPushButton.setIconSize(QtCore.QSize(50, 50))
        self.queenPushButton.setObjectName("queenPushButton")
        self.queenPushButton.setFocus() #Set the initial focus of the GUI.
        self.horizontalLayout_3.addWidget(self.queenPushButton)
        self.rookPushButton = QtWidgets.QPushButton(WhitePromotion)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rookPushButton.sizePolicy().hasHeightForWidth())
        self.rookPushButton.setSizePolicy(sizePolicy)
        self.rookPushButton.setText("")
        self.icon_rook = QtGui.QIcon()
        self.icon_rook.addPixmap(QtGui.QPixmap(GUI_SCRIPTS_DIR + "/images/W_Rook_w.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.rookPushButton.setIcon(self.icon_rook)
        self.rookPushButton.setIconSize(QtCore.QSize(50, 50))
        self.rookPushButton.setObjectName("rookPushButton")
        self.horizontalLayout_3.addWidget(self.rookPushButton)
        self.knightPushButton = QtWidgets.QPushButton(WhitePromotion)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.knightPushButton.sizePolicy().hasHeightForWidth())
        self.knightPushButton.setSizePolicy(sizePolicy)
        self.knightPushButton.setText("")
        self.icon_knight = QtGui.QIcon()
        self.icon_knight.addPixmap(QtGui.QPixmap(GUI_SCRIPTS_DIR + "/images/W_Knight_w.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.knightPushButton.setIcon(self.icon_knight)
        self.knightPushButton.setIconSize(QtCore.QSize(50, 50))
        self.knightPushButton.setObjectName("knightPushButton")
        self.horizontalLayout_3.addWidget(self.knightPushButton)
        self.bishopPushButton = QtWidgets.QPushButton(WhitePromotion)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bishopPushButton.sizePolicy().hasHeightForWidth())
        self.bishopPushButton.setSizePolicy(sizePolicy)
        self.bishopPushButton.setText("")
        self.icon_bishop = QtGui.QIcon()
        self.icon_bishop.addPixmap(QtGui.QPixmap(GUI_SCRIPTS_DIR + "/images/W_Bishop_w.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bishopPushButton.setIcon(self.icon_bishop)
        self.bishopPushButton.setIconSize(QtCore.QSize(50, 50))
        self.bishopPushButton.setObjectName("bishopPushButton")
        self.horizontalLayout_3.addWidget(self.bishopPushButton)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)

        self.retranslateUi(WhitePromotion)
        QtCore.QMetaObject.connectSlotsByName(WhitePromotion)

    def retranslateUi(self, WhitePromotion):
        _translate = QtCore.QCoreApplication.translate
        WhitePromotion.setWindowTitle(_translate("WhitePromotion", "Form"))
        self.QuestionLabel.setText(_translate("WhitePromotion", "Which is the promoted piece?"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    WhitePromotion = QtWidgets.QWidget()
    ui = Ui_WhitePromotion()
    ui.setupUi(WhitePromotion)
    WhitePromotion.show()
    sys.exit(app.exec_())

