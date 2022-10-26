# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets

GUI_PKG_DIR = '/home/luca/tiago_public_ws/src/chess_gui'

class Ui_SearchExecution(object):
    def setupUi(self, SearchExecution):
        SearchExecution.setObjectName("SearchExecution")
        SearchExecution.resize(400, 198)
        self.verticalLayout = QtWidgets.QVBoxLayout(SearchExecution)
        self.verticalLayout.setObjectName("verticalLayout")
        self.SearchLabel = QtWidgets.QLabel(SearchExecution)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.SearchLabel.sizePolicy().hasHeightForWidth())
        self.SearchLabel.setSizePolicy(sizePolicy)
        self.SearchLabel.setMinimumSize(QtCore.QSize(0, 50))
        self.SearchLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.SearchLabel.setObjectName("SearchLabel")
        self.verticalLayout.addWidget(self.SearchLabel)
        self.ProgressGifLabel = QtWidgets.QLabel(SearchExecution)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ProgressGifLabel.sizePolicy().hasHeightForWidth())
        self.ProgressGifLabel.setSizePolicy(sizePolicy)
        self.ProgressGifLabel.setMaximumSize(QtCore.QSize(50, 50))
        self.ProgressGifLabel.setText("")
        self.movie = QtGui.QMovie(GUI_PKG_DIR + "/images/ElaborationGif.gif")
        self.ProgressGifLabel.setMovie(self.movie)
        self.movie.start()
        #self.ProgressGifLabel.setPixmap(QtGui.QPixmap("images/Pulse-1s-211px.gif"))
        self.ProgressGifLabel.setScaledContents(True)
        self.ProgressGifLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.ProgressGifLabel.setObjectName("ProgressGifLabel")
        self.verticalLayout.addWidget(self.ProgressGifLabel, 0, QtCore.Qt.AlignHCenter)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.OperationCompletedLabel = QtWidgets.QLabel(SearchExecution)
        self.OperationCompletedLabel.setEnabled(False)
        self.OperationCompletedLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.OperationCompletedLabel.setObjectName("OperationCompletedLabel")
        self.horizontalLayout.addWidget(self.OperationCompletedLabel)
        self.CheckResultsPushButton = QtWidgets.QPushButton(SearchExecution)
        self.CheckResultsPushButton.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.CheckResultsPushButton.sizePolicy().hasHeightForWidth())
        self.CheckResultsPushButton.setSizePolicy(sizePolicy)
        self.CheckResultsPushButton.setMaximumSize(QtCore.QSize(16777215, 50))
        self.CheckResultsPushButton.setObjectName("CheckResultsPushButton")
        self.horizontalLayout.addWidget(self.CheckResultsPushButton)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(SearchExecution)
        QtCore.QMetaObject.connectSlotsByName(SearchExecution)

    def retranslateUi(self, SearchExecution):
        _translate = QtCore.QCoreApplication.translate
        SearchExecution.setWindowTitle(_translate("SearchExecution", "Form"))
        self.SearchLabel.setText(_translate("SearchExecution", " ARUCO markers search in progress..."))
        self.OperationCompletedLabel.setText(_translate("SearchExecution", "Operation Completed!"))
        self.CheckResultsPushButton.setText(_translate("SearchExecution", "Check Search Results"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    SearchExecution = QtWidgets.QWidget()
    ui = Ui_SearchExecution()
    ui.setupUi(SearchExecution)
    SearchExecution.show()
    sys.exit(app.exec_())

