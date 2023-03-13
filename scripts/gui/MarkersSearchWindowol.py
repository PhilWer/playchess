# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_SearchExecution(object):
    def setupUi(self, SearchExecution):
        SearchExecution.setObjectName("SearchExecution")
        SearchExecution.resize(400, 300)
        self.verticalLayout = QtWidgets.QVBoxLayout(SearchExecution)
        self.verticalLayout.setObjectName("verticalLayout")
        self.SearchLabel = QtWidgets.QLabel(SearchExecution)
        self.SearchLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.SearchLabel.setObjectName("SearchLabel")
        self.verticalLayout.addWidget(self.SearchLabel)
        self.SearchProgressBar = QtWidgets.QProgressBar(SearchExecution)
        self.SearchProgressBar.setProperty("value", 24)
        self.SearchProgressBar.setObjectName("SearchProgressBar")
        self.verticalLayout.addWidget(self.SearchProgressBar)
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

