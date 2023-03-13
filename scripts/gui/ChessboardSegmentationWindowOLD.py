# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_SegmentationExecution(object):
    def setupUi(self, SegmentationExecution):
        SegmentationExecution.setObjectName("SegmentationExecution")
        SegmentationExecution.resize(400, 300)
        self.verticalLayout = QtWidgets.QVBoxLayout(SegmentationExecution)
        self.verticalLayout.setObjectName("verticalLayout")
        self.SegmentationLabel = QtWidgets.QLabel(SegmentationExecution)
        self.SegmentationLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.SegmentationLabel.setObjectName("SegmentationLabel")
        self.verticalLayout.addWidget(self.SegmentationLabel)
        self.SegmentationProgressBar = QtWidgets.QProgressBar(SegmentationExecution)
        self.SegmentationProgressBar.setProperty("value", 24)
        self.SegmentationProgressBar.setObjectName("SegmentationProgressBar")
        self.verticalLayout.addWidget(self.SegmentationProgressBar)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.OperationCompletedLabel = QtWidgets.QLabel(SegmentationExecution)
        self.OperationCompletedLabel.setEnabled(False)
        self.OperationCompletedLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.OperationCompletedLabel.setObjectName("OperationCompletedLabel")
        self.horizontalLayout.addWidget(self.OperationCompletedLabel)
        self.CheckSegmentationPushButton = QtWidgets.QPushButton(SegmentationExecution)
        self.CheckSegmentationPushButton.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.CheckSegmentationPushButton.sizePolicy().hasHeightForWidth())
        self.CheckSegmentationPushButton.setSizePolicy(sizePolicy)
        self.CheckSegmentationPushButton.setMaximumSize(QtCore.QSize(16777215, 50))
        self.CheckSegmentationPushButton.setObjectName("CheckSegmentationPushButton")
        self.horizontalLayout.addWidget(self.CheckSegmentationPushButton)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(SegmentationExecution)
        QtCore.QMetaObject.connectSlotsByName(SegmentationExecution)


    def retranslateUi(self, SegmentationExecution):
        _translate = QtCore.QCoreApplication.translate
        SegmentationExecution.setWindowTitle(_translate("SegmentationExecution", "Form"))
        self.SegmentationLabel.setText(_translate("SegmentationExecution", "Chessboard segmentation in progress..."))
        self.OperationCompletedLabel.setText(_translate("SegmentationExecution", "Operation Completed!"))
        self.CheckSegmentationPushButton.setText(_translate("SegmentationExecution", "Check Segmentation"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    SegmentationExecution = QtWidgets.QWidget()
    ui = Ui_SegmentationExecution()
    ui.setupUi(SegmentationExecution)
    SegmentationExecution.show()
    sys.exit(app.exec_())

