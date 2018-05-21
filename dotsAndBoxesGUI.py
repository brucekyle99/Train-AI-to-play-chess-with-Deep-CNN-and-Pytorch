'''
game GUI
'''

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPalette
from dotAndBoxes import *
class Ui_MainWindow(object):
    def setupUi(self, MainWindow,boardSize):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(900, 700)
        MainWindow.setMinimumSize(QtCore.QSize(900, 700))
        MainWindow.setMaximumSize(QtCore.QSize(900, 700))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.chessBoard = ChessBoard(boardSize=boardSize,parent=self.centralwidget)
        self.chessBoard.setGeometry(QtCore.QRect(0, 0, 700, 700))
        self.chessBoard.setMinimumSize(QtCore.QSize(700, 700))
        self.chessBoard.setMaximumSize(QtCore.QSize(700, 700))

        # self.chessBoard.setStyleSheet("border-image: url(:/img/img/chessBoard_4_4.png);")

        self.chessBoard.setAutoFillBackground(True)
        palette1 = QPalette()  # 设置棋盘背景
        if boardSize == 4:
            palette1.setBrush(self.backgroundRole(), QtGui.QBrush(QtGui.QPixmap('img/chessBoard_4_4.png')))
        elif boardSize == 6:
            palette1.setBrush(self.backgroundRole(), QtGui.QBrush(QtGui.QPixmap('img/chessBoard_6_6.png')))
        self.chessBoard.setPalette(palette1)


        self.chessBoard.setObjectName("chessBoard")

        self.save_model = QtWidgets.QPushButton(self.centralwidget)
        self.save_model.setGeometry(QtCore.QRect(750, 450, 101, 51))
        self.save_model.setObjectName("save_model")

        self.save_data = QtWidgets.QPushButton(self.centralwidget)
        self.save_data.setGeometry(QtCore.QRect(750, 520, 101, 51))
        self.save_data.setObjectName("save_data")

        self.startTrain = QtWidgets.QPushButton(self.centralwidget)
        self.startTrain.setGeometry(QtCore.QRect(750, 590, 101, 51))
        self.startTrain.setObjectName("startTrain")

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.startTrain.setText(_translate("MainWindow", "开始训练"))
        self.save_data.setText(_translate("MainWindow", "保存数据"))
        self.save_model.setText(_translate("MainWindow", "保存模型"))

