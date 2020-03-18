from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class UI(object):
    def setupUi(self, launcherWindow):
        launcherWindow.setObjectName("launcherWindow")
        launcherWindow.resize(1000,650)
        self.centralwidget = QtWidgets.QWidget(launcherWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.background=QtWidgets.QLabel(self.centralwidget)
        self.background.setGeometry(QtCore.QRect(0,0,1000,650))
        self.background.setStyleSheet("background-color:rbg(51,51,51)")

        self.open = QtWidgets.QPushButton(self.centralwidget)
        self.open.setGeometry(QtCore.QRect(25, 520, 270, 60))
        self.open.setStyleSheet("background-color:rgb(110,200,209);color:white;border-style:outset;font-weight:bold;font-family:幼圆,YouYuan;font-size:25px")
        self.open.setObjectName("open")
        self.denoising = QtWidgets.QPushButton(self.centralwidget)
        self.denoising.setGeometry(QtCore.QRect(320, 520, 270, 60))
        self.denoising.setStyleSheet("background-color:rgb(110,200,209);color:white;border-style:outset;font-weight:bold;font-family:幼圆,YouYuan;font-size:25px")
        self.denoising.setObjectName("denoising")
        self.save = QtWidgets.QPushButton(self.centralwidget)
        self.save.setGeometry(QtCore.QRect(615, 520, 270, 60))
        self.save.setStyleSheet("background-color:rgb(110,200,209);color:white;border-style:outset;font-weight:bold;font-family:幼圆,YouYuan;font-size:25px")
        self.save.setObjectName("save")
        self.exit = QtWidgets.QPushButton(self.centralwidget)
        self.exit.setGeometry(QtCore.QRect(915, 520, 60, 60))
        self.exit.setStyleSheet("QPushButton{background-image: url(DATA/UIFiles/exit.png);border-style:outset}")
        self.exit.setObjectName("exit")

        self.orgPic=QtWidgets.QLabel(self.centralwidget)
        self.orgPic.setGeometry(QtCore.QRect(25, 25, 465, 475))
        self.orgPic.setStyleSheet("background-color:rgb(51,51,51);border-style:dotted;border-width:1px;border-color:rgb(110,200,209);color:white;font-weight:bold;font-family:幼圆,YouYuan;font-size:15px")
        self.orgPic.setAlignment(Qt.AlignCenter)
        self.denoisedPic=QtWidgets.QLabel(self.centralwidget)
        self.denoisedPic.setGeometry(QtCore.QRect(510, 25, 465, 475))
        self.denoisedPic.setStyleSheet("background-color:rgb(51,51,51);border-style:dotted;border-width:1px;border-color:rgb(110,200,209);color:white;font-weight:bold;font-family:幼圆,YouYuan;font-size:15px")
        self.denoisedPic.setAlignment(Qt.AlignCenter)

        self.saveTip=QtWidgets.QLabel(self.centralwidget)
        self.saveTip.setGeometry(QtCore.QRect(0, 200, 1000, 100))
        self.saveTip.setStyleSheet("background-color:rgb(51,51,51);border-style:dotted;border-width:1px;border-color:rgb(110,200,209);color:white;font-weight:bold;font-family:幼圆,YouYuan;font-size:20px")
        self.saveTip.setAlignment(Qt.AlignCenter)
        self.saveTip.setVisible(False)

        self.buttonDisable=QtWidgets.QLabel(self.centralwidget)
        self.buttonDisable.setGeometry(QtCore.QRect(0, 520, 1000, 60))
        self.buttonDisable.setStyleSheet("QLabel{background-image: url(DATA/UIFiles/button_disable.png);}")
        self.buttonDisable.close()

        self.title=QtWidgets.QLabel(self.centralwidget)
        self.title.setGeometry(QtCore.QRect(0, 600, 1000, 50))
        self.title.setStyleSheet("color:white;font-weight:bold;font-family:幼圆,YouYuan;font-size:20px")
        self.title.setAlignment(Qt.AlignCenter)

        self.retranslateUi(launcherWindow)
        QtCore.QMetaObject.connectSlotsByName(launcherWindow)

        self.timer=QTimer()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("launcherWindow", "AI图像降噪"))
        self.denoising.setText(_translate("launcherWindow", "降噪"))
        self.open.setText(_translate("launcherWindow", "打开"))
        self.save.setText(_translate("launcherWindow", "保存"))
        self.title.setText('AI图像降噪')
