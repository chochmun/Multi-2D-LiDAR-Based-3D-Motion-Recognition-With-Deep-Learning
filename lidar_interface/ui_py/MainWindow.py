# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'lidar_interface\ui\MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(600, 300)
        MainWindow.setStyleSheet("background-color: #FFFFFF;")
        self.StartWindow = QtWidgets.QWidget(MainWindow)
        self.StartWindow.setObjectName("StartWindow")
        self.Button_dataview = QtWidgets.QPushButton(self.StartWindow)
        self.Button_dataview.setGeometry(QtCore.QRect(30, 60, 161, 51))
        self.Button_dataview.setStyleSheet("background-color: rgb(221, 236, 255);")
        self.Button_dataview.setObjectName("Button_dataview")
        self.Button_setting = QtWidgets.QPushButton(self.StartWindow)
        self.Button_setting.setGeometry(QtCore.QRect(420, 10, 161, 41))
        self.Button_setting.setStyleSheet("background-color: rgb(213, 218, 227);")
        self.Button_setting.setCheckable(False)
        self.Button_setting.setChecked(False)
        self.Button_setting.setAutoRepeatDelay(300)
        self.Button_setting.setAutoDefault(False)
        self.Button_setting.setObjectName("Button_setting")
        self.Button_envset = QtWidgets.QPushButton(self.StartWindow)
        self.Button_envset.setGeometry(QtCore.QRect(210, 60, 161, 51))
        self.Button_envset.setStyleSheet("background-color: rgb(221, 236, 255);")
        self.Button_envset.setObjectName("Button_envset")
        self.Button_transferlearn = QtWidgets.QPushButton(self.StartWindow)
        self.Button_transferlearn.setGeometry(QtCore.QRect(210, 190, 161, 51))
        self.Button_transferlearn.setStyleSheet("background-color: rgb(221, 236, 255);")
        self.Button_transferlearn.setObjectName("Button_transferlearn")
        self.Button_csvsave = QtWidgets.QPushButton(self.StartWindow)
        self.Button_csvsave.setGeometry(QtCore.QRect(210, 120, 161, 51))
        self.Button_csvsave.setStyleSheet("background-color: rgb(221, 236, 255);")
        self.Button_csvsave.setObjectName("Button_csvsave")
        self.label = QtWidgets.QLabel(self.StartWindow)
        self.label.setGeometry(QtCore.QRect(90, 30, 60, 13))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.StartWindow)
        self.label_2.setGeometry(QtCore.QRect(270, 30, 60, 13))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.Button_unity = QtWidgets.QPushButton(self.StartWindow)
        self.Button_unity.setGeometry(QtCore.QRect(30, 190, 161, 51))
        self.Button_unity.setStyleSheet("background-color: rgb(221, 236, 255);")
        self.Button_unity.setObjectName("Button_unity")
        self.graphicsView = QtWidgets.QGraphicsView(self.StartWindow)
        self.graphicsView.setGeometry(QtCore.QRect(-40, -10, 661, 331))
        self.graphicsView.setStyleSheet("\n"
"background-image: url(:/newPrefix/images/background.JPG);\n"
"\n"
"")
        self.graphicsView.setObjectName("graphicsView")
        self.Button_usbconnection = QtWidgets.QPushButton(self.StartWindow)
        self.Button_usbconnection.setGeometry(QtCore.QRect(510, 60, 71, 41))
        font = QtGui.QFont()
        font.setFamily("굴림")
        font.setPointSize(8)
        self.Button_usbconnection.setFont(font)
        self.Button_usbconnection.setStyleSheet("background-color: rgb(213, 218, 227);")
        self.Button_usbconnection.setCheckable(False)
        self.Button_usbconnection.setChecked(False)
        self.Button_usbconnection.setAutoRepeatDelay(300)
        self.Button_usbconnection.setAutoDefault(False)
        self.Button_usbconnection.setObjectName("Button_usbconnection")
        self.Button_wificonnection = QtWidgets.QPushButton(self.StartWindow)
        self.Button_wificonnection.setGeometry(QtCore.QRect(420, 60, 71, 41))
        font = QtGui.QFont()
        font.setFamily("굴림")
        font.setPointSize(8)
        self.Button_wificonnection.setFont(font)
        self.Button_wificonnection.setStyleSheet("background-color: rgb(213, 218, 227);")
        self.Button_wificonnection.setCheckable(False)
        self.Button_wificonnection.setChecked(False)
        self.Button_wificonnection.setAutoRepeatDelay(300)
        self.Button_wificonnection.setAutoDefault(False)
        self.Button_wificonnection.setObjectName("Button_wificonnection")
        self.Button_3dview = QtWidgets.QPushButton(self.StartWindow)
        self.Button_3dview.setGeometry(QtCore.QRect(30, 120, 161, 51))
        self.Button_3dview.setStyleSheet("background-color: rgb(221, 236, 255);")
        self.Button_3dview.setObjectName("Button_3dview")
        self.graphicsView.raise_()
        self.Button_dataview.raise_()
        self.Button_setting.raise_()
        self.Button_envset.raise_()
        self.Button_transferlearn.raise_()
        self.Button_csvsave.raise_()
        self.label.raise_()
        self.label_2.raise_()
        self.Button_unity.raise_()
        self.Button_usbconnection.raise_()
        self.Button_wificonnection.raise_()
        self.Button_3dview.raise_()
        MainWindow.setCentralWidget(self.StartWindow)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Button_dataview.setText(_translate("MainWindow", "Multi-Lidar Data View"))
        self.Button_setting.setText(_translate("MainWindow", "Settings"))
        self.Button_envset.setText(_translate("MainWindow", "Environment Set"))
        self.Button_transferlearn.setText(_translate("MainWindow", "Transfer Learning"))
        self.Button_csvsave.setText(_translate("MainWindow", "Save to CSV"))
        self.label.setText(_translate("MainWindow", "View"))
        self.label_2.setText(_translate("MainWindow", "Utility"))
        self.Button_unity.setText(_translate("MainWindow", "Motion Estimation\n"
"And Unity"))
        self.Button_usbconnection.setText(_translate("MainWindow", "Lidar\n"
"Connection"))
        self.Button_wificonnection.setText(_translate("MainWindow", "Wi-Fi\n"
"Connection"))
        self.Button_3dview.setText(_translate("MainWindow", "3D View"))
import qrc.background_rc


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
