# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'lidar_interface\ui\ConnectUnityWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ConnectUnityWindow(object):
    def setupUi(self, ConnectUnityWindow):
        ConnectUnityWindow.setObjectName("ConnectUnityWindow")
        ConnectUnityWindow.resize(758, 375)
        self.label_title = QtWidgets.QLabel(ConnectUnityWindow)
        self.label_title.setGeometry(QtCore.QRect(10, 10, 201, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_title.setFont(font)
        self.label_title.setObjectName("label_title")
        self.Button_back = QtWidgets.QPushButton(ConnectUnityWindow)
        self.Button_back.setGeometry(QtCore.QRect(10, 340, 82, 25))
        self.Button_back.setObjectName("Button_back")
        self.Button_Connect = QtWidgets.QPushButton(ConnectUnityWindow)
        self.Button_Connect.setGeometry(QtCore.QRect(110, 340, 91, 25))
        self.Button_Connect.setObjectName("Button_Connect")
        self.Button_start = QtWidgets.QPushButton(ConnectUnityWindow)
        self.Button_start.setGeometry(QtCore.QRect(220, 340, 91, 25))
        self.Button_start.setObjectName("Button_start")
        self.Button_stop = QtWidgets.QPushButton(ConnectUnityWindow)
        self.Button_stop.setGeometry(QtCore.QRect(330, 340, 91, 25))
        self.Button_stop.setObjectName("Button_stop")
        self.verticalLayoutWidget = QtWidgets.QWidget(ConnectUnityWindow)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(30, 40, 691, 281))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")

        self.retranslateUi(ConnectUnityWindow)
        QtCore.QMetaObject.connectSlotsByName(ConnectUnityWindow)

    def retranslateUi(self, ConnectUnityWindow):
        _translate = QtCore.QCoreApplication.translate
        ConnectUnityWindow.setWindowTitle(_translate("ConnectUnityWindow", "ConnectUnityWindow"))
        self.label_title.setText(_translate("ConnectUnityWindow", "Connect Unity"))
        self.Button_back.setText(_translate("ConnectUnityWindow", "Back"))
        self.Button_Connect.setText(_translate("ConnectUnityWindow", "Connect"))
        self.Button_start.setText(_translate("ConnectUnityWindow", "Start"))
        self.Button_stop.setText(_translate("ConnectUnityWindow", "Stop"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ConnectUnityWindow = QtWidgets.QWidget()
    ui = Ui_ConnectUnityWindow()
    ui.setupUi(ConnectUnityWindow)
    ConnectUnityWindow.show()
    sys.exit(app.exec_())
