# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'lidar_interface\ui\TransferLearnWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_TransferLearnWindow(object):
    def setupUi(self, TransferLearnWindow):
        TransferLearnWindow.setObjectName("TransferLearnWindow")
        TransferLearnWindow.resize(600, 301)
        self.label_title = QtWidgets.QLabel(TransferLearnWindow)
        self.label_title.setGeometry(QtCore.QRect(10, 20, 201, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_title.setFont(font)
        self.label_title.setObjectName("label_title")
        self.Button_back = QtWidgets.QPushButton(TransferLearnWindow)
        self.Button_back.setGeometry(QtCore.QRect(10, 260, 82, 25))
        self.Button_back.setObjectName("Button_back")
        self.Button_start = QtWidgets.QPushButton(TransferLearnWindow)
        self.Button_start.setGeometry(QtCore.QRect(110, 260, 91, 25))
        self.Button_start.setObjectName("Button_start")
        self.Button_stop = QtWidgets.QPushButton(TransferLearnWindow)
        self.Button_stop.setGeometry(QtCore.QRect(230, 260, 91, 25))
        self.Button_stop.setObjectName("Button_stop")
        self.List_pose = QtWidgets.QListWidget(TransferLearnWindow)
        self.List_pose.setGeometry(QtCore.QRect(30, 50, 201, 181))
        self.List_pose.setObjectName("List_pose")
        self.Label_explain = QtWidgets.QLabel(TransferLearnWindow)
        self.Label_explain.setGeometry(QtCore.QRect(310, 50, 221, 201))
        self.Label_explain.setLineWidth(0)
        self.Label_explain.setTextFormat(QtCore.Qt.AutoText)
        self.Label_explain.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.Label_explain.setWordWrap(True)
        self.Label_explain.setObjectName("Label_explain")

        self.retranslateUi(TransferLearnWindow)
        QtCore.QMetaObject.connectSlotsByName(TransferLearnWindow)

    def retranslateUi(self, TransferLearnWindow):
        _translate = QtCore.QCoreApplication.translate
        TransferLearnWindow.setWindowTitle(_translate("TransferLearnWindow", "TransferLearnWindow"))
        self.label_title.setText(_translate("TransferLearnWindow", "Transfer Learning"))
        self.Button_back.setText(_translate("TransferLearnWindow", "Back"))
        self.Button_start.setText(_translate("TransferLearnWindow", "Start"))
        self.Button_stop.setText(_translate("TransferLearnWindow", "Stop"))
        self.Label_explain.setText(_translate("TransferLearnWindow", "좌측 리스트에 보이는 동작이\n"
"목표로하는 모션들입니다.\n"
"\n"
"전이학습 시에는 반드시 필터가\n"
"잘 동작하는지 Data View를\n"
"통해 확인하는 것이 권장 됩니다"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    TransferLearnWindow = QtWidgets.QWidget()
    ui = Ui_TransferLearnWindow()
    ui.setupUi(TransferLearnWindow)
    TransferLearnWindow.show()
    sys.exit(app.exec_())
