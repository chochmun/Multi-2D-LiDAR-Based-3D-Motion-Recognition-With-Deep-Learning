# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'lidar_interface\ui\SkeletonViewWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_SkeletonViewWindow(object):
    def setupUi(self, SkeletonViewWindow):
        SkeletonViewWindow.setObjectName("SkeletonViewWindow")
        SkeletonViewWindow.resize(600, 300)
        self.label_title = QtWidgets.QLabel(SkeletonViewWindow)
        self.label_title.setGeometry(QtCore.QRect(10, 20, 201, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_title.setFont(font)
        self.label_title.setObjectName("label_title")
        self.Button_back = QtWidgets.QPushButton(SkeletonViewWindow)
        self.Button_back.setGeometry(QtCore.QRect(10, 260, 82, 25))
        self.Button_back.setObjectName("Button_back")
        self.Button_start = QtWidgets.QPushButton(SkeletonViewWindow)
        self.Button_start.setGeometry(QtCore.QRect(110, 260, 91, 25))
        self.Button_start.setObjectName("Button_start")
        self.Button_stop = QtWidgets.QPushButton(SkeletonViewWindow)
        self.Button_stop.setGeometry(QtCore.QRect(230, 260, 91, 25))
        self.Button_stop.setObjectName("Button_stop")
        self.List_pose = QtWidgets.QListWidget(SkeletonViewWindow)
        self.List_pose.setGeometry(QtCore.QRect(30, 50, 201, 181))
        self.List_pose.setObjectName("List_pose")
        self.Label_explain = QtWidgets.QLabel(SkeletonViewWindow)
        self.Label_explain.setGeometry(QtCore.QRect(310, 50, 221, 201))
        self.Label_explain.setLineWidth(0)
        self.Label_explain.setTextFormat(QtCore.Qt.AutoText)
        self.Label_explain.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.Label_explain.setWordWrap(True)
        self.Label_explain.setObjectName("Label_explain")

        self.retranslateUi(SkeletonViewWindow)
        QtCore.QMetaObject.connectSlotsByName(SkeletonViewWindow)

    def retranslateUi(self, SkeletonViewWindow):
        _translate = QtCore.QCoreApplication.translate
        SkeletonViewWindow.setWindowTitle(_translate("SkeletonViewWindow", "SkeletonViewWindow"))
        self.label_title.setText(_translate("SkeletonViewWindow", "Transfer Learning"))
        self.Button_back.setText(_translate("SkeletonViewWindow", "Back"))
        self.Button_start.setText(_translate("SkeletonViewWindow", "Start"))
        self.Button_stop.setText(_translate("SkeletonViewWindow", "Stop"))
        self.Label_explain.setText(_translate("SkeletonViewWindow", "좌측 리스트에 보이는 동작을\n"
"동작마다 2초씩 5번 취해주세요.\n"
"\n"
"Start버튼을 누르시면 상세한 설명이 음성으로 안내 되오니\n"
"안내 절차에 따라주십시오"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    SkeletonViewWindow = QtWidgets.QWidget()
    ui = Ui_SkeletonViewWindow()
    ui.setupUi(SkeletonViewWindow)
    SkeletonViewWindow.show()
    sys.exit(app.exec_())
