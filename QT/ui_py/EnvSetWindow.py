# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'QT\ui\EnvSetWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_EnvSetWindow(object):
    def setupUi(self, EnvSetWindow):
        EnvSetWindow.setObjectName("EnvSetWindow")
        EnvSetWindow.resize(600, 300)
        self.Label_title = QtWidgets.QLabel(EnvSetWindow)
        self.Label_title.setGeometry(QtCore.QRect(10, 20, 201, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.Label_title.setFont(font)
        self.Label_title.setObjectName("Label_title")
        self.Input_json = QtWidgets.QLineEdit(EnvSetWindow)
        self.Input_json.setGeometry(QtCore.QRect(150, 60, 113, 20))
        self.Input_json.setAccessibleName("")
        self.Input_json.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.Input_json.setObjectName("Input_json")
        self.Button_back = QtWidgets.QPushButton(EnvSetWindow)
        self.Button_back.setGeometry(QtCore.QRect(10, 270, 82, 25))
        self.Button_back.setObjectName("Button_back")
        self.Label_json = QtWidgets.QLabel(EnvSetWindow)
        self.Label_json.setGeometry(QtCore.QRect(30, 60, 141, 16))
        self.Label_json.setObjectName("Label_json")
        self.Label_loadingtime = QtWidgets.QLabel(EnvSetWindow)
        self.Label_loadingtime.setGeometry(QtCore.QRect(30, 100, 91, 16))
        self.Label_loadingtime.setObjectName("Label_loadingtime")
        self.Input_loadingtime = QtWidgets.QLineEdit(EnvSetWindow)
        self.Input_loadingtime.setGeometry(QtCore.QRect(150, 100, 113, 20))
        self.Input_loadingtime.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.Input_loadingtime.setObjectName("Input_loadingtime")
        self.Label_margindist = QtWidgets.QLabel(EnvSetWindow)
        self.Label_margindist.setGeometry(QtCore.QRect(30, 140, 111, 16))
        self.Label_margindist.setObjectName("Label_margindist")
        self.Input_margindist = QtWidgets.QLineEdit(EnvSetWindow)
        self.Input_margindist.setGeometry(QtCore.QRect(150, 140, 113, 20))
        self.Input_margindist.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.Input_margindist.setObjectName("Input_margindist")
        self.Button_adjust = QtWidgets.QPushButton(EnvSetWindow)
        self.Button_adjust.setGeometry(QtCore.QRect(50, 190, 91, 25))
        self.Button_adjust.setObjectName("Button_adjust")
        self.Label_explain = QtWidgets.QLabel(EnvSetWindow)
        self.Label_explain.setGeometry(QtCore.QRect(310, 50, 221, 201))
        self.Label_explain.setLineWidth(0)
        self.Label_explain.setTextFormat(QtCore.Qt.AutoText)
        self.Label_explain.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.Label_explain.setWordWrap(True)
        self.Label_explain.setObjectName("Label_explain")
        self.Button_save = QtWidgets.QPushButton(EnvSetWindow)
        self.Button_save.setGeometry(QtCore.QRect(160, 190, 91, 25))
        self.Button_save.setObjectName("Button_save")
        self.Label_saveformat = QtWidgets.QLabel(EnvSetWindow)
        self.Label_saveformat.setGeometry(QtCore.QRect(50, 230, 201, 16))
        self.Label_saveformat.setObjectName("Label_saveformat")

        self.retranslateUi(EnvSetWindow)
        QtCore.QMetaObject.connectSlotsByName(EnvSetWindow)

    def retranslateUi(self, EnvSetWindow):
        _translate = QtCore.QCoreApplication.translate
        EnvSetWindow.setWindowTitle(_translate("EnvSetWindow", "EnvSetWindow"))
        self.Label_title.setText(_translate("EnvSetWindow", "Environment Set"))
        self.Input_json.setText(_translate("EnvSetWindow", "default"))
        self.Button_back.setText(_translate("EnvSetWindow", "Back"))
        self.Label_json.setText(_translate("EnvSetWindow", "json file name : "))
        self.Label_loadingtime.setText(_translate("EnvSetWindow", "loading time :"))
        self.Input_loadingtime.setText(_translate("EnvSetWindow", "10"))
        self.Label_margindist.setText(_translate("EnvSetWindow", "Margin distance :"))
        self.Input_margindist.setText(_translate("EnvSetWindow", "0"))
        self.Button_adjust.setText(_translate("EnvSetWindow", "Adjust"))
        self.Label_explain.setText(_translate("EnvSetWindow", "환경을 저장하는 동안 라이다모듈 근처에 최대한 장애물을 없게 하는 것이 권장됩니다.\n"
"\n"
"Loading Time은 환경정보를 읽어오는 시간이며, 적정 10초가 권장됩니다.\n"
"\n"
"Margin_distance( 단위 : mm)은\n"
"환경정보를 저장할때 조금더 여유를 줄 수 있는 거리를 의미합니다. 0~30mm가 권장됩니다."))
        self.Button_save.setText(_translate("EnvSetWindow", "Start Save"))
        self.Label_saveformat.setText(_translate("EnvSetWindow", "Save format : <file_name>.json"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    EnvSetWindow = QtWidgets.QWidget()
    ui = Ui_EnvSetWindow()
    ui.setupUi(EnvSetWindow)
    EnvSetWindow.show()
    sys.exit(app.exec_())
