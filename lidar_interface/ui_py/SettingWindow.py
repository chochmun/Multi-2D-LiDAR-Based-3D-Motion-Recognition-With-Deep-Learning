# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'lidar_interface\ui\SettingWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_SettingWindow(object):
    def setupUi(self, SettingWindow):
        SettingWindow.setObjectName("SettingWindow")
        SettingWindow.resize(600, 300)
        self.Label_title = QtWidgets.QLabel(SettingWindow)
        self.Label_title.setGeometry(QtCore.QRect(520, 10, 71, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.Label_title.setFont(font)
        self.Label_title.setObjectName("Label_title")
        self.Button_back = QtWidgets.QPushButton(SettingWindow)
        self.Button_back.setGeometry(QtCore.QRect(10, 260, 82, 25))
        self.Button_back.setObjectName("Button_back")
        self.Button_savesetting = QtWidgets.QPushButton(SettingWindow)
        self.Button_savesetting.setGeometry(QtCore.QRect(120, 260, 111, 25))
        self.Button_savesetting.setObjectName("Button_savesetting")
        self.TabSetting = QtWidgets.QTabWidget(SettingWindow)
        self.TabSetting.setGeometry(QtCore.QRect(20, 20, 561, 231))
        self.TabSetting.setObjectName("TabSetting")
        self.Tab_basic = QtWidgets.QWidget()
        self.Tab_basic.setObjectName("Tab_basic")
        self.label_text1 = QtWidgets.QLabel(self.Tab_basic)
        self.label_text1.setGeometry(QtCore.QRect(30, 30, 111, 16))
        self.label_text1.setObjectName("label_text1")
        self.Input_maxdist = QtWidgets.QLineEdit(self.Tab_basic)
        self.Input_maxdist.setGeometry(QtCore.QRect(140, 30, 121, 20))
        self.Input_maxdist.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.Input_maxdist.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.Input_maxdist.setObjectName("Input_maxdist")
        self.label_text3 = QtWidgets.QLabel(self.Tab_basic)
        self.label_text3.setGeometry(QtCore.QRect(30, 60, 91, 16))
        self.label_text3.setObjectName("label_text3")
        self.radioButton_90 = QtWidgets.QRadioButton(self.Tab_basic)
        self.radioButton_90.setGeometry(QtCore.QRect(150, 60, 98, 17))
        self.radioButton_90.setChecked(True)
        self.radioButton_90.setObjectName("radioButton_90")
        self.radioButton_180 = QtWidgets.QRadioButton(self.Tab_basic)
        self.radioButton_180.setGeometry(QtCore.QRect(270, 60, 141, 17))
        self.radioButton_180.setObjectName("radioButton_180")
        self.label_text2 = QtWidgets.QLabel(self.Tab_basic)
        self.label_text2.setGeometry(QtCore.QRect(290, 30, 161, 16))
        self.label_text2.setObjectName("label_text2")
        self.label_text4 = QtWidgets.QLabel(self.Tab_basic)
        self.label_text4.setGeometry(QtCore.QRect(30, 90, 171, 16))
        self.label_text4.setObjectName("label_text4")
        self.Input_buzzduration = QtWidgets.QLineEdit(self.Tab_basic)
        self.Input_buzzduration.setGeometry(QtCore.QRect(210, 90, 51, 20))
        self.Input_buzzduration.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.Input_buzzduration.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.Input_buzzduration.setObjectName("Input_buzzduration")
        self.label_text5 = QtWidgets.QLabel(self.Tab_basic)
        self.label_text5.setGeometry(QtCore.QRect(270, 90, 161, 16))
        self.label_text5.setObjectName("label_text5")
        self.label_text4_2 = QtWidgets.QLabel(self.Tab_basic)
        self.label_text4_2.setGeometry(QtCore.QRect(30, 120, 211, 16))
        self.label_text4_2.setObjectName("label_text4_2")
        self.Input_lidar_fps = QtWidgets.QLineEdit(self.Tab_basic)
        self.Input_lidar_fps.setGeometry(QtCore.QRect(250, 120, 51, 20))
        self.Input_lidar_fps.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.Input_lidar_fps.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.Input_lidar_fps.setObjectName("Input_lidar_fps")
        self.label_text6 = QtWidgets.QLabel(self.Tab_basic)
        self.label_text6.setGeometry(QtCore.QRect(310, 120, 161, 16))
        self.label_text6.setObjectName("label_text6")
        self.TabSetting.addTab(self.Tab_basic, "")
        self.Tab_port = QtWidgets.QWidget()
        self.Tab_port.setObjectName("Tab_port")
        self.Label_port_top = QtWidgets.QLabel(self.Tab_port)
        self.Label_port_top.setGeometry(QtCore.QRect(30, 40, 60, 13))
        self.Label_port_top.setObjectName("Label_port_top")
        self.Label_port_mid = QtWidgets.QLabel(self.Tab_port)
        self.Label_port_mid.setGeometry(QtCore.QRect(30, 70, 60, 13))
        self.Label_port_mid.setObjectName("Label_port_mid")
        self.Label_port_bot = QtWidgets.QLabel(self.Tab_port)
        self.Label_port_bot.setGeometry(QtCore.QRect(30, 100, 60, 13))
        self.Label_port_bot.setObjectName("Label_port_bot")
        self.Input_port_top = QtWidgets.QLineEdit(self.Tab_port)
        self.Input_port_top.setGeometry(QtCore.QRect(100, 40, 113, 20))
        self.Input_port_top.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.Input_port_top.setObjectName("Input_port_top")
        self.Input_port_mid = QtWidgets.QLineEdit(self.Tab_port)
        self.Input_port_mid.setGeometry(QtCore.QRect(100, 70, 113, 20))
        self.Input_port_mid.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.Input_port_mid.setObjectName("Input_port_mid")
        self.Input_port_bot = QtWidgets.QLineEdit(self.Tab_port)
        self.Input_port_bot.setGeometry(QtCore.QRect(100, 100, 113, 20))
        self.Input_port_bot.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.Input_port_bot.setObjectName("Input_port_bot")
        self.Label_explain = QtWidgets.QLabel(self.Tab_port)
        self.Label_explain.setGeometry(QtCore.QRect(280, 40, 251, 91))
        self.Label_explain.setLineWidth(0)
        self.Label_explain.setTextFormat(QtCore.Qt.AutoText)
        self.Label_explain.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.Label_explain.setWordWrap(True)
        self.Label_explain.setObjectName("Label_explain")
        self.FuncLable_portlist = QtWidgets.QLabel(self.Tab_port)
        self.FuncLable_portlist.setGeometry(QtCore.QRect(100, 150, 401, 16))
        self.FuncLable_portlist.setObjectName("FuncLable_portlist")
        self.Label_portlist = QtWidgets.QLabel(self.Tab_port)
        self.Label_portlist.setGeometry(QtCore.QRect(30, 150, 60, 13))
        self.Label_portlist.setObjectName("Label_portlist")
        self.TabSetting.addTab(self.Tab_port, "")
        self.Tab_env = QtWidgets.QWidget()
        self.Tab_env.setObjectName("Tab_env")
        self.List_env = QtWidgets.QListWidget(self.Tab_env)
        self.List_env.setGeometry(QtCore.QRect(10, 10, 241, 186))
        self.List_env.setObjectName("List_env")
        self.FuncLabel_selected_env = QtWidgets.QLabel(self.Tab_env)
        self.FuncLabel_selected_env.setGeometry(QtCore.QRect(280, 20, 231, 16))
        self.FuncLabel_selected_env.setObjectName("FuncLabel_selected_env")
        self.TabSetting.addTab(self.Tab_env, "")
        self.Tab_model = QtWidgets.QWidget()
        self.Tab_model.setObjectName("Tab_model")
        self.FuncLabel_selected_model = QtWidgets.QLabel(self.Tab_model)
        self.FuncLabel_selected_model.setGeometry(QtCore.QRect(280, 20, 231, 16))
        self.FuncLabel_selected_model.setObjectName("FuncLabel_selected_model")
        self.List_model = QtWidgets.QListWidget(self.Tab_model)
        self.List_model.setGeometry(QtCore.QRect(10, 10, 241, 186))
        self.List_model.setObjectName("List_model")
        self.TabSetting.addTab(self.Tab_model, "")

        self.retranslateUi(SettingWindow)
        self.TabSetting.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(SettingWindow)

    def retranslateUi(self, SettingWindow):
        _translate = QtCore.QCoreApplication.translate
        SettingWindow.setWindowTitle(_translate("SettingWindow", "SettingWindow"))
        self.Label_title.setText(_translate("SettingWindow", "Settings"))
        self.Button_back.setText(_translate("SettingWindow", "Back"))
        self.Button_savesetting.setText(_translate("SettingWindow", "Setting Save"))
        self.label_text1.setText(_translate("SettingWindow", "Max dist ( mm) :"))
        self.Input_maxdist.setText(_translate("SettingWindow", "2000"))
        self.label_text3.setText(_translate("SettingWindow", "Angle Choice :"))
        self.radioButton_90.setText(_translate("SettingWindow", "90 (Solo)"))
        self.radioButton_180.setText(_translate("SettingWindow", "180 ( Multi )"))
        self.label_text2.setText(_translate("SettingWindow", "2000mm (recommend)"))
        self.label_text4.setText(_translate("SettingWindow", "Start Buzz Sound Duration :"))
        self.Input_buzzduration.setText(_translate("SettingWindow", "3"))
        self.label_text5.setText(_translate("SettingWindow", "second"))
        self.label_text4_2.setText(_translate("SettingWindow", "Frequeny of getting Lidars datas :"))
        self.Input_lidar_fps.setText(_translate("SettingWindow", "20"))
        self.label_text6.setText(_translate("SettingWindow", "FPS"))
        self.TabSetting.setTabText(self.TabSetting.indexOf(self.Tab_basic), _translate("SettingWindow", "Basic set"))
        self.Label_port_top.setText(_translate("SettingWindow", "Port Top :"))
        self.Label_port_mid.setText(_translate("SettingWindow", "Port Mid :"))
        self.Label_port_bot.setText(_translate("SettingWindow", "Port Bot :"))
        self.Input_port_top.setText(_translate("SettingWindow", "None"))
        self.Input_port_mid.setText(_translate("SettingWindow", "None"))
        self.Input_port_bot.setText(_translate("SettingWindow", "None"))
        self.Label_explain.setText(_translate("SettingWindow", "메인 메뉴의 Data View에서\n"
"인식되는 라이다의 상,중,하 위치를 보고\n"
"필요 시 Port 위치를 재배열하십시오.\n"
"또한 아래의 포트 리스트를 보고 1,2,3을 재배열 하십시오,"))
        self.FuncLable_portlist.setText(_translate("SettingWindow", "error"))
        self.Label_portlist.setText(_translate("SettingWindow", "Port List :"))
        self.TabSetting.setTabText(self.TabSetting.indexOf(self.Tab_port), _translate("SettingWindow", "port"))
        self.FuncLabel_selected_env.setText(_translate("SettingWindow", "selected model file"))
        self.TabSetting.setTabText(self.TabSetting.indexOf(self.Tab_env), _translate("SettingWindow", "Environment"))
        self.FuncLabel_selected_model.setText(_translate("SettingWindow", "selected model file"))
        self.TabSetting.setTabText(self.TabSetting.indexOf(self.Tab_model), _translate("SettingWindow", "AI Model"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    SettingWindow = QtWidgets.QWidget()
    ui = Ui_SettingWindow()
    ui.setupUi(SettingWindow)
    SettingWindow.show()
    sys.exit(app.exec_())
