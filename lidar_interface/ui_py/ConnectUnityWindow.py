import sys
from PyQt5 import QtGui,QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout




# 기존에 덮어쓴 ConnectUnityWindow.py의 UI 클래스
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

        self.retranslateUi(ConnectUnityWindow)
        QtCore.QMetaObject.connectSlotsByName(ConnectUnityWindow)

    def retranslateUi(self, ConnectUnityWindow):
        _translate = QtCore.QCoreApplication.translate
        ConnectUnityWindow.setWindowTitle(_translate("ConnectUnityWindow", "Connect Unity"))
        self.label_title.setText(_translate("ConnectUnityWindow", "Connect Unity"))
        self.Button_back.setText(_translate("ConnectUnityWindow", "Back"))
        self.Button_Connect.setText(_translate("ConnectUnityWindow", "Connect"))
        self.Button_start.setText(_translate("ConnectUnityWindow", "Start"))
        self.Button_stop.setText(_translate("ConnectUnityWindow", "Stop"))



# ConnectUnityWindow에 그래프 추가하기 위한 클래스
class ConnectUnityApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # 기존 UI를 로드
        self.ui = Ui_ConnectUnityWindow()
        self.ui.setupUi(self)

        # 그래프 캔버스를 추가할 레이아웃 위젯을 가져옴
        self.layout_widget = self.ui.Button_Connect.parent()

        # 그래프 캔버스 추가
        self.canvas = MplCanvas(self.layout_widget, width=5, height=4, dpi=100)
        self.ui.verticalLayout = QtWidgets.QVBoxLayout(self.layout_widget)
        self.ui.verticalLayout.addWidget(self.canvas)

        # 타이머 설정: 0.05초마다 업데이트
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.canvas.update_plot)

        # Start 버튼과 Stop 버튼 연결
        self.ui.Button_start.clicked.connect(self.start_graph)
        self.ui.Button_stop.clicked.connect(self.stop_graph)

    def start_graph(self):
        """Start 버튼 클릭 시 그래프 업데이트 시작"""
        self.timer.start()

    def stop_graph(self):
        """Stop 버튼 클릭 시 그래프 업데이트 중지"""
        self.timer.stop()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    connect_unity_app = ConnectUnityApp()
    connect_unity_app.show()

    sys.exit(app.exec_())
