import sys
from PyQt5 import QtGui,QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

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

# MplCanvas 클래스 정의: 그래프 생성 및 실시간 업데이트
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=7, height=2.5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes1 = fig.add_subplot(121)  # 첫 번째 서브플롯 (좌)
        self.axes2 = fig.add_subplot(122)  # 두 번째 서브플롯 (우)
        super(MplCanvas, self).__init__(fig)
        self.setFixedSize(700, 250)  # 그래프 크기 고정
        self.init_plot()

    def init_plot(self):
        labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']  # Y축 레이블 설정

        # 첫 번째 서브플롯 초기화
        self.bar1 = self.axes1.barh(np.arange(10), np.zeros(10), color='#17FD79', height=0.8)
        self.axes1.set_xlim(0, 1)
        self.axes1.set_yticks(np.arange(10))
        self.axes1.set_yticklabels(labels)  # Y축 레이블을 'a' ~ 'j'로 설정
        self.axes1.set_xticks(np.arange(0, 1.1, 0.1))  # X축 눈금 0.1 단위로 설정

        # 두 번째 서브플롯 초기화
        self.bar2 = self.axes2.barh(np.arange(10), np.zeros(10), color='#17FD79', height=0.8)
        self.axes2.set_xlim(0, 1)
        self.axes2.set_yticks(np.arange(10))
        self.axes2.set_yticklabels(labels)  # Y축 레이블을 'a' ~ 'j'로 설정
        self.axes2.set_xticks(np.arange(0, 1.1, 0.1))  # X축 눈금 0.1 단위로 설정

        self.figure.tight_layout()

    def update_plot(self):
        # 첫 번째 서브플롯 데이터 업데이트
        values1 = np.random.rand(10)
        max_val1 = np.max(values1)
        for i, bar in enumerate(self.bar1):
            bar.set_width(values1[i])  # 막대 너비 설정
            if values1[i] == max_val1:
                bar.set_color('#0EB2F1')  # 가장 큰 값은 다른 색으로 설정
            else:
                bar.set_color('#17FD79')  # 나머지 값은 기본 색으로 설정

        # 두 번째 서브플롯 데이터 업데이트
        values2 = np.random.rand(10)
        max_val2 = np.max(values2)
        for i, bar in enumerate(self.bar2):
            bar.set_width(values2[i])  # 막대 너비 설정
            if values2[i] == max_val2:
                bar.set_color('#0EB2F1')  # 가장 큰 값은 다른 색으로 설정
            else:
                bar.set_color('#17FD79')  # 나머지 값은 기본 색으로 설정

        self.draw()  # 그래프 다시 그리기

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
