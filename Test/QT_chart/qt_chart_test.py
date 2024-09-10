import sys
import random
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QTimer
import numpy as np

class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes1 = fig.add_subplot(121)  # 첫 번째 서브플롯 (좌)
        self.axes2 = fig.add_subplot(122)  # 두 번째 서브플롯 (우)
        super(MplCanvas, self).__init__(fig)
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

        # 서브플롯 간 간격 설정
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

class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Create the matplotlib FigureCanvas object
        sc = MplCanvas(self, width=5, height=4, dpi=100)

        # 창 크기 설정
        self.setFixedSize(800, 400)

        # Layout setup
        layout = QVBoxLayout()
        layout.addWidget(sc)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Timer setup for real-time updates
        self.timer = QTimer()
        self.timer.setInterval(500)  # 0.05초마다 업데이트
        self.timer.timeout.connect(sc.update_plot)
        self.timer.start()

app = QApplication(sys.argv)
w = MainWindow()
w.show()
sys.exit(app.exec_())
