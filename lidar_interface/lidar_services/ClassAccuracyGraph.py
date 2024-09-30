import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


#디버그시 아래거 사용
#from lidar_interface.ui_py.ConnectUnityWindow import  Ui_ConnectUnityWindow
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=7, height=2.5, dpi=100,angle=90):  # 크기를 700x250으로 설정 (inch로 설정: 7x2.5)
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.angle=angle
        self.class_num=10
        if self.angle>91: 
            self.axes1 = fig.add_subplot(121)  # 첫 번째 서브플롯 (좌)
            self.axes2 = fig.add_subplot(122)  # 두 번째 서브플롯 (우)
        elif angle==90:
            self.axes1 = fig.add_subplot(111)  # 첫 번째 서브플롯 (좌)

        super(MplCanvas, self).__init__(fig)
        self.setFixedSize(700, 250)  # 실제 캔버스의 고정 크기 설정
        self.init_plot(angle)

        self.scenario = [1,1,1,5,1,1,3,1,5,6,1,1,1,1,7]  # 시나리오 배열
        self.scenario_index = 0  # 현재 시나리오 인덱스
        self.motion_dict = {
            0: [0.9957, 0.2752, 0.1518, 0.0477, 0.4664, 0.2625, 0.2588, 0.3517, 0.3893, 0.3076],
            1: [0.0939, 0.6405, 0.0718, 0.0716, 0.2727, 0.3014, 0.1056, 0.0876, 0.0564, 0.3165],
            2: [0.1879, 0.4452, 0.8258, 0.1753, 0.0308, 0.1528, 0.1679, 0.1422, 0.2114, 0.0192],
            3: [0.2827, 0.2761, 0.0722, 0.6208, 0.0669, 0.2421, 0.0842, 0.0179, 0.1143, 0.1083],
            4: [0.0985, 0.2149, 0.1809, 0.2698, 0.7102, 0.1022, 0.0658, 0.0938, 0.2411, 0.2409],
            5: [0.2144, 0.0613, 0.2315, 0.1006, 0.0919, 0.6934, 0.0704, 0.2657, 0.1995, 0.3304],
            6: [0.2195, 0.1894, 0.0163, 0.0974, 0.2282, 0.0818, 0.7262, 0.0224, 0.0977, 0.2615],
            7: [0.0389, 0.2531, 0.0418, 0.2956, 0.0403, 0.0903, 0.2668, 0.9866, 0.1065, 0.0889],
            8: [0.2677, 0.1999, 0.0304, 0.0682, 0.0914, 0.1541, 0.0696, 0.3115, 0.9628, 0.0134],
            9: [0.1999, 0.0369, 0.2365, 0.1621, 0.2667, 0.1518, 0.0857, 0.0574, 0.1373, 0.9284]
        }

    def init_plot(self,angle):
        labels = ["stand", "none", "left_hand", "right_hand", "wave", "squat", "plank", "too_close", "jump", "walk"]

        # 첫 번째 서브플롯 초기화
        self.bar1 = self.axes1.barh(np.arange(9), np.zeros(9), color='#17FD79', height=0.8)
        self.axes1.set_xlim(0, 1)
        self.axes1.set_yticks(np.arange(10))
        self.axes1.set_yticklabels(labels)  # Y축 레이블을 'a' ~ 'j'로 설정
        self.axes1.set_xticks(np.arange(0, 1.1, 0.2))  # X축 눈금 0.1 단위로 설정
        if self.angle>91:
            # 두 번째 서브플롯 초기화
            self.bar2 = self.axes2.barh(np.arange(9), np.zeros(9), color='#17FD79', height=0.8)
            self.axes2.set_xlim(0, 1)
            self.axes2.set_yticks(np.arange(10))
            self.axes2.set_yticklabels(labels)  # Y축 레이블을 'a' ~ 'j'로 설정
            self.axes2.set_xticks(np.arange(0, 1.1, 0.2))  # X축 눈금 0.1 단위로 설정

        # 서브플롯 간 간격 설정
        self.figure.tight_layout()
  

    def update_plot_random(self):
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
        if self.angle>91:
            values2 = np.random.rand(10)
            max_val2 = np.max(values2)
            for i, bar in enumerate(self.bar2):
                bar.set_width(values2[i])  # 막대 너비 설정
                if values2[i] == max_val2:
                    bar.set_color('#0EB2F1')  # 가장 큰 값은 다른 색으로 설정
                else:
                    bar.set_color('#17FD79')  # 나머지 값은 기본 색으로 설정

        self.draw()  # 그래프 다시 그리기
    def update_plot_scenario(self):
        # 시나리오 인덱스에 맞는 motion_dict 값을 사용
        current_key = self.scenario[self.scenario_index]
        values1 = np.zeros(10)  # 10개의 값을 가진 배열 초기화

        # motion_dict에서 현재 시나리오의 키 값을 사용하여 평균을 계산
        values1[:len(self.motion_dict[current_key])] = self.motion_dict[current_key]

        max_val1 = np.max(values1)
        for i, bar in enumerate(self.bar1):
            bar.set_width(values1[i])
            if values1[i] == max_val1:
                idx_max_val1=i
                bar.set_color('#0EB2F1')
            else:
                bar.set_color('#17FD79')

        # 두 번째 서브플롯 업데이트 (필요 시)
        idx_max_val2=None
        if self.angle > 91:
            values2 = np.zeros(10)
            values2[:len(self.motion_dict[current_key])] = self.motion_dict[current_key]

            max_val2 = np.max(values2)
            for i, bar in enumerate(self.bar2):
                bar.set_width(values2[i])
                if values2[i] == max_val2:
                    idx_max_val2=i
                    bar.set_color('#0EB2F1')
                else:
                    bar.set_color('#17FD79')

        # 시나리오 인덱스를 다음으로 넘어감
        self.scenario_index = (self.scenario_index + 1) % len(self.scenario)

        self.draw()  # 그래프 다시 그리기
        return [idx_max_val1,idx_max_val2]
    
    def update_plot_by_realtime_motion(self,data1,data2):
        
        max_val1 = np.max(data1)
        for i, bar in enumerate(self.bar1):
            bar.set_width(data1[i])
            if data1[i] == max_val1:
                idx_max_val1=i
                bar.set_color('#0EB2F1')
            else:
                bar.set_color('#17FD79')

        # 두 번째 서브플롯 업데이트 (필요 시)
        idx_max_val2=None
        if self.angle > 91:

            max_val2 = np.max(data2)
            for i, bar in enumerate(self.bar2):
                bar.set_width(data2[i])
                if data2[i] == max_val2:
                    idx_max_val2=i
                    bar.set_color('#0EB2F1')
                else:
                    bar.set_color('#17FD79')

        # 시나리오 인덱스를 다음으로 넘어감
        self.scenario_index = (self.scenario_index + 1) % len(self.scenario)

        self.draw()  # 그래프 다시 그리기
        return [idx_max_val1,idx_max_val2]