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