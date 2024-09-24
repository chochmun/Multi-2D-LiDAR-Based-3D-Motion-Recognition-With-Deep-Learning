import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
import time

class STEP50CNN:
    def __init__(self, model_path):
        """
        STEP50CNN 클래스 초기화 함수.
        모델 경로를 받아 모델을 로드하고, 실시간 시퀀스를 저장할 큐를 초기화.
        """
        self.model = self.load_motion_model(model_path)
        self.sequence_queue = deque(maxlen=50)

    # 1. 모델 로드 함수
    def load_motion_model(self, model_path):
        """
        학습된 CNN 모델을 불러오는 함수.
        """
        try:
            model = load_model(model_path)
            print(f"모델 {model_path} 로드 성공.")
            return model
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            return None

    # 2. 데이터 전처리 함수 (queue에서 가져온 데이터를 전처리)
    def preprocess_real_time_data(self, queue_data):
        """
        실시간 큐 데이터를 모델에 입력할 수 있도록 전처리하는 함수.
        queue_data: (50, 270) 형식의 시퀀스 데이터.
        """
        return np.array(queue_data)

    # 3. 실시간 예측 함수
    def predict_real_time_motion(self, processed_data):
        """
        전처리된 데이터를 이용해 실시간으로 모션 예측을 수행하는 함수.
        model: 불러온 CNN 모델.
        processed_data: 전처리된 시퀀스 데이터.
        """
        try:
            input_data = np.expand_dims(processed_data, axis=0)  # 배치 차원 추가
            prediction = self.model.predict(input_data)
            predicted_label = np.argmax(prediction, axis=1)  # 가장 높은 확률의 라벨 선택
            return predicted_label[0]
        except Exception as e:
            print(f"실시간 예측 중 오류 발생: {e}")
            return None

    # 4. 실시간 데이터 추가 및 예측 수행 함수
    def run(self):
        """
        실시간으로 데이터를 추가하고, 큐가 50개의 시퀀스로 가득 차면 모션 예측을 수행하는 메인 함수.
        """
        while True:
            # 실시간으로 새로운 데이터 입력 (1x270 크기의 데이터가 계속 들어옴)
            new_data = np.random.rand(270)  # 새로운 데이터 (1, 270)

            # 큐에 새로운 데이터 추가
            self.sequence_queue.append(new_data)

            # 큐가 50개의 시퀀스로 가득 찼을 때만 예측 진행
            if len(self.sequence_queue) == 50:
                processed_data = self.preprocess_real_time_data(self.sequence_queue)
                predicted_label = self.predict_real_time_motion(processed_data)

                if predicted_label is not None:
                    print(f"예측된 모션 라벨: {predicted_label}")
                else:
                    print("예측 실패.")
            
            # 다음 입력을 위한 딜레이 (0.05초)
            time.sleep(0.05)

if __name__ == "__main__":
    # 모델 경로 설정
    model_path = 'estimation_model/motion_cnn_3_5to0_5.h5'

    # STEP50CNN 객체 생성 및 실행
    step50_cnn = STEP50CNN(model_path)

    if step50_cnn.model is not None:
        step50_cnn.run()
    else:
        print("모델을 불러오지 못해 프로그램을 종료합니다.")
