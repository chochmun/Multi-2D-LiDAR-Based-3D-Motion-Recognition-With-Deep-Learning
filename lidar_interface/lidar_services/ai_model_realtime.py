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

    def predict_real_time_motion(self, processed_data):
        """
        전처리된 데이터를 이용해 실시간으로 모션 예측을 수행하는 함수.
        """
        if self.model is None:
            print("모델이 로드되지 않았습니다.")
            return None

        try:
            input_data = np.expand_dims(processed_data, axis=0)  # 배치 차원 추가
            prediction = self.model.predict(input_data)  # 모델로 예측 수행

            # 예측 결과의 합 계산
            #prediction_sum = np.sum(prediction)

            # 각 예측 값을 합으로 나눠 비율로 변환
            #ratio_output = np.round((prediction / prediction_sum) * 100, 2)

            return prediction
        except Exception as e:
            print(f"실시간 예측 중 오류 발생: {e}")
            return None

    # 4. 실시간 데이터 추가 및 예측 수행 함수
    def predict(self,new_data):
        """
        실시간으로 데이터를 추가하고, 큐가 50개의 시퀀스로 가득 차면 모션 예측을 수행하는 메인 함수.
        """
        # 큐에 새로운 데이터 추가
        self.sequence_queue.append(new_data)
        

        # 큐가 50개의 시퀀스로 가득 찼을 때만 예측 진행
        if len(self.sequence_queue) == 50:
            processed_data = self.preprocess_real_time_data(self.sequence_queue)
            predicted_output = self.predict_real_time_motion(processed_data)
            #predicted_label
            return predicted_output
        else:
            return [[0.998,0,0, 0,0,0, 0,0,0,0.001]]
            #if predicted_label is not None:
            #    print(f"예측된 모션 라벨: {predicted_label}")
            #else:
            #    print("예측 실패.")
            
            
            

if __name__ == "__main__":
    # 모델 경로 설정
    model_path = 'estimation_model/motion_cnn_3_5to0_5.h5'

    # STEP50CNN 객체 생성 및 실행
    step50_cnn = STEP50CNN(model_path)

    if step50_cnn.model is not None:
        step50_cnn.predict(new_data)
    else:
        print("모델을 불러오지 못해 프로그램을 종료합니다.")
