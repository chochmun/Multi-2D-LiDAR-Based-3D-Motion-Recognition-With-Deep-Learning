import numpy as np
from tensorflow.keras.models import load_model

class STEP50CNN:
    def __init__(self, model_path):
        """
        STEP50CNN 클래스 초기화 함수.
        모델 경로를 받아 모델을 로드.
        """
        self.model = self.load_motion_model(model_path)

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

    # 2. 데이터 전처리 함수 (바로 들어온 데이터를 전처리)
    def preprocess_real_time_data(self, new_data):
        """
        실시간 데이터를 모델에 입력할 수 있도록 전처리하는 함수.
        new_data: (1, 270) 형식의 단일 프레임 데이터.
        """
        return np.expand_dims(np.array(new_data), axis=0)  # 배치 차원 추가

    def predict_real_time_motion(self, processed_data):
        """
        전처리된 데이터를 이용해 실시간으로 모션 예측을 수행하는 함수.
        """
        if self.model is None:
            print("모델이 로드되지 않았습니다.")
            return None

        try:
            prediction = self.model.predict(processed_data)  # 모델로 예측 수행
            return prediction
        except Exception as e:
            print(f"실시간 예측 중 오류 발생: {e}")
            return None

    # 실시간 데이터 예측 함수
    def predict(self, new_data):
        """
        실시간으로 데이터를 입력받아 바로 모션 예측을 수행하는 함수.
        """
        processed_data = self.preprocess_real_time_data(new_data)
        predicted_output = self.predict_real_time_motion(processed_data)
        return predicted_output if predicted_output is not None else [[0.998, 0, 0, 0, 0, 0, 0, 0, 0.001]]


class STEP50LSTM:
    def __init__(self, model_path):
        """
        STEP50LSTM 클래스 초기화 함수.
        모델 경로를 받아 모델을 로드.
        """
        self.model = self.load_motion_model(model_path)

    def load_motion_model(self, model_path):
        """
        학습된 LSTM 모델을 불러오는 함수.
        """
        try:
            model = load_model(model_path)
            print(f"모델 {model_path} 로드 성공.")
            return model
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            return None

    def preprocess_real_time_data(self, new_data):
        """
        실시간 데이터를 모델에 입력할 수 있도록 전처리하는 함수.
        new_data: (1, 270) 형식의 단일 프레임 데이터.
        """
        return np.expand_dims(np.array(new_data), axis=0)  # 배치 차원 추가

    def predict_real_time_motion(self, processed_data):
        """
        전처리된 데이터를 이용해 실시간으로 모션 예측을 수행하는 함수.
        """
        if self.model is None:
            print("모델이 로드되지 않았습니다.")
            return None

        try:
            prediction = self.model.predict(processed_data)  # 모델로 예측 수행
            return prediction
        except Exception as e:
            print(f"실시간 예측 중 오류 발생: {e}")
            return None

    # 실시간 데이터 예측 함수
    def predict(self, new_data):
        """
        실시간으로 데이터를 입력받아 바로 모션 예측을 수행하는 함수.
        """
        processed_data = self.preprocess_real_time_data(new_data)
        predicted_motion = self.predict_real_time_motion(processed_data)
        return predicted_motion if predicted_motion is not None else [[0.998, 0, 0, 0, 0, 0, 0, 0, 0, 0.001]]


if __name__ == "__main__":
    # 모델 경로 설정
    model_path = 'estimation_model/motion_cnn_3_5to0_5.h5'

    # STEP50CNN 객체 생성 및 실행
    step50_cnn = STEP50CNN(model_path)

    # 예시 데이터
    new_data = np.random.rand(1, 270)  # 1x270 크기의 임의 데이터

    if step50_cnn.model is not None:
        result = step50_cnn.predict(new_data)
        print(f"예측 결과: {result}")
    else:
        print("모델을 불러오지 못해 프로그램을 종료합니다.")
