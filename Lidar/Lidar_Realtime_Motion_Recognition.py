import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import pyautogui
import Ydlidar_interface as ydlidar

# 모델 정의
class AdvancedRNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(AdvancedRNNModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # 마지막 타임 스텝의 출력만 사용
        return x

# 실시간 데이터 처리 및 모델 입력 준비
def process_data(data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data.reshape(1, -1))
    return torch.tensor(data, dtype=torch.float32)

# 키 입력 함수
def press_key(label):
    key_mapping = {1: 'w', 2: 'a', 3: 's', 4: 'd'}
    if label in key_mapping:
        pyautogui.press(key_mapping[label])
        print(f"Pressed {key_mapping[label]} for label {label}")

# PORT 설정
port1 = 'COM5' #머리
port2 = 'COM11' #허리
port3 = 'COM12' #다리

# LiDAR 설정
lid = ydlidar.YDLidarX2(port1)
lid.connect()
lid.start_scan()
lid2 = ydlidar.YDLidarX2(port2)
lid2.connect()
lid2.start_scan()
lid3 = ydlidar.YDLidarX2(port3)
lid3.connect()
lid3.start_scan()
print("LiDAR started")

# 모델 로드
model = AdvancedRNNModel(input_dim=270, hidden_dim=128, num_layers=2, output_dim=5)  # 예상되는 입력 차원
model.load_state_dict(torch.load('saved_model\\cnn_all_model.pth'))
model.eval()

try:
    while True:
        if lid.available:
            distances1 = lid.get_data()
            distances2 = lid2.get_data()
            distances3 = lid3.get_data()
            total_dist = np.concatenate((distances1,distances2,distances3))
            total_dist = total_dist.unsqueeze(0)  # 배치 차원 추가
            
            # 모델 예측
            with torch.no_grad():
                outputs = model(total_dist)
                _, predicted = torch.max(outputs, 1)
                press_key(predicted.item())
except KeyboardInterrupt:
    print("Stopping")

lid.stop_scan()
lid.disconnect()
lid2.stop_scan()
lid2.disconnect()
lid3.stop_scan()
lid3.disconnect()
print("Done")