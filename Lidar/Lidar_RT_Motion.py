import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import yaml
import Ydlidar_Interface as ydlidar
from PIL import Image
import matplotlib.pyplot as plt
import serial.tools.list_ports
import time
model_path='Motion_Recognition_Model\Saved_Models\cnn_ex3_5_munc_32000aug_model_15.pth'

D_ANGLE= 90
D_start_angle=int(180-(90-D_ANGLE/2))
D_end_angle= int(90-D_ANGLE/2)

def list_serial_ports():
    ports = serial.tools.list_ports.comports()
    port_names = []

    for port in ports:
        port_names.append(port.device)
    
    return port_names
port_names = list_serial_ports()
print(port_names)

port1 = port_names[1]
port2 = port_names[0]
port3 = port_names[2]

# 모델 로드
with open('Motion_Recognition_Model/parameters.yaml', 'r', encoding='utf-8') as file:
    yaml_data = yaml.safe_load(file)
    #num_classes = yaml_data['model']['num_classes']

# 모델 정의
class LidarCNN(nn.Module):
    def __init__(self, num_classes):
        super(LidarCNN, self).__init__()
        self.conv1 = nn.Conv1d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 22, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x.view(-1, 32 * 22))
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

# 실시간 데이터 처리 및 모델 입력 준비
def process_data(data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data.reshape(1, -1))
    return torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # 배치 차원 추가

# 이미지 표시 함수
def show_image(label):
    image_mapping = {
        1: 'jpgs/슬라이드1.jpg',
        2: 'jpgs/슬라이드2.jpg',
        3: 'jpgs/슬라이드3.jpg',
        4: 'jpgs/슬라이드4.jpg',
        5: 'jpgs/슬라이드5.jpg'
    }
    if label in image_mapping:
        img_path = image_mapping[label]
        img = Image.open(img_path)
        plt.imshow(img)
        plt.axis('off')
        plt.draw()
        plt.pause(0.001)  # 갱신 속도 조정
        print(f"Displayed image for label {label}")

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


model = LidarCNN(num_classes=5)
model.load_state_dict(torch.load(model_path))
model.eval()

plt.ion()  # 인터랙티브 모드 켜기
time.sleep(2)
try:
    while True:
        if lid.available:
            distances1 = lid.get_data()
            distances2 = lid2.get_data()
            distances3 = lid3.get_data()
            total_dist = np.concatenate((distances1, distances2, distances3))
            print(total_dist)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(total_dist.reshape(-1, 1)).reshape(-1, 3, 90)

            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            # 모델 예측
            input_data = process_data(X_tensor)
            with torch.no_grad():
                outputs = model(input_data)
                _, predicted = torch.max(outputs, 1)
                show_image(predicted.item())
except KeyboardInterrupt:
    print("Stopping")

plt.ioff()  # 인터랙티브 모드 끄기
plt.close()  # 모든 열린 창 닫기

lid.stop_scan()
lid.disconnect()
lid2.stop_scan()
lid2.disconnect()
lid3.stop_scan()
lid3.disconnect()
print("Done")
