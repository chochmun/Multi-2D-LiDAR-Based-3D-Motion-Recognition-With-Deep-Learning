import numpy as np
import pandas as pd

train_csv_files = [
    'C:/capstone_data/3/0501/stop/Junmo/jun_all_part2.csv',  # 1
    'C:/capstone_data/3/0501/stop/seano/sun_all_part2.csv',  # 2
    'C:/capstone_data/3/0501/stop/mini/min_all_part2.csv',   # 3
    'C:/capstone_data/3/0501/stop/chungmoon/chung_all_part2.csv',  # 4
]

# 가중치 배열 초기화
weights = np.zeros(270)

# 초반과 후반에 대한 가중치 설정
weights[0:11] = 0
weights[11:26] = 0.8
weights[26:36] = 1
weights[36:56] = 1
weights[56:66] = 1
weights[66:79] = 0.8
weights[79:89] = 0

weights[90:101] = 0
weights[101:116] = 0.5
weights[116:126] = 0.7
weights[126:146] = 1
weights[146:156] = 0.7
weights[156:169] = 0.5
weights[169:179] = 0

weights[180:191] = 0
weights[192:196] = 0.1
weights[197:201] = 0.5
weights[202:206] = 0.7
weights[207:216] = 0.95
weights[217:236] = 1
weights[237:246] = 0.95
weights[247:251] = 0.7
weights[252:256] = 0.5
weights[257:260] = 0.1
weights[251:269] = 0

def preprocess_data(data):
    # 가중치를 생성합니다. 중앙에 가까울수록 가중치가 높아집니다.
    for i in range(df.shape[0]):
        df.iloc[i, 1:] = df.iloc[i, 1:] * weights
    return data

substring_to_remove = '.csv'
for i in [0,1,2,3]:
    df = pd.read_csv(train_csv_files[i]) 
    # 첫 번째 열을 제외한 나머지 열 선택
    cols = df.columns[1:]
    # 첫 번째 열을 제외한 모든 열에 2001을 빼고 절대값을 취하는 작업 수행
    df[cols] = df[cols].apply(lambda x: abs(x - 2001))

    for j, col in enumerate(df.columns[1:]):  # 첫 번째 열(Label)을 제외한 열에 대해서만 작업
        df[col] = df[col] * weights[j]
    
    # 첫 번째 열을 제외한 나머지 열 선택

    # 수정된 데이터프레임을 CSV 파일로 저장
    file_path=train_csv_files[i]
    modified_file_path = file_path.replace(substring_to_remove, "")
    modified_file_path = modified_file_path + '_filter.csv'  # 실제 파일 경로 입력
    df.to_csv(modified_file_path, index=False)
    print(modified_file_path,"####### - updated")


