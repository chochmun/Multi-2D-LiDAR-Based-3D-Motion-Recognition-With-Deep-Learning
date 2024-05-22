#data_증강.py
import pandas as pd
import numpy as np

train_csv_files = [
    'C:/capstone_data/3/0501/stop/Junmo/jun_all_part1_filter.csv',  # 1
    'C:/capstone_data/3/0501/stop/seano/sun_all_part1_filter.csv',  # 2
    'C:/capstone_data/3/0501/stop/mini/min_all_part1_filter.csv',   # 3
    'C:/capstone_data/3/0501/stop/chungmoon/chung_all_part1_filter.csv',  # 4
]
def raw_data(data):
    # 데이터 읽기
    raw_data = data.copy()
    return raw_data
def mirror_data(data):
    
    columns = data.columns
    first_column = data[columns[0]]  # 첫 번째 열 저장
    other_columns = data.iloc[:, 1:]  # 첫 번째 열을 제외한 나머지 열
    # 대칭 마스크 생성
    symmetric_mask = first_column.isin([1, 4, 5])
    # 각 부분의 열을 반전
    part1_reversed = other_columns.iloc[:, 0:90][symmetric_mask].iloc[:, ::-1]
    part2_reversed = other_columns.iloc[:, 90:180][symmetric_mask].iloc[:, ::-1]
    part3_reversed = other_columns.iloc[:, 180:270][symmetric_mask].iloc[:, ::-1]
    
    # 반전된 부분을 결합
    symmetric_data = pd.concat([part1_reversed, part2_reversed, part3_reversed], axis=1)
    # 열 인덱스를 원래 순서대로 재배열
    symmetric_data.columns = other_columns.columns
    symmetric_data.insert(0, columns[0], first_column[symmetric_mask])

    return symmetric_data


    # 2.big_ Noise Addition
def big_noise_data(data):
    columns = data.columns
    first_column = data[columns[0]]  # 첫 번째 열 저장
    other_columns = data.iloc[:, 1:]  # 첫 번째 열을 제외한 나머지 열

    noise = np.random.normal(0, 10, (other_columns.shape[0], 1))
    noisy_data = other_columns + noise
    noisy_data.insert(0, columns[0], first_column)
    return noisy_data

    # 3. Scaling
def scale_data(data):
    columns = data.columns
    first_column = data[columns[0]]  # 첫 번째 열 저장
    other_columns = data.iloc[:, 1:]  # 첫 번째 열을 제외한 나머지 열
    
    scale_factors = np.random.normal(1.0, 0.05, (other_columns.shape[0], 1))  # Slight scale around 1
    scaled_data = other_columns * scale_factors
    scaled_data.insert(0, columns[0], first_column)

    return scaled_data

    #5. SHIFT
def shif_data(data):
    #print(type(data))
    columns = data.columns
    first_column = data[columns[0]]  # 첫 번째 열 저장
    shift_values = [-10,-9,-8,-7,-6,-5,5,6,7,8,9,10]

    # 첫 번째 열을 제외하고 각 행을 랜덤한 수만큼 시프트
    shifted_data = data.iloc[:, 1:].apply(lambda row: row.shift(np.random.choice(shift_values)).fillna(0), axis=1)
    # 첫 번째 열을 다시 결합
    shifted_data.insert(0, data.columns[0], first_column)
    return shifted_data

# 함수 호출
for i in [0,1,2,3]:  # 0:준 1:선 2:민 3:청
    file_path = train_csv_files[i]
    data = pd.read_csv(file_path)

    substring_to_remove = '.csv'
    modified_file_path = file_path.replace(substring_to_remove, "")
    modified_file_path = modified_file_path + '_Aug_32000.csv'  # 실제 파일 경로 입력
    
    raw=raw_data(data)
    mirror =mirror_data(data)
    
    raw_scale=scale_data(raw)
    mirror_scale=mirror_data(mirror)

    raw_shift=shif_data(raw)
    mirror_shift=shif_data(mirror)

    raw_scale_shift=shif_data(raw_scale)
    mirror_scale_shift=shif_data(mirror_scale)

    raw_noise_scale=big_noise_data(raw_scale)
    mirror_noise_scale=big_noise_data(mirror_scale)
 
    raw_noise=big_noise_data(raw)
    mirror_noise=big_noise_data(mirror)

    raw_noise_shift=big_noise_data(raw_shift)
    mirror_noise_shift=big_noise_data(mirror_shift)

    raw_noise_shift_scale=big_noise_data(raw_scale_shift)
    mirror_noise_shift_scale=big_noise_data(mirror_scale_shift)
    
#증강량 알아서 조절하기==============================합치기===============합치기=================CONCAT===========
#참고로 MIRROR은 왼쪽기울기 우로기울기 제외시켜서, 2500개중 1500개만 증강시킴
#그래서 2500+1500 =4000 >>>> 4000 X 8 = 32000
    augmented_data = pd.concat([raw, mirror, raw_scale, mirror_scale,
                                raw_shift, mirror_shift,raw_scale_shift,mirror_scale_shift,
                                raw_noise,mirror_noise,raw_noise,mirror_noise,
                                raw_noise_shift,mirror_noise_shift,
                                raw_noise_shift_scale,mirror_noise_shift_scale,])
    #augmented_data = pd.concat([raw, mirror, 
    #                            raw_shift, mirror_shift])
    # 결과를 새 CSV 파일로 저장
    #print(augmented_data)
    #augmented_data.to_csv(modified_file_path, index=False)

    augmented_data.to_csv(modified_file_path,index=False)
    #output_file = augment_data(train_csv_files[i], modified_file_path)
    print(f'Augmented data saved to {modified_file_path}')