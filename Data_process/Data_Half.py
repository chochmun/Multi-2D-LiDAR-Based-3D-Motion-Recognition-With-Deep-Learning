import os
import pandas as pd
import random

def split_csv(input_file, output_file1, output_file2):
    # CSV 파일 불러오기
    df = pd.read_csv(input_file)
    
    # 랜덤하게 데이터를 절반씩 나누기
    num_rows = len(df)
    half_num_rows = num_rows // 2
    indices = list(range(num_rows))
    random.shuffle(indices)
    
    # 첫 번째 파일에 쓸 데이터 선택
    indices_1 = indices[:half_num_rows]
    df1 = df.iloc[indices_1]
    
    # 두 번째 파일에 쓸 데이터 선택
    indices_2 = indices[half_num_rows:]
    df2 = df.iloc[indices_2]
    
    # 데이터를 CSV 파일로 저장
    df1.to_csv(output_file1, index=False)
    df2.to_csv(output_file2, index=False)
raw_train_csv_files = [
    "C:/capstone_data/3/0501/stop/Junmo/jun_all.csv",
    "C:/capstone_data/3/0501/stop/seano/sun_all.csv",
    "C:/capstone_data/3/0501/stop/mini/min_all.csv",
    "C:/capstone_data/3/0501/stop/chungmoon/chung_all.csv"
]

# 출력 디렉토리 설정
output_directory = "C:/capstone_data/3/0501/stop/split_files/"

# 디렉토리가 없다면 생성
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 파일 별로 처리
for file_path in raw_train_csv_files:
    # 파일 이름 추출
    file_name = os.path.basename(file_path)
    
    # 출력 파일 경로 설정
    output_file1 = os.path.join(output_directory, file_name.replace(".csv", "_part1.csv"))
    output_file2 = os.path.join(output_directory, file_name.replace(".csv", "_part2.csv"))
    
    # CSV 파일을 나누고 저장
    split_csv(file_path, output_file1, output_file2)

print("Splitting complete.")