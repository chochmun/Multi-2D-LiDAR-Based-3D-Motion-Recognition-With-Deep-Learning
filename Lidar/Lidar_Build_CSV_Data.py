import Ydlidar_Interface
import time
import numpy as np
import csv
from datetime import datetime
import winsound

poses = {
    '0': "사람없는 상태",
    '1': "만세",
    '2': "몸좌로기울기",
    '3': "몸우로기울기",
    '4': "스쿼트",
    '5': "한다리 완전 들기",
    '6': "엎드려",
    '7': "",
    '8': "걷기",
    '9': "양손번갈",
    '10': "왼쨉",
    '11': "오른쨉"
}

print(" ".join(f"{key}.{value}" for key, value in poses.items()))

gesture_num = input("입력할 제스처 번호를 선택하세요: ")
gesture_name = ""
gesture_name = poses[gesture_num]

CSV_name = 'csv/'+gesture_name+'_lidar_data'+ datetime.now().strftime('%m_%d_%H-%M-%S') +'.csv'
port1 = 'COM12' #머리
port2 = 'COM5' #허리
port3 = 'COM7' #다리

max_range=2000
TOTAL_FRAME=2000

lid = ydlidar_x2_driver.YDLidarX2(port1)
lid.connect()
lid.start_scan()
lid2 = ydlidar_x2_driver.YDLidarX2(port2)
lid2.connect()
lid2.start_scan()
lid3 = ydlidar_x2_driver.YDLidarX2(port3)
lid3.connect()
lid3.start_scan()
print("LiDAR started")

def beep_after_5_seconds():
    winsound.Beep(1000, 100)  # 1000Hz 주파수로 1초 동안 비프음 재생
    time.sleep(1)  # 5초 대기
    winsound.Beep(1000, 100)  # 1000Hz 주파수로 1초 동안 비프음 재생
    time.sleep(1)  # 5초 대기
    winsound.Beep(1500, 100)  # 1000Hz 주파수로 1초 동안 비프음 재생

count_t=0
with open(CSV_name, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    beep_after_5_seconds()
    try:
        start_time=time.time()
        
        while True:

            if lid.available:
                distances1 = lid.get_data()
                
                distances2 = lid2.get_data()
                distances3 = lid3.get_data()

                total_dist = np.concatenate((distances1,distances2,distances3))
                
                total_dist= np.insert(total_dist, 0, gesture_num)
                writer.writerow(total_dist)
                count_t+=1

                if(count_t==TOTAL_FRAME): #프레임이 모두끝나면 while 탈출4
                    print(time.time()-start_time)
                    break
                #print(distances1)
                #print(distances2)
                #print(f"========{time.time()-start_time}=======")

            else:
                pass
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass


lid.stop_scan()
lid.disconnect()
lid2.stop_scan()
lid2.disconnect()
lid3.stop_scan()
lid3.disconnect()
print("Done")