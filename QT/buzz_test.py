import winsound
import time

def beep_duration(duration):
    freq = 1050  # 주파수는 1500Hz로 고정
    for _ in range(duration):
        winsound.Beep(freq, 700)  # 1000ms = 1초 동안 소리 발생
        time.sleep(0.3)  # 소리 사이에 약간의 지연을 추가 (필요시 제거 가능)

# 사용 예제
beep_duration(3)  # 2초 동안 부저음 발생