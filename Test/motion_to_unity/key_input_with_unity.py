from pynput.keyboard import Key, Controller
import time
import psutil
import os

keyboard = Controller()

# 확장된 key_mapping 딕셔너리
key_mapping = {
    1: Key.up,
    2: Key.down,
    3: Key.left,
    4: Key.right,
    5: 'w',
    6: 'a',
    7: 's',
    8: 'd',
    9: 'r',
    10: 't',
    11: 'y',
    12: 'u',
    13: Key.space
}

# 배열 기반의 시나리오 동작 함수
def execute_scenario(scenario):
    for action in scenario:
        if action in key_mapping:
            key = key_mapping[action]
            if isinstance(key, str):  # 문자열로 주어진 키는 직접 char로 처리
                keyboard.press(key)
                # time.sleep(0)  # 각 키 입력 동작 간 n초 대기
                keyboard.release(key)
            else:  # pynput의 Key 값으로 처리되는 경우
                keyboard.press(key)
                # time.sleep(0)  # 각 키 입력 동작 간 n초 대기
                keyboard.release(key)
            # print(f"{key} 키가 눌렸습니다.")
        # time.sleep(0)  # 요소 간 0.1초 대기

# CPU 및 메모리 사용량 측정 함수
def get_process_usage():
    process = psutil.Process(os.getpid())
    cpu_usage = process.cpu_percent(interval=0.1)  # 1초 동안 CPU 사용량 측정
    memory_info = process.memory_info()

    print(f"CPU 사용량: {cpu_usage}%")
    print(f"메모리 사용량: {memory_info.rss / (1024 * 1024)} MB")  # MB 단위로 출력wasdrtyu wasdrtyu wasdrtyu 

# 임의의 시나리오 작성 (예시)
example_scenario = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# 메인 함수
if __name__ == "__main__":
    print(len(example_scenario))
    time.sleep(3)

    # 시나리오 실행 전 CPU 및 메모리 사용량 측정
    print("시나리오 실행 전 자원 사용량:")
    get_process_usage()

    start_time = time.time()

    # 시나리오 실행
    execute_scenario(example_scenario)

    elapsed_time = time.time() - start_time
    print('경과 시간 : ', elapsed_time)

    # 시나리오 실행 후 CPU 및 메모리 사용량 측정
    print("시나리오 실행 후 자원 사용량:")
    get_process_usage()
