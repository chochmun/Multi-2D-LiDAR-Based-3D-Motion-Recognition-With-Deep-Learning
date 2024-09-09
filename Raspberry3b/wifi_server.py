import Ydlidar_Interface
import socket
import struct
import time
import numpy as np
import threading

# 라즈베리 파이의 IPv4 주소와 포트
HOST = '192.168.0.2'
PORT = 12345

# 소켓 생성
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print('서버가 시작되었습니다.')

# 클라이언트 연결을 기다림
client_socket, addr = server_socket.accept()
print('연결됨:', addr)

def receive_data():
    try:
        while True:
            # 클라이언트로부터 데이터 수신
            data = client_socket.recv(1024)
            if data.decode('utf-8').strip().lower() == 'stop':
                print('데이터 전송 중지')
                break
    finally:
        # 연결 종료
        client_socket.close()
        server_socket.close()

# 수신을 위한 스레드 생성
receive_thread = threading.Thread(target=receive_data)
receive_thread.start()

try:
    while True:
        # 0부터 100 사이의 무작위 값을 가지는 배열 생성
        random_array = np.random.randint(0, 101, size=270)
        
        # 배열을 바이트로 변환하여 클라이언트에게 전송
        client_socket.sendall(random_array.tobytes())
        
        # 1초 대기
        time.sleep(1)
except:
    pass
