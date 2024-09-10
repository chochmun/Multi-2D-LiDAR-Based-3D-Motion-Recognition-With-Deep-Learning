import os
import time
import numpy as np
import json
import csv
import serial.tools.list_ports

import threading

import sys
from . import ydlidar_driver
#import ydlidar_driver

class MultiLidardriver:
    def __init__(self, angle=90,max_distance=2000, ports_choice=[1, 2, 3], FPS=20):
        self.max_distance = max_distance
        self.angle=angle
        self.lidar_object_list = [] #라이다 객체 모음 : 리스트 포트에 따라 3개의 객체가 들어갈예정
        self.ports=self.detect_lidar_ports()
        
        self.ports_selected = self.choose_ports(ports_choice)
        self.get_data_delay= 1 / FPS
        

    def detect_lidar_ports(self):
        tmp_ports = serial.tools.list_ports.comports()
        tmp_ports=[port.device for port in tmp_ports]
        return tmp_ports

    def validate_ports_choice(self,ports_choice):
    # 각 요소가 정수인지 확인
        if not all(isinstance(port, int) for port in ports_choice):
            raise ValueError("모든 요소는 정수여야 합니다.")

        # 각 요소가 0보다 크고 4보다 작은지 확인
        if not all(0 < port < 4 for port in ports_choice):
            raise ValueError("포트는 1,2,3중 하나이어야 합니다.")

        # 요소들이 서로 다른지 확인
        if len(ports_choice) != len(set(ports_choice)):
            raise ValueError("모든 요소의 값이 서로 달라야 합니다.")
    
    def choose_ports(self,ports_choice):
        try:
            self.validate_ports_choice(ports_choice)
            tmp= [self.ports[i-1] for i in ports_choice]
            print("ports_choice가 유효합니다.")
            return tmp
        except ValueError as e:
            print(f"에러: {e}")
        

    def setup_lidars(self):#Multi_Lidars_re_setup
        self.lidar_object_list=[]
        for port in self.ports_selected:
            lidar = ydlidar_driver.YDLidarX2(port, angle=self.angle, max_distance=self.max_distance)
            self.lidar_object_list.append(lidar)
        print(f"{len(self.ports_selected)} LiDARs setuped")

    def start_lidars(self):
        for lidar in self.lidar_object_list:
            lidar.connect()
            lidar.start_scan()
        print(f"{len(self.ports_selected)} LiDARs Scan started")
            
    def stop_lidars(self):#Multi_lidars_OFF
        for lidar in self.lidar_object_list:
            lidar.stop_scan()
            lidar.disconnect()
        print("LiDARs Scan stopped")

    def get_distances(self):
        
        if self.lidar_object_list[0].available:
            
            dist_top= self.lidar_object_list[0].get_data()
            dist_mid= self.lidar_object_list[1].get_data()
            dist_bot= self.lidar_object_list[2].get_data()

        time.sleep(self.get_data_delay)#이것때문에 모터프리퀀시가 10이어도, 20프레임으로나옴
        return dist_top,dist_mid,dist_bot


    




if __name__ == "__main__":
    driver = MultiLidardriver(ports_choice=[1, 3, 2], angle=180, max_distance=2000, FPS=2)
    ports=driver.detect_lidar_ports()
    print(ports)
    print(driver.ports_selected)
    driver.setup_lidars()

    distances = []
      # 시작 시간을 기록
    for _ in range(0,3):
        start_time = time.time()
        driver.start_lidars()

        while time.time() - start_time < 5:  # 10초 동안 반복
            distances = driver.get_distances()  # 거리 값을 얻음
            print(distances)
            time.sleep(0.05)  # 0.05초 대기
        driver.stop_lidars()
        print("잠시 라이다를 2초간 멈추겠어요")
        time.sleep(2)


