import serial
import math
import numpy as np
import time

import warnings
import threading


class YDLidarX2:
    
    def __init__(self, port, chunk_size=1000,angle=90,max_distance=2000): #spot 탐지용은 청크를 1000내려도되는데, 그게 아닌이상 1000이상 유지
        self.__version = 1.03
        self._port = port                # string denoting the serial interface
        self._ser = None
        self._chunk_size = chunk_size    # reasonable range: 1000 ... 10000
        self._min_range = 10			 # 제품 특성상 fixed - minimal measurable distance
        self._max_range = max_distance +1			 # maximal measurable distance
        self._max_data = 20              # maximum number of datapoints per angle
        self._out_of_range = max_distance +1     # indicates invalid data
        self._is_connected = False
        self._is_scanning = False
        self._scan_is_active = False
        self._availability_flag = False
        self._debug_level = 0
        self._error_cnt = 0
        self._lock = threading.Lock()
        self._last_chunk = None
        # 탐지 각도 설정
        self.D_ANGLE= angle
        self.D_start_angle = int(180-(self.D_ANGLE/2)) #90도이면 135도~225도
        self.D_end_angle = int (180+(self.D_ANGLE/2)) #180도면 
        # 2D array capturing the distances for angles from 0 to 359
        self._distances = np.array([[self._out_of_range for _ in range(self._max_data)] for l in range(360)], dtype=np.uint32)
        # 1D array capturing the number of measurements for angles from 0 to 359
        self._distances_pnt = np.array([0 for _ in range(360)], dtype=np.uint32)
        # predefined list of angle corrections for distances from 0 to 8000
        self._corrections = np.array([0.0] + [math.atan(21.8*((155.3-dist)/(155.3*dist)))*(180/math.pi) for dist in range(1, 8001)])
        # measured distances for angles from 0 to 359
        self._result = np.array([self._out_of_range for _ in range(self.D_ANGLE)], dtype=np.int32)#different point
        # operating variables for plot functions
        self._org_x, self._org_y = 0, 0
        self._scale_factor = 0.
        
        
    def connect(self):
        """ Connects on serial interface """
        if not self._is_connected:
            try:
                self._ser = serial.Serial(self._port, 115200, timeout = 1)
                self._is_connected = True
            except Exception as e:
                print(e)
                self._is_connected = False
        else:
            warnings.warn("connect: LiDAR already connected", RuntimeWarning)
        return self._is_connected
    
    
    def disconnect(self):
        """ Disconnects the serial interface """
        if self._is_connected:
            self._ser.close()
            self._is_connected = False
        else:
            warnings.warn("disconnect: LiDAR not connected", RuntimeWarning)
            
            
    def start_scan(self):
        """ Starts a thread to run the scan process. """
        if not self._is_connected:
            warnings.warn("start_scan: LiDAR not connected", RuntimeWarning)
            return False
        self._is_scanning = True
        self._scan_thread = threading.Thread(target = self._scan)
        self._scan_thread.start()
        self._availability_flag = False
        return True
    
    
    def stop_scan(self):
        """ Stops the thread running the scan process. """
        if not self._is_scanning:
            warnings.warn("stop_scan: LiDAR is not scanning", RuntimeWarning)
            return False
        else:
            self._is_scanning = False
            while not self._scan_is_active:
                time.sleep(0.1)
            time.sleep(self._chunk_size / 6000)		# wait for the last chunk to finish reading
        return True
    
    
    def _scan(self):
        """ Core routine to retrieve and decode lidar data.
            Availaility flag is set after each successful decoding process. """
        self._scan_is_active = True
        while self._is_scanning:
            # Retrieve data
            data = self._ser.read(self._chunk_size).split(b"\xaa\x55")
  
            if self._last_chunk is not None:
                data[0] = self._last_chunk + data[0]
            self._last_chunk = data.pop()
            # Clear array for new scan
            distances_pnt = np.array([0 for _ in range(360)], dtype=np.uint32)
            error_cnt = 0
            # Decode data
            for idx, d in enumerate(data):
                # d[0]   = CT(1byte)  Package type
                # d[1]   = LSN(1byte) Sameple quantity
                # d[2:3] = FSA(2byte) Start angle
                # d[4:5] = LSA(2byte) End angle
                # d[6:7] = CS(2byte)  Check code
                # d[8:9] = Si(2byte)  Sample data
                # Reasonable length of the data slice?
                l = len(d)
                if l < 10:  # Typically > 10
                    error_cnt += 1
                    if self._debug_level > 0:
                        print("Idx:", idx, "ignored - len:", len(d))
                    continue
                # Get sample count and start and end angle
                sample_cnt = d[1]
                # Do we have any samples?
                if sample_cnt == 0:
                    error_cnt += 1
                    if self._debug_level > 0:
                        print("Idx:", idx, "ignored - sample_cnt: 0")
                    continue
                # Get start and end angle
                start_angle = ((d[2] + 256 * d[3]) >> 1) / 64
                end_angle = ((d[4] + 256 * d[5]) >> 1) / 64

                # Start data block
                if sample_cnt == 1:
                    dist = round((d[8] + 256 * d[9]) / 4)
                    if self._debug_level > 1:
                        print("Start package: angle:", start_angle, "   dist:", dist)
                    if dist > self._min_range:
                        if dist > self._max_range:
                            dist = self._max_range
                        angle = round(start_angle + self._corrections[dist])
                        if angle < 0:
                            angle += 360
                        if angle >= 360:
                            angle -= 360
                        # Only consider angles between 135 and 225 degrees
                        if self.D_start_angle <= angle <= self.D_end_angle:
                            self._distances[angle][distances_pnt[angle]] = dist
                            if distances_pnt[angle] < self._max_data - 1:
                                distances_pnt[angle] += 1
                            else:
                                if self._debug_level > 0:
                                    print("Idx:", idx, " - pointer overflow")
                                error_cnt += 1

                # Cloud data block
                else:
                    if start_angle == end_angle:
                        if self._debug_level > 0:
                            print("Idx:", idx, "ignored - start angle equals end angle for cloud package")
                        error_cnt += 1
                        continue
                    if l != 8 + 2 * sample_cnt:
                        if self._debug_level > 0:
                            print("Idx:", idx, "ignored - len does not match sample count - len:", l,
                                  " - sample_cnt:", sample_cnt)
                        error_cnt += 1
                        continue
                    if self._debug_level > 1:
                        print("Cloud package: angle:", start_angle, "-", end_angle)
                    if end_angle < start_angle:
                        step_angle = (end_angle + 360 - start_angle) / (sample_cnt - 1)
                    else:
                        step_angle = (end_angle - start_angle) / (sample_cnt - 1)
                    pnt = 8
                    while pnt < l:
                        dist = round((d[pnt] + 256 * d[pnt + 1]) / 4)
                        if dist > self._min_range:
                            if dist > self._max_range:
                                dist = self._max_range
                            angle = round(start_angle + self._corrections[dist])
                            if angle < 0:
                                angle += 360
                            if angle >= 360:
                                angle -= 360
                            # Only consider angles between 135 and 225 degrees
                            if self.D_start_angle <= angle <= self.D_end_angle:
                                self._distances[angle][distances_pnt[angle]] = dist
                                if distances_pnt[angle] < self._max_data - 1:
                                    distances_pnt[angle] += 1
                                else:
                                    if self._debug_level > 0:
                                        print("Idx:", idx, " - pointer overflow")
                                    error_cnt += 1
                        start_angle += step_angle
                        if start_angle >= 360:
                            start_angle -= 360
                        pnt += 2

            # Calculate result
            if self._debug_level > 0 and error_cnt > 0:
                print("Error cnt:", error_cnt)

            for angle in range(self.D_start_angle, self.D_end_angle):
                if distances_pnt[angle] == 0:
                    self._result[angle-self.D_start_angle] = self._out_of_range
                else:
                    self._result[angle-self.D_start_angle] = self._distances[angle][:distances_pnt[angle]].mean()

            self._error_cnt = error_cnt
            self._availability_flag = True
        # End of decoding loop
        self._scan_is_active = False
        
        
    def get_data(self):
        """ 리턴 an array of 거리값 (360 values, one for each degree).
            Resets availability flag"""
        if not self._is_scanning:
            warnings.warn("get_data: Lidar is not scanning", RuntimeWarning)
        self._lock.acquire()
        distances = self._result.copy()
        self._availability_flag = False
        self._lock.release()
        return distances

    def available(self):
        """ Indicates whether a new dataset is available """
        return self._availability_flag
