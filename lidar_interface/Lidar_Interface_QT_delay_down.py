import sys
import os
import json
import time
import threading
import winsound
import random
import numpy as np
from PyQt5 import QtWidgets,QtCore
from ui_py.MainWindow import Ui_MainWindow
from ui_py.SettingWindow import Ui_SettingWindow
from ui_py.DataViewWindow import Ui_DataViewWindow
from ui_py.EnvSetWindow import Ui_EnvSetWindow
from ui_py.TransferLearnWindow import  Ui_TransferLearnWindow
from ui_py.SkeletonViewWindow import  Ui_SkeletonViewWindow
from ui_py.SaveCsvWindow import Ui_SaveCsvWindow
from ui_py.ConnectUnityWindow import  Ui_ConnectUnityWindow

from PyQt5.QtWidgets import QApplication, QMessageBox,QVBoxLayout

from lidar_services.ClassAccuracyGraph import MplCanvas
from lidar_services.multi_lidar_services import MultiLidarServices

from datetime import datetime
import csv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pynput.keyboard import Key, Controller
from matplotlib.animation import FuncAnimation

class MainApp(QtWidgets.QMainWindow):
    
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.folder_name_qt= "C:\capstone_data\Multi-2D-LiDAR-Based-3D-Motion-Recognition-With-Deep-Learning\lidar_interface\\"
        #이전 사용자의 세팅값을 기억
        self.SETTINGS_FILE= self.folder_name_qt +"saved_settings.json"
        self.settings_loaded = False
        if os.path.exists(self.SETTINGS_FILE):
            self.load_settings()
            self.settings_loaded = True
        else:
            self.default_initialization()
        
        self.multi_lidar_services=None
        
        self.envpath=self.folder_name_qt+"env_jsons"
        self.key_mapping_play1,self.key_mapping_play2=self.load_key_mapping()
        
        # LiDAR 서비스와 캔버스를 비동기 작업 처리 클래스로 넘겨줌
        
        
        print(self.key_mapping_play1)
        print(self.key_mapping_play2)
        # 각 윈도우 객체 생성
        self.setting_window = QtWidgets.QWidget()
        self.setting_ui = Ui_SettingWindow()
        self.setting_ui.setupUi(self.setting_window)

        self.data_view_window = QtWidgets.QWidget()
        self.data_view_ui = Ui_DataViewWindow()
        self.data_view_ui.setupUi(self.data_view_window)

        self.env_set_window = QtWidgets.QWidget()
        self.env_set_ui = Ui_EnvSetWindow()
        self.env_set_ui.setupUi(self.env_set_window)

        self.transfer_learn_window = QtWidgets.QWidget()
        self.transfer_learn_ui = Ui_TransferLearnWindow()
        self.transfer_learn_ui.setupUi(self.transfer_learn_window)

        self.skeleton_view_window = QtWidgets.QWidget()
        self.skeleton_view_ui = Ui_SkeletonViewWindow()
        self.skeleton_view_ui.setupUi(self.skeleton_view_window)

        self.save_csv_window = QtWidgets.QWidget()
        self.save_csv_ui = Ui_SaveCsvWindow()
        self.save_csv_ui.setupUi(self.save_csv_window)

        self.connect_unity_window = QtWidgets.QWidget()
        self.connect_unity_ui = Ui_ConnectUnityWindow()
        self.connect_unity_ui.setupUi(self.connect_unity_window)
        
        self.connect_unity_window = QtWidgets.QWidget()
        self.connect_unity_ui = Ui_ConnectUnityWindow()
        self.connect_unity_ui.setupUi(self.connect_unity_window)
        
        # ConnectUnityApp 추가: MplCanvas를 통해 그래프를 추가
        self.layout_widget = self.connect_unity_ui.Button_Connect.parent()
        self.mplcanvas = MplCanvas(self.layout_widget, width=5, height=4, dpi=100,angle=int(self.selected_angle))
        self.connect_unity_ui.verticalLayout.addWidget(self.mplcanvas)
        #self.connect_unity_app = ConnectUnityApp()

        
        
        #위에서 객체생성한 후에 ,불러온세팅값으로 업데이트
        if self.settings_loaded:
                    self.update_ui_from_settings()
        
        # 필드 변수 초기화
        self.usr_click_count=0 #동적데이터 저장시에 버튼클릭 세주는것
        self.key_input_approval = False #유니티 키입력 , Connection누르면 활성화되고 stop누르면 비활성화
        self.is_running = False  # 라이다 구동함수 실행여부
        self.elapsed_time = 0  # 경과 시간 저장
        self.timer_thread = None  # 타이머 스레드
        self.count_thread = None  # 카운트 스레드
        self.frames=int(self.save_csv_ui.Input_frames.text())
        self.name=self.save_csv_ui.Input_name.text()
        self.selected_pose=self.save_csv_ui.FuncLabel_count.text()

        
        
        self.max_dist = int(self.setting_ui.Input_maxdist.text())
        self.buzz_duration = int(self.setting_ui.Input_buzzduration.text())
        self.selected_angle = 90 if self.setting_ui.radioButton_90.isChecked() else 180
        self.selected_env = self.setting_ui.FuncLabel_selected_env.text()
        self.selected_model = self.setting_ui.FuncLabel_selected_model.text()
        self.ports_choice[0] = int(self.setting_ui.Input_port_top.text())
        self.ports_choice[1] = int(self.setting_ui.Input_port_mid.text())
        self.ports_choice[2] = int(self.setting_ui.Input_port_bot.text())
        print(f"Settings default2: MaxDist={self.max_dist}, BuzzDuration={self.buzz_duration}, Angle={self.selected_angle}, Env={self.selected_env}, Model={self.selected_model}, ports={self.ports_choice}")

        #Environment set 필드 초기화
        self.new_env_name=self.env_set_ui.Input_json.text()
        self.new_env_loading_time=int(self.env_set_ui.Input_loadingtime.text())
        self.new_env_margin_dist=int(self.env_set_ui.Input_margindist.text())
        # 필드 초기화
        self.setting_ui.Input_buzzduration.setText(str(self.buzz_duration))

        # 프로그램 시작시 settings 불러오기
        #self.load_settings()

        self.load_env_file_list()
        self.load_model_files()
        self.load_pose_files()

        self.connect_ui_buttons()
        #Start 버튼 초기 활성화 /Stop 버튼 초기 비활성화
        self.set_buttons_by_connection()
        



    def set_buttons_initial_state(self):
        # Stop 버튼 비활성화 및 Start 버튼 활성화
        self.data_view_ui.Button_stop.setEnabled(False)
        self.transfer_learn_ui.Button_stop.setEnabled(False)
        self.save_csv_ui.Button_stop.setEnabled(False)
        self.connect_unity_ui.Button_stop.setEnabled(False)
        self.skeleton_view_ui.Button_stop.setEnabled(False)

        self.data_view_ui.Button_start.setEnabled(True)
        self.skeleton_view_ui.Button_start.setEnabled(True)
        self.transfer_learn_ui.Button_start.setEnabled(True)
        self.save_csv_ui.Button_start.setEnabled(True)
        self.connect_unity_ui.Button_start.setEnabled(True)

        # Count for Sequence 버튼 비활성화
        self.save_csv_ui.Button_countforsequence.setEnabled(False)

    def load_env_file_list(self):
        self.setting_ui.List_env.clear()
        if os.path.exists(self.envpath):
            env_files = os.listdir(self.envpath)
            self.setting_ui.List_env.addItems(env_files)

    def load_model_files(self):
        model_files_path = self.folder_name_qt+"models"
        self.setting_ui.List_model.clear()
        if os.path.exists(model_files_path):
            model_files = os.listdir(model_files_path)
            self.setting_ui.List_model.addItems(model_files)

    def load_pose_files(self):
        pose_file_path = "lidar_interface/poses.json"
        self.save_csv_ui.List_pose.clear()
        if os.path.exists(pose_file_path):
            with open(pose_file_path, 'r', encoding='utf-8') as f:
                poses = json.load(f)
                for key, value in poses.items():
                    display_text = f"{key}: {value}"
                    self.save_csv_ui.List_pose.addItem(display_text)
                    self.transfer_learn_ui.List_pose.addItem(display_text)

    def load_key_mapping(self):
        return {
            #0: Stand motion -> None press
            0: [Key.up,None],#차렷자세 -> 전진
            -1: [Key.down,None], #(미구현) 
            2: [Key.left,Key.up], #왼손 들기 -> 좌 이동
            3: [Key.right,Key.up],#오른손 들기 -> 우 이동
            8: [Key.space,Key.up], #점프  -> 점프ㅇ
            4: ['t',None],#만세 - 상호작용1
            5: ['y',None],#스쿼트 -상호작용2
            6: ['u',None],#플랭크 - 상호작용3 
            7: ['p',None],#너무 가까움 -Pause
            1:['p',None]#사람없음 -Pause
        }, {
            #0: Stand motion -> None press
            1: 'w',   2: 'a',    3: 's', 4: 'd',
            5: 'r',
            6: 't',7: 'y',8: 'u',  
            9: 'p'    
        }
        
        # Return the loaded key mappings
        return key_mappings["player1"], key_mappings["player2"]

    def select_env(self, item):
        self.setting_ui.FuncLabel_selected_env.setText(item.text())

    def select_model(self, item):
        self.setting_ui.FuncLabel_selected_model.setText(item.text())
    def select_pose(self, item):
        self.save_csv_ui.FunLabel_selectedpose.setText(item.text())
    def show_setting_window(self):
        self.setting_window.show()
        self.close()

    def show_data_view_window(self):
        self.data_view_window.show()
        self.close()

    def show_skeleton_view_window(self):
        self.skeleton_view_window.show()
        self.close()

    def show_env_set_window(self):
        self.env_set_window.show()
        self.close()

    def show_transfer_learn_window(self):
        self.transfer_learn_window.show()
        self.close()

    def show_save_csv_window(self):
        self.save_csv_window.show()
        self.close()

    def show_connect_unity_window(self):
        self.connect_unity_window.show()
        self.close()

    def show_main_window(self):
        self.show()
        self.setting_window.close()
        self.data_view_window.close()
        self.env_set_window.close()
        self.skeleton_view_window.close()
        self.transfer_learn_window.close()
        self.save_csv_window.close()
        self.connect_unity_window.close()

    def count_for_sequence(self):
        
        self.usr_click_count += 1
        self.save_csv_ui.FuncLabel_count.setText(str(self.usr_click_count))


    def save_setting(self):
        self.max_dist = int(self.setting_ui.Input_maxdist.text())
        self.buzz_duration = int(self.setting_ui.Input_buzzduration.text())
        self.selected_angle = 90 if self.setting_ui.radioButton_90.isChecked() else 180
        self.selected_env = self.setting_ui.FuncLabel_selected_env.text()
        self.selected_model = self.setting_ui.FuncLabel_selected_model.text()
        self.ports_choice[0] = int(self.setting_ui.Input_port_top.text())
        self.ports_choice[1] = int(self.setting_ui.Input_port_mid.text())
        self.ports_choice[2] = int(self.setting_ui.Input_port_bot.text())
        self.data_view_ui.FuncLabel_max_dist.setText(str(self.max_dist))
        self.data_view_ui.FuncLabel_angle.setText(str(self.selected_angle))

        try:
            if self.multi_lidar_services is not None:
                self.multi_lidar_services.reset_multi_lidar(new_maxdist=self.max_dist,new_angle=self.selected_angle,new_ports_choice=self.ports_choice, env_path=self.envpath,new_selected_env=self.selected_env)
            
            self.mplcanvas = MplCanvas(self.layout_widget, width=5, height=4, dpi=100,angle=int(self.selected_angle))
            #self.connect_unity_ui.verticalLayout.addWidget(self.mplcanvas)
        except FileNotFoundError as e:
            message="path is wrong"
            self.show_error_message(message)
        print(f"Settings Saved: MaxDist={self.max_dist}, BuzzDuration={self.buzz_duration}, Angle={self.selected_angle}, Env={self.selected_env}, Model={self.selected_model}")
        self.save_settings_to_jsonfile()

    def save_csv_settings(self):
        self.frames = int(self.save_csv_ui.Input_frames.text())
        self.name = self.save_csv_ui.Input_name.text()
        self.selected_pose = self.save_csv_ui.FunLabel_selectedpose.text()
        print(f"CSV Settings: Frames={self.frames}, Name={self.name}, Pose={self.selected_pose}")
        self.save_settings_to_jsonfile()

    def start_new_env_save(self):
        self.start_buzzer(self.buzz_duration)
        #============================
        #======================
        #========환경정보 저장 함수, 
        self.multi_lidar_services.save_environment(self.envpath,self.new_env_name,self.new_env_loading_time,self.new_env_margin_dist)
        #======================
        self.load_env_file_list()
        #===========================
        self.start_buzzer(1)
        print(f"{self.new_env_name}.json file is saved")
        self.multi_lidar_services.multi_lidar_driver.stop_lidars()

    def new_env_settings(self):
        self.new_env_name = self.env_set_ui.Input_json.text() +".json"
        self.new_env_loading_time = int(self.env_set_ui.Input_loadingtime.text())
        self.new_env_margin_dist = int(self.env_set_ui.Input_margindist.text())
        print(f"New Environment Settings: json file name={self.new_env_name}, Loading time={self.new_env_loading_time}, Margin dist={self.new_env_margin_dist}")
        self.save_settings_to_jsonfile()
    ##=================================라이다 직접 구동 섹션====================================================##########
    #======데이터 뷰 함수=======#
    def start_data_view_function(self):
        self.update_button_states(starting=True)
        QApplication.processEvents() #button update

        self.start_buzzer(self.buzz_duration)
        
        
        self.is_running = True
        self.multi_lidar_services.multi_lidar_driver.start_lidars()
        time.sleep(1)

        start_time = time.time()

        while self.is_running:
            data_strings=self.multi_lidar_services.view_datas()
            print(data_strings)
            self.data_view_ui.FuncLabel_topdatas.setText(data_strings[0])
            self.data_view_ui.FuncLabel_middatas.setText(data_strings[1])
            self.data_view_ui.FuncLabel_botdatas.setText(data_strings[2])
            self.update_elapsed_time(start_time)
            QApplication.processEvents()
        self.stop_function()
        
    #======스켈레톤 뷰 함수=======#

    def start_skeleton_view_function(self):
        
        self.update_button_states(starting=True)
        self.start_buzzer(self.buzz_duration)
        QApplication.processEvents() #button update

        self.is_running = True
        self.multi_lidar_services.multi_lidar_driver.start_lidars()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        time.sleep(1)
        
        while self.is_running:
            
            points=self.multi_lidar_services.get_detected_point()

            self.plot_real_time(ax,points)
            QApplication.processEvents()
        self.stop_function()

    # ===========전이학습============
    def start_transfer_learn_function(self):
        
        self.update_button_states(starting=True)
        self.start_buzzer(self.buzz_duration)
        QApplication.processEvents() #button update

        self.is_running = True
        self.multi_lidar_services.multi_lidar_driver.start_lidars()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        time.sleep(1)
        
        while self.is_running:
            
            points=self.multi_lidar_services.get_detected_point()
            self.plot_real_time(ax,points)
            QApplication.processEvents()
        self.stop_function()

    #======CSV 데이터 저장 함수=======#
    def start_save_csv_function(self):
        try:
            if self.selected_pose is None:
                
                raise ValueError
            self.usr_click_count=0
            self.save_csv_ui.FuncLabel_count.setText(str(self.usr_click_count))
            self.save_csv_ui.progressBar.setProperty("value", 0)
            self.update_button_states(starting=True)
            QApplication.processEvents() #button update
            
            self.start_buzzer(self.buzz_duration)
            self.is_running = True
            cnt_frame=0


            
            pose_label_num, pose_label = self.selected_pose.split(": ")
            csv_path = f'csv_files/{self.name}/{pose_label}_lidar_data{datetime.now().strftime("%m_%d_%H-%M-%S")}.csv'
            print(csv_path)

            self.multi_lidar_services.multi_lidar_driver.start_lidars()
        
            try:
                csv_saver = WorkerThread()
                csv_saver.start()
                start_time = time.time()
                while cnt_frame < self.frames:
                    filtered_data=self.multi_lidar_services.return_filtered_data()
                    total_dist = np.concatenate(filtered_data)
                    total_dist = np.insert(total_dist, 0, pose_label_num)
                    total_dist = np.insert(total_dist, 1, self.usr_click_count)
                    total_dist = np.insert(total_dist, 2, cnt_frame)
                    csv_saver.start_work(csv_path,total_dist)
                    

                    cnt_frame += 1
                    gaze=cnt_frame*100/self.frames
                    self.save_csv_ui.progressBar.setProperty("value", gaze)
                    QApplication.processEvents()
                    
                    
                    if cnt_frame == self.frames:
                        print(f"Total time: {time.time() - start_time:.2f} seconds\n")
                        break

                    time.sleep(0.05)
            except KeyboardInterrupt:
                pass
            self.stop_function()
        except ValueError as e:
            self.show_error_message("choose any pose")
            self.stop_function()


    def start_connect_unity_function(self):
            #===========유니티 연동 함수=======#
            self.async_handler = AsyncTaskHandler(self.multi_lidar_services, MainApp, self.mplcanvas)
            self.idx_best_motion1,self.idx_best_motion2=0,0
            self.keyboard = Controller()
            self.connect_unity_ui.Button_Connect.setEnabled(True)
            self.update_button_states(starting=True)
            self.start_buzzer(self.buzz_duration)
            self.is_running = True
            self.multi_lidar_services.multi_lidar_driver.start_lidars()

            delays = []  # List to store delays
            motion_predictions = []  # List to store idx_best_motion1 predictions
            self.labels = ["stand", "none", "left_hand", "right_hand", "wave", "squat", "plank", "too_close", "jump", "walk"]

            # Setup the matplotlib figure and axis for real-time plotting
            fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))  # Two subplots: one for delay, one for predictions

            # Plot for delays (subplot 1)
            self.ax1.set_title('Real-Time Delay Plot')
            self.ax1.set_xlabel('Data Points')
            self.ax1.set_ylabel('Delay (s)')
            line1, = self.ax1.plot([], [], lw=2)
            self.ax1.set_xlim(0, 50)  # Set the x-axis to 0-50 for data points
            self.ax1.set_ylim(0, 0.025)  # Adjust the y-axis limits based on expected delays

            # Scatter plot for motion predictions (subplot 2)
            self.ax2.set_title('Real-Time Motion Predictions (idx_best_motion1)')
            self.ax2.set_xlabel('Data Points')
            self.ax2.set_ylabel('Prediction')
            scatter = self.ax2.scatter([], [], color='orange')
            self.ax2.set_xlim(0, 50)  # Set the x-axis to 0-50 for data points
            self.ax2.set_ylim(-1, 10)  # Adjust the y-axis based on possible motion prediction values

            def update_plot(i):
                # Update the delay line with new data
                line1.set_data(list(range(len(delays))), delays)
                # Update the scatter plot with new data
                if delays and motion_predictions:  # Check if there's data to plot
                    scatter.set_offsets(np.column_stack([list(range(len(motion_predictions))), motion_predictions]))

                return line1, scatter,

            ani = FuncAnimation(fig, update_plot, blit=True, interval=100)  # Update the plot every 100 ms
            plt.ion()  # Enable interactive mode
            plt.tight_layout()  # Ensure subplots fit neatly
            plt.show()
            self.async_handler.async_thread.start()
            self.plot_loop()
            
    def start_key_input_function(self):
        self.key_input_approval=True

    #============================라이다 구동함수 끝================================================================================
    def start_buzzer(self, duration):
        print(f"Buzzing for {duration} seconds...")
        freq = 1050  # 주파수는 1500Hz로 고정
        for _ in range(duration):
            winsound.Beep(freq, 700)  # 1000ms = 1초 동안 소리 발생
            time.sleep(0.1)  # 소리 사이에 약간의 지연을 추가 (필요시 제거 가능)        
        
    """def start_elapsed_time_tracking(self):
        self.is_running = True
        self.elapsed_time = 0
        self.timer_thread = threading.Thread(target=self.update_elapsed_time)
        self.timer_thread.start()"""

    def update_elapsed_time(self,start_time):
        self.elapsed_time = time.time() - start_time
        self.data_view_ui.FuncLabel_elaspedtime.setText(f"{self.elapsed_time:.1f}sec")
            #time.sleep(0.05)  # 0.1초마다 경과 시간을 갱신

    def update_button_states(self, starting=False):
        if starting==True:
            # Start 버튼 비활성화, Stop 버튼 활성화
            print("Activating stop button")  # 디버그 메시지
            self.data_view_ui.Button_start.setEnabled(False)
            self.data_view_ui.Button_stop.setEnabled(True)
            self.transfer_learn_ui.Button_start.setEnabled(False)
            self.transfer_learn_ui.Button_stop.setEnabled(True)


            self.skeleton_view_ui.Button_start.setEnabled(False)
            self.skeleton_view_ui.Button_stop.setEnabled(True)

            self.save_csv_ui.Button_start.setEnabled(False)
            self.save_csv_ui.Button_stop.setEnabled(True)
            self.connect_unity_ui.Button_start.setEnabled(False)
            self.connect_unity_ui.Button_stop.setEnabled(True)

            # SaveCsvWindow의 Count for Sequence 버튼 활성화
            self.save_csv_ui.Button_countforsequence.setEnabled(True)
        else:
            # Stop 버튼 비활성화, Start 버튼 활성화
            self.set_buttons_initial_state()

    def stop_function(self):
        print("Stopping the function...")
        self.multi_lidar_services.multi_lidar_driver.stop_lidars()
        self.is_running = False
        self.key_input_approval=False
        if self.timer_thread:###데이터 뷰의 타이머 쓰레드 
            self.timer_thread.join()
        if self.count_thread:###CSV 데이터구축의 카운트 쓰레드 
            self.count_thread.join()

        
        QApplication.processEvents()
        self.update_button_states(starting=False)
    def show_error_message(self, message):
        # QMessageBox를 사용하여 오류 메시지를 표시합니다.
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)  # 오류 아이콘 사용
        msg_box.setText(message)
        msg_box.setWindowTitle("Error")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def use_filter(self,state):
        if state !=2:
            self.multi_lidar_services.use_filter=True
        else:
            self.multi_lidar_services.use_filter=False
        print(self.multi_lidar_services.use_filter)
    
    def save_settings_to_jsonfile(self):
        """현재 설정을 JSON 파일로 저장"""
        settings = {
            "max_dist": self.max_dist,
            "buzz_duration": self.buzz_duration,
            "selected_angle": self.selected_angle,
            "selected_env": self.selected_env,
            "selected_model": self.selected_model,
            "ports_choice": self.ports_choice,
            "frames": self.frames,
            "name": self.name,
            "selected_pose": self.selected_pose,
            "new_env_name": self.new_env_name,
            "new_env_loading_time": self.new_env_loading_time,
            "new_env_margin_dist": self.new_env_margin_dist,
        }
        with open(self.SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=4)
        print("Settings saved to file.")

    def load_settings(self):
        """JSON 파일에서 설정을 불러오기"""
        if os.path.exists(self.SETTINGS_FILE):
            with open(self.SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
                self.max_dist = settings.get("max_dist", 1000)
                self.buzz_duration = settings.get("buzz_duration", 5)
                self.selected_angle = settings.get("selected_angle", 90)
                self.selected_env = settings.get("selected_env", "")
                self.selected_model = settings.get("selected_model", "")
                self.ports_choice = settings.get("ports_choice", [1, 2, 3])
                
                
                self.frames = settings.get("frames", 100)
                self.name = settings.get("name", "")
                self.selected_pose = settings.get("selected_pose", "")
                self.new_env_name = settings.get("new_env_name", "")
                self.new_env_loading_time = settings.get("new_env_loading_time", 10)
                self.new_env_margin_dist = settings.get("new_env_margin_dist", 50)
            print("Settings loaded from file.")
            
        else:
            print("No settings file found. Using default values.")
    def default_initialization(self):
        """기본 초기화 설정"""
        self.ports_choice = [1, 2, 3]
        self.envpath = self.folder_name_qt+"env_jsons"
        self.max_dist = 1000
        self.buzz_duration = 5
        self.selected_angle = 90
        self.selected_env = "None"
        self.selected_model = "None"
        self.frames = 500
        self.name = "None"
        self.selected_pose = "None"
        self.new_env_name = "default"
        self.new_env_loading_time = 10
        self.new_env_margin_dist = 0
        print("Default settings applied.")

    def update_ui_from_settings(self):
        """로드된 설정에 따라 UI 요소들을 업데이트"""
        self.setting_ui.Input_maxdist.setText(str(self.max_dist))
        self.setting_ui.Input_buzzduration.setText(str(self.buzz_duration))
        if self.selected_angle == 90:
            self.setting_ui.radioButton_90.setChecked(True)
        else:
            self.setting_ui.radioButton_180.setChecked(True)
        self.setting_ui.FuncLabel_selected_env.setText(self.selected_env)
        self.setting_ui.FuncLabel_selected_model.setText(self.selected_model)
        self.setting_ui.Input_port_top.setText(str(self.ports_choice[0]))
        self.setting_ui.Input_port_mid.setText(str(self.ports_choice[1]))
        self.setting_ui.Input_port_bot.setText(str(self.ports_choice[2]))
        self.save_csv_ui.Input_frames.setText(str(self.frames))
        self.save_csv_ui.Input_name.setText(self.name)
        self.save_csv_ui.FunLabel_selectedpose.setText(self.selected_pose)
        
        dot_index= self.new_env_name.rfind('.')
        new_env_name = self.new_env_name[:dot_index]
        self.env_set_ui.Input_json.setText(new_env_name)
        self.env_set_ui.Input_loadingtime.setText(str(self.new_env_loading_time))
        self.env_set_ui.Input_margindist.setText(str(self.new_env_margin_dist))
        
        print("UI updated from loaded settings.")
    def connect_ui_buttons(self):
        """모든 UI 버튼 이벤트를 연결"""
        self.ui.Button_setting.clicked.connect(self.show_setting_window)
        self.ui.Button_dataview.clicked.connect(self.show_data_view_window)
        self.ui.Button_envset.clicked.connect(self.show_env_set_window)
        self.ui.Button_transferlearn.clicked.connect(self.show_transfer_learn_window)

        self.ui.Button_3dview.clicked.connect(self.show_skeleton_view_window)
        self.ui.Button_csvsave.clicked.connect(self.show_save_csv_window)
        self.ui.Button_unity.clicked.connect(self.show_connect_unity_window)
        self.ui.Button_usbconnection.clicked.connect(self.connect_lidars_with_usb)
        self.ui.Button_wificonnection.clicked.connect(self.connect_lidars_with_wifi)

        # SettingWindow 이벤트 연결
        self.setting_ui.List_env.itemClicked.connect(self.select_env)
        self.setting_ui.List_model.itemClicked.connect(self.select_model)
        self.setting_ui.Button_back.clicked.connect(self.show_main_window)
        self.setting_ui.Button_savesetting.clicked.connect(self.save_setting)

        # DataViewWindow 이벤트 연결
        self.data_view_ui.Button_back.clicked.connect(self.show_main_window)
        self.data_view_ui.checkBox_nonefilter.stateChanged.connect(self.use_filter)

        # EnvSetWindow 이벤트 연결
        self.env_set_ui.Button_back.clicked.connect(self.show_main_window)
        self.env_set_ui.Button_save.clicked.connect(self.start_new_env_save)
        self.env_set_ui.Button_adjust.clicked.connect(self.new_env_settings)

        # Transfer_learn 이벤트 연결
        self.transfer_learn_ui.Button_back.clicked.connect(self.show_main_window)

        # SkeletonViewWindow 이벤트 연결
        self.skeleton_view_ui.Button_back.clicked.connect(self.show_main_window)

        # SaveCsvWindow 이벤트 연결
        self.save_csv_ui.List_pose.itemClicked.connect(self.select_pose)
        self.save_csv_ui.Button_back.clicked.connect(self.show_main_window)
        self.save_csv_ui.Button_countforsequence.clicked.connect(self.count_for_sequence)
        self.save_csv_ui.Button_adjust.clicked.connect(self.save_csv_settings)

        # ConnectUnityWindow 이벤트 연결
        self.connect_unity_ui.Button_back.clicked.connect(self.show_main_window)

        # Start/Stop 버튼 연결
        self.data_view_ui.Button_start.clicked.connect(self.start_data_view_function)
        self.data_view_ui.Button_stop.clicked.connect(self.stop_function)
        self.transfer_learn_ui.Button_start.clicked.connect(self.start_transfer_learn_function)
        self.transfer_learn_ui.Button_stop.clicked.connect(self.stop_function)
        self.skeleton_view_ui.Button_start.clicked.connect(self.start_skeleton_view_function)
        self.skeleton_view_ui.Button_stop.clicked.connect(self.stop_function)
        
        self.save_csv_ui.Button_start.clicked.connect(self.start_save_csv_function)
        self.save_csv_ui.Button_stop.clicked.connect(self.stop_function)
        self.connect_unity_ui.Button_start.clicked.connect(self.start_connect_unity_function)
        self.connect_unity_ui.Button_stop.clicked.connect(self.stop_function)
        self.connect_unity_ui.Button_Connect.clicked.connect(self.start_key_input_function)
        self.connect_unity_ui.Button_stop.clicked.connect(self.stop_function)

    def set_buttons_by_connection(self):
        if self.multi_lidar_services is None:
            # 라이다 서비스가 없으면 Start 버튼을 비활성화
            self.data_view_ui.Button_start.setEnabled(False)
            self.skeleton_view_ui.Button_start.setEnabled(False)
            self.transfer_learn_ui.Button_start.setEnabled(False)
            self.save_csv_ui.Button_start.setEnabled(False)
            self.connect_unity_ui.Button_start.setEnabled(False)
            self.connect_unity_ui.Button_Connect.setEnabled(False)
            # 라이다 서비스가 없으면 Stop 버튼을 비활성화
            self.data_view_ui.Button_stop.setEnabled(False)
            self.skeleton_view_ui.Button_stop.setEnabled(False)
            self.transfer_learn_ui.Button_stop.setEnabled(False)
            self.save_csv_ui.Button_stop.setEnabled(False)
            self.connect_unity_ui.Button_stop.setEnabled(False)

        else:
            # 라이다 서비스가 있을 때 기본 버튼 활성화
            self.data_view_ui.Button_start.setEnabled(True)
            self.skeleton_view_ui.Button_start.setEnabled(True)
            self.transfer_learn_ui.Button_start.setEnabled(True)
            self.save_csv_ui.Button_start.setEnabled(True)
            self.connect_unity_ui.Button_start.setEnabled(True)

    def connect_lidars_with_usb(self):
        # 멀티라이다 서비스 객체 생성 및 오류처리
        try:
            self.multi_lidar_services = MultiLidarServices(model_path=self.folder_name_qt+'models\\'+ self.selected_model)
            
            #프로그램 시작시 settings 디폴트값으로 초기화
            tmp=self.multi_lidar_services.multi_lidar_driver.detect_lidar_ports()
            self.setting_ui.FuncLable_portlist.setText(", ".join(tmp))
            self.multi_lidar_services.reset_multi_lidar(new_maxdist=self.max_dist,new_angle=self.selected_angle,new_ports_choice=self.ports_choice, env_path=self.envpath,new_selected_env=self.selected_env)

            self.set_buttons_by_connection()
        except IndexError as e:
            print(f"Error: {e}")
            self.show_error_message("Connect three Lidars correctly") #라이다 미연결 시 예외처리
            self.multi_lidar_services = None

    def connect_lidars_with_wifi(self):
        # 멀티라이다 서비스 객체 생성 및 오류처리
        try:
            self.multi_lidar_services = MultiLidarServices(model_path=self.selected_model)
            self.multi_lidar_services.reset_multi_lidar(new_maxdist=self.max_dist,new_angle=self.selected_angle,new_ports_choice=self.ports_choice, env_path=self.envpath,new_selected_env=self.selected_env)

            self.set_buttons_by_connection()
        except IndexError as e:
            print(f"Error: {e}")
            self.show_error_message("Connect three Lidars correctly") #라이다 미연결 시 예외처리
            self.multi_lidar_services = None

    def plot_real_time(self, ax, points):
        ax.cla()  # 이전 플롯 클리어

        # 라이다 센서의 위치를 먼저 플로팅
        sensor_positions = [
            (self.multi_lidar_services.sensor_top_x, self.multi_lidar_services.sensor_top_y, self.multi_lidar_services.sensor_top_z),
            (self.multi_lidar_services.sensor_mid_x, self.multi_lidar_services.sensor_mid_y, self.multi_lidar_services.sensor_mid_z),
            (self.multi_lidar_services.sensor_bot_x, self.multi_lidar_services.sensor_bot_y, self.multi_lidar_services.sensor_bot_z)
        ]

        for pos in sensor_positions:
            ax.scatter(pos[0], pos[1], pos[2], color='red', s=100, label='Sensor Position')
            break

        # 포인트 플로팅
        row_0_points = [p for p in points if p['row'] == 0 and p['center']]
        row_1_points = [p for p in points if p['row'] == 1 and p['center']]
        row_2_points = [p for p in points if p['row'] == 2]  # row 2는 center 여부 상관없이 모두 사용

        non_center_points = [p for p in points if (p['row'] == 0 or p['row'] == 1) and not p['center']]


        if row_0_points and row_1_points:
            # 첫 번째 row=0, center=True인 점과 row=1, center=True인 점
            row_0_point = row_0_points[0]
            row_1_point = row_1_points[0]

            # 중간 점 계산 (x, y, z 값의 평균)
            mid_point = {
                'x': (row_0_point['x'] + row_1_point['x']) / 2,
                'y': (row_0_point['y'] + row_1_point['y']) / 2,
                'z': (row_0_point['z'] + row_1_point['z']) / 2
            }

            # 중간 점을 플로팅
            ax.scatter(mid_point['x'], mid_point['y'], mid_point['z'], color='purple', s=100)

            # 중간점과 row 0 연결
            ax.plot([mid_point['x'], row_0_point['x']], 
                    [mid_point['y'], row_0_point['y']], 
                    [mid_point['z'], row_0_point['z']], 
                    color='green')

            # 중간점과 row 1 연결
            ax.plot([mid_point['x'], row_1_point['x']], 
                    [mid_point['y'], row_1_point['y']], 
                    [mid_point['z'], row_1_point['z']], 
                    color='orange')

            # row=2의 모든 점들과 중간 점을 연결 (center 여부 상관없이)
            for row_2_point in row_2_points:
                ax.plot([row_1_point['x'], row_2_point['x']], 
                        [row_1_point['y'], row_2_point['y']], 
                        [row_1_point['z'], row_2_point['z']], 
                        color='blue')
            for non_center_point in non_center_points:
                ax.plot([mid_point['x'], non_center_point['x']], 
                        [mid_point['y'], non_center_point['y']], 
                        [mid_point['z'], non_center_point['z']], 
                        color='red')

        # 나머지 점들을 각각 플로팅
        xs = [p['x'] for p in points]
        ys = [p['y'] for p in points]
        zs = [p['z'] for p in points]

        ax.scatter(xs, ys, zs, color='blue', s=50)

        # 좌표 축 제한
        ax.set_xlim([-500, 500])
        ax.set_ylim([0, 1500])
        ax.set_zlim([0, 2000])
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.legend()  # 범례 추가


        plt.pause(0.01)  # 플롯 갱신 대기 시간 (0.1초)

    def press_keyboard(self, idx_best_motion, player_keys, keydown_time=0.05):
        keys=[None,None]
        if idx_best_motion in player_keys:
            keys = player_keys[idx_best_motion]
        key_first,key_second=keys[0],keys[1]
        print(key_first,key_second)
        def press_and_release_key(key):
            if key is None:
                return  # None일 경우 아무 동작도 하지 않음
            elif isinstance(key, str):
                self.keyboard.press(key)
                time.sleep(keydown_time)
                self.keyboard.release(key)
            else:
                self.keyboard.press(key)
                time.sleep(keydown_time)
                self.keyboard.release(key)
         # 두 키가 모두 유효할 때 동시에 입력 처리
        if key_first and key_second:
            self.keyboard.press(key_first)
            self.keyboard.press(key_second)
            time.sleep(keydown_time)
            self.keyboard.release(key_first)
            self.keyboard.release(key_second)
        
        # 첫 번째 키만 유효할 때
        elif key_first:
            press_and_release_key(key_first)
        
        # 두 번째 키만 유효할 때
        elif key_second:
            press_and_release_key(key_second)
    def plot_loop(self):
        delays = []
        motion_predictions = []

        while True:
            # 비동기 작업이 끝났다면 다시 시작
            if not self.async_handler.flag.is_set():
                self.async_handler.flag.set()

            # Plot 갱신 관련 작업
            #idx_best_motion1, idx_best_motion2 = self.mplcanvas.update_plot_by_realtime_motion(self.idx_best_motion1, self.idx_best_motion2)
            motion_predictions.append(self.idx_best_motion1)

            # 최신 50개의 데이터만 유지
            if len(delays) > 50:
                delays.pop(0)
            if len(motion_predictions) > 50:
                motion_predictions.pop(0)

            self.motion_update_plot(delays, motion_predictions)

            # 키보드 입력 처리 (별도 쓰레드로 실행)
            if self.key_input_approval:
                threading.Thread(target=self.press_keyboard, args=(idx_best_motion1, self.key_mapping_play1, 0.05)).start()

            QtWidgets.QApplication.processEvents()
            time.sleep(0.0001)  # 메인 루프 딜레이 (CPU 사용량 조절)

    def motion_update_plot(self, delays, motion_predictions):
        self.ax1.set_xlim(0, 50)
        self.ax2.set_xlim(0, 50)

        if len(delays) > 0 and delays[-1] > self.ax1.get_ylim()[1]:
            self.ax1.set_ylim(0, delays[-1] + 0.01)

        if len(motion_predictions) > 0 and motion_predictions[-1] > self.ax2.get_ylim()[1]:
            self.ax2.set_ylim(-1, motion_predictions[-1] + 1)

        self.ax2.set_yticks(range(len(self.labels)))
        self.ax2.set_yticklabels(self.labels)

    def motion_stop_function(self):
        self.async_handler.stop()
class WorkerThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self._flag = threading.Event()  # 스레드가 동작할 때만 True가 되는 이벤트 플래그
        self._stop_event = threading.Event()  # 스레드 중지를 제어하는 플래그
        self.data = None  # 작업할 데이터를 저장하는 변수
        self.path= None

    def run(self):
        while not self._stop_event.is_set():
            self._flag.wait()  # 호출될 때까지 대기
            if self.data is not None:
                # 실제 작업 수행
                # 여기가 `process_data`를 실제로 호출하는 부분입니다.
                result = self.process_data(self.path,self.data)
                self.data = None  # 데이터 처리 후 초기화
            self._flag.clear()  # 작업이 끝나면 다시 대기 상태로 전환

    def process_data(self, path,data):
        directory = os.path.dirname(path)
    
        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(data)

    def start_work(self, path,data):
        self.path= path # 작업할 경로를 설정
        self.data = data  # 작업할 데이터를 설정
        self._flag.set()  # 작업 시작을 알리는 플래그를 설정

    def stop(self):
        self._stop_event.set()  # 스레드 중지 플래그를 설정
        self._flag.set()  # 대기 중인 스레드를 깨워서 종료시킴

class AsyncTaskHandler:
    def __init__(self, lidar_service, Mainapp,canvas):
        self.main_app=Mainapp
        self.is_running = True
        self.lidar_service = lidar_service
        self.canvas = canvas
        self.flag = threading.Event()  # 비동기 작업을 제어하는 플래그
        self.flag.set()  # 기본적으로 비동기 작업 실행 가능 상태로 설정

        # 쓰레드 시작
        self.async_thread = threading.Thread(target=self.run)
        

    def run(self):
        while self.is_running:
            print("================wait했음")
            self.flag.wait()  # 플래그가 설정될 때까지 대기

            # LiDAR 데이터를 가져오고 비동기 작업을 수행
            filtered_data = self.lidar_service.return_filtered_data()
            filtered_data = [item for sublist in filtered_data for item in sublist]
            
            motion_accuracies, delay = self.lidar_service.get_motion_by_AI(filtered_data)
            print(f"비동기 작업 딜레이: {delay}")

            # Plot 업데이트
            self.main_app.idx_best_motion1, self.main_app.idx_best_motion2 = self.canvas.update_plot_by_realtime_motion(motion_accuracies, None)

            # 작업 완료 후 플래그 해제 (다음 작업까지 대기 상태로 전환)
            self.flag.clear()

            time.sleep(0.0001)  # 작업 간 간격 조정

    def stop(self):
        self.is_running = False
        self.flag.set()  # 플래그를 설정해 대기 중인 쓰레드를 깨움

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())
