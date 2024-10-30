from . import multi_lidar_driver
import json
import os
import time
import numpy as np
from PyQt5.QtWidgets import QApplication
from . import ai_model_realtime


class MultiLidarServices:
    def __init__(self,model_path):
        self.multi_lidar_driver=multi_lidar_driver.MultiLidardriver(angle=90, max_distance=2000,ports_choice=[1, 2, 3],FPS=20)
        self.model=ai_model_realtime.STEP50CNN(model_path=model_path)
        self.use_filter=True
        self.env_max_distances=3*[90*[0]] #json으로부터 불러와야할 거
        self.env_name="QT\env_jsons\hi.json" #일단 디폴트값
        self.multi_lidar_driver.setup_lidars()

        # 라이다 센서의 설치 위치 (x, y, z)
        self.sensor_top_x, self.sensor_top_y, self.sensor_top_z = 0, 0, 1170
        self.sensor_mid_x, self.sensor_mid_y, self.sensor_mid_z = 0, 0, 1110
        self.sensor_bot_x, self.sensor_bot_y, self.sensor_bot_z = 0, 0, 1030

        # 라이다 센서의 기울기 각도
        self.pitch_angle_top = 16  # abdomen을 기준으로
        self.pitch_angle_mid = 0 #abdomen은 x,y평면, z는 1110
        self.pitch_angle_bot = -23.4  # abdomen을 기준으로
        self.center_angle=45 #측정각 절반

            
    def get_detected_point(self):
        filtered_data=self.return_filtered_data()
        
        all_summaries =self.process_and_summarize_all(filtered_data)
        return self.calculate_points(all_summaries)

    def reset_multi_lidar(self,new_angle=90,new_maxdist=2000,new_ports_choice=[1, 2, 3],new_FPS=20,env_path=None,new_selected_env=None):
        self.multi_lidar_driver=multi_lidar_driver.MultiLidardriver(angle=new_angle, max_distance=new_maxdist,ports_choice=new_ports_choice,FPS=new_FPS)
        self.multi_lidar_driver.setup_lidars()
        self.env_max_distances = self.load_environment(env_path,new_selected_env)
        print(self.env_max_distances)
        if self.env_max_distances==0:
            raise FileNotFoundError

    def save_csv_datas(self,selected_pose,name):
        print(1)
        QApplication.processEvents()

    def return_filtered_data(self):
        distances=self.multi_lidar_driver.get_distances()

        if self.use_filter==True:
            filted_dist_top=np.where(distances[0] < self.env_max_distances[0], distances[0], 0)
            filted_dist_mid=np.where(distances[1] < self.env_max_distances[1], distances[1], 0)
            filted_dist_bot=np.where(distances[2] < self.env_max_distances[2], distances[2], 0)
            return [filted_dist_top,filted_dist_mid,filted_dist_bot]
        else:
            return [distances[0],distances[1],distances[2]]

    def view_datas(self):
        
        distances=self.multi_lidar_driver.get_distances()
        #print(distances[0])
        #print(self.env_max_distances[0])
        #print(env_max_distances[0])
        if self.use_filter==True:
            filted_dist_top=np.where(distances[0] < self.env_max_distances[0], distances[0], 0)
            filted_dist_mid=np.where(distances[1] < self.env_max_distances[1], distances[1], 0)
            filted_dist_bot=np.where(distances[2] < self.env_max_distances[2], distances[2], 0)
            filted_dist_top=', '.join(map(str, filted_dist_top))
            filted_dist_mid=', '.join(map(str, filted_dist_mid))
            filted_dist_bot=', '.join(map(str, filted_dist_bot))
            return [filted_dist_top,filted_dist_mid,filted_dist_bot]
        else:
            dist_top=', '.join(map(str, distances[0]))
            dist_mid=', '.join(map(str, distances[1]))
            dist_bot=', '.join(map(str, distances[2]))
            return [dist_top,dist_mid,dist_bot]
            
    def environment_filtering(self,Input_loadingtime,env_margin):
        environment_distances = [[],[],[]]
        start_time = time.time()
        current_time = 0
        self.multi_lidar_driver.start_lidars()
        time.sleep(1)
        while current_time < Input_loadingtime:
            distances=self.multi_lidar_driver.get_distances()
            for i, dist in enumerate(distances):
                environment_distances[i].append(dist)
            current_time = time.time() - start_time
            print(f"Filtering Mode - Elapsed Time: {current_time:.2f}s")

        print("Done filtering environment")

        min_distances = [np.min(dist, axis=0) - env_margin for dist in environment_distances if len(dist) > 0]
        print(min_distances)
        return min_distances

    def save_environment(self, env_path,user_env_name, Input_loadingtime,env_margin):
        new_env_max_distances=self.environment_filtering(Input_loadingtime,env_margin)
        lidar_data = {f"lidar{i+1}": dist.tolist() for i, dist in enumerate(new_env_max_distances)}
        with open(os.path.join(env_path, f"{user_env_name}"), 'w') as json_file:
            json.dump(lidar_data, json_file)

    def load_environment(self, env_path, env_name):
        #print(env_path,env_name)
        if env_name.endswith('.json') == False:
            print("Environemnt file is not selected")
            return 0
        if env_path is None:
            print("Environemnt path is wrong")
            return 0
        
        with open(os.path.join(env_path, f"{env_name}"), 'r') as json_file:
            lidar_data = json.load(json_file)
        self.env_max_distances = [np.array(lidar_data[f"lidar{i+1}"]) for i in range(3)]
        return self.env_max_distances
    
    def process_array_clusters(self,row):
        clusters = []
        current_cluster = []
        zero_count = 0

        for index, value in enumerate(row):
            if value != 0:
                current_cluster.append({'row': None, 'index': index, 'value': value})
                zero_count = 0
            else:
                zero_count += 1
                if zero_count == 2:
                    if current_cluster:
                        clusters.append(current_cluster)
                        current_cluster = []
                    zero_count = 0

        if current_cluster:
            clusters.append(current_cluster)

        return clusters

    def summarize_clusters(self, clusters, row_index):
        summaries = []

        for cluster in clusters:
            if not cluster:
                continue

            for item in cluster:
                item['row'] = row_index

            indices = [item['index'] for item in cluster]
            values = [item['value'] for item in cluster if item['value'] != 0]

            if not values:
                continue

            index_mean = round(sum(indices) / len(indices))
            value_mean = round(sum(values) / len(values))

            summary = {
                'row': row_index,
                'index_mean': index_mean,
                'value_mean': value_mean
            }

            summaries.append(summary)

        return summaries
    
    def process_and_summarize_all(self,data):
        """
        주어진 데이터의 각 행에 대해 클러스터를 식별하고 요약 정보를 생성하여 반환합니다.
        
        :param data: 2차원 리스트 형태의 정수 데이터
        :return: 클러스터 요약 정보가 담긴 리스트
        """
        all_summaries = []

        for row_index, row in enumerate(data):
            clusters = self.process_array_clusters(row)
            
            summaries = self.summarize_clusters(clusters, row_index)
            all_summaries.extend(summaries)

        return all_summaries
    
    def calculate_points(self, all_summaries):
        points = []
        thetas=[]
        center_true=False
        for summary in all_summaries:
            row = summary['row']
            angle = summary['index_mean']
            distance = summary['value_mean']
            
            # Convert angle to radians
            
            theta = angle-self.center_angle
            thetas.append(theta)
            if theta <10 and theta>-10:
                center_true=True
            theta = np.radians(angle-self.center_angle)
            
            
            if row == 0:
                sensor_z = self.sensor_top_z
                pitch_angle = np.radians(self.pitch_angle_top)
            elif row == 1:
                sensor_z = self.sensor_mid_z
                pitch_angle = np.radians(self.pitch_angle_mid)
            elif row == 2:
                sensor_z = self.sensor_bot_z
                pitch_angle = np.radians(self.pitch_angle_bot)
            else:
                continue  # Invalid row
            
            # Calculate the x, y, z coordinates
            x = distance * np.sin(theta) * np.cos(pitch_angle)
            y = distance * np.cos(theta) * np.cos(pitch_angle)
            z = sensor_z + distance * np.sin(pitch_angle)
            
            points.append({'x': x, 'y': y, 'z': z, 'row':row, 'center':center_true})
            center_true=False
        print("클러스터 중심 각도 : ", thetas)
        print(points)
        
        return points
            
    def get_motion_by_AI(self,data):
        
        result = self.model.predict(data)
        
        
        return result[0]
    
    def plot_3d_points(self, points):
        fig = plt.figure()
        ax.cla()

        # 라이다 센서의 위치를 먼저 플로팅
        sensor_positions = [
            (self.sensor_top_x, self.sensor_top_y, self.sensor_top_z),
            (self.sensor_mid_x, self.sensor_mid_y, self.sensor_mid_z),
            (self.sensor_bot_x, self.sensor_bot_y, self.sensor_bot_z)
        ]

        for pos in sensor_positions:
            ax.scatter(pos[0], pos[1], pos[2], color='red', s=100, label='Sensor Position')

        # 포인트 플로팅
        xs = [p['x'] for p in points]
        ys = [p['y'] for p in points]
        zs = [p['z'] for p in points]

        ax.scatter(xs, ys, zs, color='blue', s=50, label='Detected Points')

        # 좌표 축 제한
        ax.set_xlim([-500, 500])
        ax.set_ylim([0, 1500])
        ax.set_zlim([0, 2000])

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        plt.legend()
        plt.show()