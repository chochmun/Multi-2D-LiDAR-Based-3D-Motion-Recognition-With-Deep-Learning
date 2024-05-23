import Ydlidar_Interface as ydlidar
import time
import math
import numpy as np
import tkinter as tk
import serial.tools.list_ports

def list_serial_ports():
    ports = serial.tools.list_ports.comports()
    port_names = []

    for port in ports:
        port_names.append(port.device)
    
    return port_names
port_names = list_serial_ports()
print(port_names)

D_ANGLE= 90
D_start_angle=int(180-(90-D_ANGLE/2))
D_end_angle= int(90-D_ANGLE/2)

port1 = port_names[1]
port2 = port_names[0]
port3 = port_names[2]

cycle_size=100
cycle_size2=150
cycle_size3=200
cycle_size4=250
point_size=4
plot_width=600
plot_height=300
max_range=4000
scaling=4

spot_count=0
spot_count_prev=0

def draw_setting(canvas):
    x=plot_width/2
    y=plot_height
    canvas.create_oval(x - cycle_size, y - cycle_size, x + cycle_size, y + cycle_size, outline="black", width=2)
    canvas.create_oval(x - cycle_size2, y - cycle_size2, x + cycle_size2, y + cycle_size2, outline="black", width=2)
    canvas.create_oval(x - cycle_size3, y - cycle_size3, x + cycle_size3, y + cycle_size3, outline="black", width=2)
    canvas.create_oval(x - cycle_size4, y - cycle_size4, x + cycle_size4, y + cycle_size4, outline="black", width=2)
    canvas.create_oval(x - 25, y - 25, x + 25, y + 25, fill="black")

    canvas.create_oval(x - cycle_size+plot_width, y - cycle_size, x + cycle_size+plot_width, y + cycle_size, outline="black", width=2)
    canvas.create_oval(x - cycle_size2+plot_width, y - cycle_size2, x + cycle_size2+plot_width, y + cycle_size2, outline="black", width=2)
    canvas.create_oval(x - cycle_size3+plot_width, y - cycle_size3, x + cycle_size3+plot_width, y + cycle_size3, outline="black", width=2)
    canvas.create_oval(x - cycle_size4+plot_width, y - cycle_size4, x + cycle_size4+plot_width, y + cycle_size4, outline="black", width=2)
    canvas.create_oval(x - 25+plot_width, y - 25, x + 25+plot_width, y + 25, fill="black")

    canvas.create_oval(x - cycle_size+plot_width*2, y - cycle_size, x + cycle_size+plot_width*2, y + cycle_size, outline="black", width=2)
    canvas.create_oval(x - cycle_size2+plot_width*2, y - cycle_size2, x + cycle_size2+plot_width*2, y + cycle_size2, outline="black", width=2)
    canvas.create_oval(x - cycle_size3+plot_width*2, y - cycle_size3, x + cycle_size3+plot_width*2, y + cycle_size3, outline="black", width=2)
    canvas.create_oval(x - cycle_size4+plot_width*2, y - cycle_size4, x + cycle_size4+plot_width*2, y + cycle_size4, outline="black", width=2)
    canvas.create_oval(x - 25+plot_width*2, y - 25, x + 25+plot_width*2, y + 25, fill="black")

def text_setting(canvas):
    label = tk.Label(root, text='Spot total :\nSpots per sec :\ntime :',width=12, height=4,font=("Helvetica", 12),anchor="n")
    label.place(x=0, y=0)


def display_num(a,b,num):
    label = tk.Label(root, width=3, height=1,text=str(num))
    label.place(x=a,y=b)

def draw_point(event,x,y,color):
    global spot_count
    if x==0 or y==0:
        pass
    else:
        spot_count = spot_count+1
        #x/=scaling
        #y/=scaling
        canvas.create_oval(x+plot_width/2 - point_size, plot_height-y - point_size, x+plot_width/2 + point_size, plot_height-y + point_size, fill=color)



def polar_to_cartesian(angle_degrees, distance):
    # 각도를 라디안으로 변환
    angle_radians = math.radians(D_start_angle-angle_degrees)
    # 삼각함수를 사용하여 x, y 좌표 계산
    x = distance * math.cos(angle_radians)
    y = distance * math.sin(angle_radians)
    return x, y

#주어진 리스트의 모든 값이 중복되있는지 확인.
def all_elements_are_different(lst):
    return len(set(lst)) != 1

def detected_point(lst):
        if  all_elements_are_different(lst):
            a,b=polar_to_cartesian(np.argmin(lst),np.min(lst) )
            if -max_range <= a <= max_range and 0 <= b <= max_range:
                return a, b
        return 0, 0
def split_point(x, y):
    half_x = x / scaling
    half_y = y / scaling
    return half_x, half_y
lid1 = ydlidar.YDLidarX2(port1)
lid1.connect()
lid1.start_scan()
lid2 = ydlidar.YDLidarX2(port2)
lid2.connect()
lid2.start_scan()
lid3 = ydlidar.YDLidarX2(port3)
lid3.connect()
lid3.start_scan()
print("LiDAR started")

root = tk.Tk()
root.title("lidar tkinter90")

canvas = tk.Canvas(root, width=plot_width*3, height=plot_height, bg="white")
canvas.pack()
draw_setting(canvas)
text_setting(canvas)
display_num(130,25,0)
start_time_start=time.time()
start_time=time.time()
try:
    while True:
        time_sec= time.time()-start_time_start
        interval= time.time()-start_time
        display_num(130, 45, int(time_sec))

        if interval >1:
            #총 카운트 - 1초전 카운트 = 1초간 총 카운트
            spots_per_sec=spot_count-spot_count_prev
            spot_count_prev=spot_count

            #init
            start_time=time.time()
            interval=0
            display_num(130,25,spots_per_sec)

        if lid1.available:
            distances1 = lid1.get_data()
            x1,y1=detected_point(distances1)
            x2,y2=split_point(x1,y1)
                                
            draw_point(canvas,x2,y2,"red")

            distances2 = lid2.get_data()
            x1,y1=detected_point(distances2)
            x2,y2=split_point(x1,y1)

            draw_point(canvas,plot_width+x2,y2,"blue")

            distances3 = lid3.get_data()
            x1,y1=detected_point(distances3)
            x2,y2=split_point(x1,y1)
            
            draw_point(canvas,plot_width*2+x2,y2,"green")

            display_num(130,5,spot_count)
            root.update()
        else:
            pass
        time.sleep(0.05)
        
except KeyboardInterrupt:
    pass


lid1.stop_scan()
lid2.stop_scan()
lid3.stop_scan()
lid1.disconnect()
lid2.disconnect()
lid3.disconnect()
print("Done")
root.mainloop()