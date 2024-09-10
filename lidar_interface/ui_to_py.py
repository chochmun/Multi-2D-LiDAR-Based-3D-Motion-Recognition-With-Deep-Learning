import os
import subprocess

# 경로 설정
ui_folder = os.path.join("lidar_interface", "ui")
output_folder = os.path.join("lidar_interface", "ui_py")  # Python 파일들을 저장할 폴더 (현재 폴더로 설정)

# ui 폴더에 있는 모든 .ui 파일들을 변환
for filename in os.listdir(ui_folder):
    if filename.endswith(".ui"):
        ui_file = os.path.join(ui_folder, filename)
        py_file = os.path.join(output_folder, filename.replace(".ui", ".py"))
        
        # pyuic5를 사용하여 변환 명령 실행
        command = f"pyuic5 -x {ui_file} -o {py_file}"
        subprocess.run(command, shell=True)

        print(f"Converted {ui_file} to {py_file}")
