import matplotlib.pyplot as plt
import numpy as np

# 데이터 설정
models = ["CNN", "Transformer", "Transformer+CNN"]
accuracies = [95, 94, 96]  # 정확도 (%)
delays = [0.09, 0.04, 0.06]  # 시간 지연 (초)

# 목표 정확도 및 지연 시간
target_accuracy = 95
target_delay = 0.085

# 그래프 그리기
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# 정확도 그래프
bars1 = ax1.bar(models, accuracies, color='blue')
ax1.axhline(target_accuracy, color='red', linestyle='--', label='Target Accuracy(95%)')
ax1.set_ylim(90, 100)
ax1.set_title("Accuracy of Models")
ax1.set_ylabel("Accuracy (%)")
ax1.legend(loc="upper right", fontsize=12)  # 범례 크기 조정

# 막대 위에 정확도 표시
for bar, accuracy in zip(bars1, accuracies):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 1, f'{accuracy}%', 
             ha='center', va='bottom', color='white', fontsize=16)

# 지연 시간 그래프
bars2 = ax2.bar(models, delays, color='red')
ax2.axhline(target_delay, color='blue', linestyle='--', label='Target Delay(0.2sec)')
ax2.set_ylim(0, 0.1)
ax2.set_title("Time Delay of Models")
ax2.set_ylabel("Time Delay (sec)")
ax2.legend(loc="upper right", fontsize=12)  # 범례 크기 조정

# 막대 위에 지연 시간 표시
for bar, delay in zip(bars2, delays):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.01, f'{delay:.2f}s', 
             ha='center', va='bottom', color='white', fontsize=16)

# x축 레이블 표시 설정 (수직 정렬)
ax1.set_xticklabels(models, rotation=0)
ax2.set_xticklabels(models, rotation=0)

# 전체 레이아웃 조정 및 표시
plt.tight_layout()
plt.show()
