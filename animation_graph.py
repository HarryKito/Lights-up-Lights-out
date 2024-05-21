import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 비디오 파일 경로 설정
video_path = 'resources/ONOFF.MOV'

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("영상 없음.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fig, ax = plt.subplots()
brightness_means = []
time_stamps = []
frame_number = 0

def update(frame_number):
    global cap
    ret, frame = cap.read()
    
    if not ret:
        return
    
    # 프레임 시간
    time_stamp = frame_number / fps
    time_stamps.append(time_stamp)
    
    # 프레임을 HSV 색 공간으로 변환
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 밝기 값(V 채널) 추출
    value_channel = hsv_frame[:, :, 2]
    
    # 밝기 값의 평균 계산
    mean_value = value_channel.mean()
    brightness_means.append(mean_value)
    
    # 그래프 업데이트
    ax.clear()
    ax.plot(time_stamps, brightness_means, label='mean of values', color='red')
    ax.set_title('Mean of values')
    ax.set_xlabel('Time')
    ax.set_ylabel('Brightness mean')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    # OpenCV 창 업데이트
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        plt.close()

# 애니메이션 설정
ani = FuncAnimation(fig, update, frames=total_frames, interval=1000/fps, repeat=False)

# Matplotlib 애니메이션 시작
plt.show()

# 비디오 캡처 객체 해제
cap.release()
cv2.destroyAllWindows()

