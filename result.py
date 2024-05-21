import cv2
import numpy as np
import matplotlib.pyplot as plt

# EKF 필터 클래스
class EKF:
    def __init__(self, Q, R):
        self.F = np.array([[1]])
        self.H = np.array([[1]])
        self.Q = Q
        self.R = R
        self.x = np.array([[0]])  # 상태 추정
        self.P = np.array([[1]])  # 오차 공분산 행렬

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q

    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))
        I = np.eye(self.H.shape[1])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

# EKF 초기화
Q = np.array([[0.1]])  # 노이즈 공분산
R = np.array([[10]])  # 측정 노이즈 공분산
ekf = EKF(Q, R)

# 영상 경로
video_path = 'resources/ONOFF.MOV'

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 값 저장 리스트
brightness_means = []
filtered_brightness_means = []
time_stamps = []

frame_number = 0
prev_brightness = None

plt.figure(figsize=(12, 6))

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 현재 프레임 시간 계산
    time_stamp_sec = round(frame_number / fps, 2)
    time_stamps.append(time_stamp_sec)

    # 프레임을 HSV 색 공간으로 변환
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 밝기 값(V 채널) 추출
    value_channel = hsv_frame[:, :, 2]

    # 밝기 값의 평균 계산
    mean_value = value_channel.mean()

    # EKF 예측 및 업데이트
    ekf.predict()
    ekf.update(mean_value)
    filtered_brightness = ekf.x[0, 0]

    brightness_means.append(mean_value)
    filtered_brightness_means.append(filtered_brightness)

    # 이전 밝기 값이 있는 경우 변화율 계산
    if prev_brightness is not None:
        brightness_change = filtered_brightness - prev_brightness
        # 변화율이 특정 임계값을 초과하는 경우 소등으로 간주
        if brightness_change > 100:  # 임계값 설정
            print(f'Frame {time_stamp_sec}: 소등 감지')
            plt.scatter([time_stamp_sec], [filtered_brightness], color='red', s=100)  # 기존 그래프에 원 추가

    prev_brightness = mean_value
    frame_number += 1

# 비디오 캡처 객체 해제
cap.release()

# 시간 축에 따라 원본 평균 밝기 값과 필터링된 평균 밝기 값 플롯
plt.plot(time_stamps, brightness_means, label='Original Brightness Mean', linestyle='dashed')
plt.plot(time_stamps, filtered_brightness_means, label='Filtered Brightness Mean')
plt.title('Brightness Mean Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Brightness Mean')
plt.legend()
plt.grid(True)
plt.show()
