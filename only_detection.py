import cv2
import numpy as np

# EKF 필터 클래스
class EKF:
    def __init__(self, Q, R):
        self.F = np.array([[1]])
        self.H = np.array([[1]])
        self.Q = Q
        self.R = R
        self.x = np.array([[125]])  # 초기 상태 추정값
        self.P = np.array([[1]])  # 초기 오차 공분산 행렬

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q

    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))
        I = np.eye(self.H.shape[1])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

# 카메라 초기화 함수
def cam_init(cam_path=0):
    cap = cv2.VideoCapture(cam_path)
    if not cap.isOpened():
        print("카메라 없음. 연결 여부나 권한 확인하기!")
        exit()
    return cap

# 실시간 밝기 변화 감지 함수
def detect_brightness_change(cap, ekf, sensitivity):
    filtered_brightness_means = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # HSV변환 및 V 채널
        value = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
        mean_brightness = np.mean(value)

        # EKF 업데이트
        ekf.predict()
        ekf.update(mean_brightness)
        filtered_brightness = ekf.x[0, 0]
        ekf_error = abs(filtered_brightness - mean_brightness)
        print(f'평균 밝기: {int(mean_brightness)}, EKF 평균 밝기: {int(filtered_brightness)}, EKF ERROR: {int(ekf_error)}')

        # 소등 및 점등 감지
        if len(filtered_brightness_means) > 0:
            brightness_change = filtered_brightness - filtered_brightness_means[-1]

            if brightness_change < -sensitivity:
                print(f'\033[93m 소등 감지 \033[0m')

            if brightness_change > sensitivity:
                print(f'\033[96m 점등 감지 \033[0m')

        filtered_brightness_means.append(filtered_brightness)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cap = cam_init(0)

    # 민감도
    sensitivity = 20  # 얼마나 급격한 변화를 감지할 것인가 ?
    Q = np.array([[.5]])  # 실제값 변화에 민감해짐! (노이즈 공분산)
    R = np.array([[10]])  # 필터 예측 모델을 더 신뢰함. (측정 노이즈 공분산)
    ekf = EKF(Q, R)

    detect_brightness_change(cap, ekf, sensitivity)
