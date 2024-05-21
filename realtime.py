import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


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


# 애니메이션 업데이트 함수 (main)
def ani(frame, cap, img_display, line, orig_line, error_line, ekf, timestamps, sensitivity):
    ret, frame = cap.read()
    if not ret:
        return img_display, line, orig_line, error_line

    # matplotlib 영상출력용 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_display.set_data(frame_rgb)

    # HSV변환 및 V 채널
    value = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]
    mean_brightness = np.mean(value)

    # EKF 업데이트
    ekf.predict()
    ekf.update(mean_brightness)
    filtered_brightness = ekf.x[0, 0]
    ekf_error = abs(filtered_brightness - mean_brightness)
    print(f'평균 밝기: {int(mean_brightness)}, EKF 평균 밝기: {int(filtered_brightness)}, EKF ERROR: {int(ekf_error)}')

    y_data = line.get_ydata()
    y_data = np.append(y_data, filtered_brightness)[-100:]  # 100개까지만
    line.set_ydata(y_data)

    orig_y_data = orig_line.get_ydata()
    orig_y_data = np.append(orig_y_data, mean_brightness)[-100:]
    orig_line.set_ydata(orig_y_data)

    error_y_data = error_line.get_ydata()
    error_y_data = np.append(error_y_data, ekf_error)[-100:]
    error_line.set_ydata(error_y_data)

    x_data = np.arange(len(y_data))
    line.set_xdata(x_data)
    orig_line.set_xdata(x_data)
    error_line.set_xdata(x_data)

    time_stamp_sec = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 2)
    timestamps.append(time_stamp_sec)

    # 소등 및 점등 감지
    if len(filtered_brightness_means) > 1:
        brightness_change = filtered_brightness_means[-1] - filtered_brightness

        if brightness_change > sensitivity:
            print(f'\033[93m 소등 감지 \033[0m')

        if brightness_change < -sensitivity:
            print(f'\033[96m 점등 감지 \033[0m')

    filtered_brightness_means.append(filtered_brightness)

    return img_display, line, orig_line, error_line


if __name__ == "__main__":
    cap = cam_init(0)

    # 소등 및 점등 감지
    # EKF 초기화
    # 민감도
    # FIXME: 파라미터입니다.
    #   Q만 다루어도 충분히 그래프의 변화를 확인할 수 있으며, 필요에 따라 R값도 적절하게 수정하시면 됩니다.
    #   sensitivity는 얼마나 큰 변화를 감지할 것인가? 에 대한 내용입니다.
    sensitivity = 20  # 얼마나 급격한 변화를 감지할 것인가 ?
    Q = np.array([[.5]])  # 실제값 변화에 민감해짐! (노이즈 공분산) 소수점 변화에도 민감하게 반응합니다!!!
    R = np.array([[10]])  # 필터 예측 모델을 더 신뢰함. (측정 노이즈 공분산)
    ekf = EKF(Q, R)

    fig, (img_ax, graph_ax) = plt.subplots(1, 2, figsize=(10, 5))

    # 실시간 영상 객체
    img_ax.set_title("Live Video")
    img_ax.axis('off')  # Turn off axis
    img_display = img_ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))

    # 평균 밝기 그래프
    graph_ax.set_title("Mean Brightness")
    graph_ax.set_xlim(0, 100)
    graph_ax.set_ylim(0, 256)

    line, = graph_ax.plot(np.zeros(100), label='Filtered Brightness')
    orig_line, = graph_ax.plot(np.zeros(100), label='Original Brightness', linestyle='dashed')
    error_line, = graph_ax.plot(np.zeros(100), label='EKF Error', linestyle='dotted')

    graph_ax.legend()

    # x축 눈금과 레이블 제거
    graph_ax.xaxis.set_ticks([])
    graph_ax.xaxis.set_ticklabels([])

    timestamps = []
    filtered_brightness_means = []

    ani_obj = animation.FuncAnimation(fig, ani, fargs=(
    cap, img_display, line, orig_line, error_line, ekf, timestamps, sensitivity), interval=50, blit=True,
                                      cache_frame_data=False)
    plt.show()

    cap.release()
    cv2.destroyAllWindows()
