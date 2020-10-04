"""
# simple video capture & edge detection
"""

import sys
import cv2
from _path import get_cut_dir

dir_home = get_cut_dir('openCV_TAcademy')


# 카메라로부터 cv2.VideoCapture 객체 생성
cap = cv2.VideoCapture(0)
video_on = cap.isOpened()

if not video_on:
    print("Camera open failed!")
    sys.exit()


# 프레임 해상도 출력
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height =  cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f'width x height = [ {width} x {height} ]')


# 새로운 프레임 사이즈로 변경
video_scale = 0.7
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width * video_scale)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height * video_scale)
cap.set(cv2.CAP_PROP_FPS, 30)


# 프레임 해상도 출력
print(f'width x height = [ {width * video_scale} x {height * video_scale} ]')



# 매 프레임 처리 및 화면 출력
while video_on:
    retVal, frame = cap.read()

    if not retVal:
        break

    edge = cv2.Canny(frame, 50, 150)

    cv2.imshow('frame', frame)
    cv2.imshow('edge', edge)

    if cv2.waitKey(10) == 27:
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
