"""
# simple saved video input test : '/src/vtest.avi'
"""

import sys
import cv2

from _path import (DIR_HOME, get_cut_dir, stop_if_none)

dir_avi = DIR_HOME + 'src\\avi\\'


# 동영상 파일로부터 cv2.VideoCapture 객체 생성
# if not cap.isOpened():
# cap = cv2.VideoCapture(0)     #Device #0 = CAMERA_ON
cap = cv2.VideoCapture(dir_avi + 'input.avi')
cap = stop_if_none(cap, message="Camera open failed!")

# 프레임 크기
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)


# 프레임 해상도, 전체 프레임수, FPS 출력
print('Frame width:', width)
print('Frame height:', height)
print('Frame count:', count)
print('FPS:', fps)

delay = round(1000 / fps)

# 매 프레임 처리 및 화면 출력
while True:
    _, frame = cap.read()
    frame = stop_if_none(frame, message="No Video Input!")

    edge = cv2.Canny(frame, 50, 150)

    cv2.imshow('frame', frame)
    cv2.imshow('edge', edge)

    if cv2.waitKey(delay) == 27:
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
