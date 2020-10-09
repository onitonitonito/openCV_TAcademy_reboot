"""
# simple saved video input test : '/src/vtest.avi'
"""

import sys
import cv2

from _path import (DIR_SRC, get_cut_dir, stop_if_none)

dir_avi = DIR_SRC + 'avi/'
# video_name = '201907-01.mp4'    # stir-separate
# video_name = '201907-02.mp4'
video_name = '201907-03.mp4'
# video_name = '202006-01.mp4'


# 동영상 파일로부터 cv2.VideoCapture 객체 생성
# if not cap.isOpened():
# cap = cv2.VideoCapture(0)     #Device #0 = CAMERA_ON

cap = cv2.VideoCapture(dir_avi + video_name)
# cap = stop_if_none(cap, message="Camera open failed!")

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
winName0 = 'Original Image'
winName1 = 'Canny Edge Detection'

sizeRate = 0.43
winResize = (int(width * sizeRate) , int(height * sizeRate))
moveTo = (0, winResize[1]+70)

while True:
    _, frame = cap.read()
    frame = stop_if_none(frame, message="No Video Input!")
    frame = cv2.resize(frame, winResize)   

    edge = cv2.Canny(frame, 50, 150)
    edge = cv2.resize(edge, winResize)   

    cv2.imshow(winName0, frame)
    cv2.moveWindow(winName0, 0,0)

    cv2.imshow(winName1, edge)
    cv2.moveWindow(winName1, *moveTo)

    if cv2.waitKey(delay) == 27:
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
