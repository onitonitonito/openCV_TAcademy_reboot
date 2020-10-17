"""
# Python OpenCV 시작 (1) - MOG
# 영상에서 배경제거하기 : https://bit.ly/2SUdiA7
"""
import cv2
import numpy as np

from _path import (DIR_SRC, get_cut_dir, stop_if_none)

dir_avi = DIR_SRC + 'avi_test/'
video_name = '201907-02.mp4'    # scale = 0.35
sizeRate = 0.3

# 매 프레임 처리 및 화면 출력
winName0 = 'Original Image'
winName1 = 'Canny Edge Detection'

# 동영상 파일로부터 cv2.VideoCapture 객체 생성
cap = cv2.VideoCapture(dir_avi + video_name)
cap = stop_if_none(cap, message="Camera open failed!")

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
winResize = (int(width * sizeRate) , int(height * sizeRate))

mog = cv2.createBackgroundSubtractorMOG2()

frame2 = None

while True:
    ret, frame = cap.read()
    frame2 = frame.copy()

    fgmask = mog.apply(frame)
    res = cv2.bitwise_and(frame2, frame2, mask=fgmask)

    fgmask = cv2.resize(fgmask, winResize)
    fgmask = cv2.Canny(fgmask, 50, 150)

    res = cv2.resize(res, winResize)
    # res = cv2.Canny(res, 50, 150)

    frame = cv2.resize(frame, winResize)

    cv2.imshow(winName0, fgmask)
    cv2.imshow(winName1, res)
    cv2.imshow('src', frame)


    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
