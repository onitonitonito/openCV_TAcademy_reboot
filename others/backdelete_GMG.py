"""
# Python OpenCV 시작(2) - GMG
# 영상에서 배경제거하기 : https://bit.ly/2SUdiA7
"""

import cv2
import numpy as np

from _path import (DIR_SRC, get_cut_dir, stop_if_none)

dir_avi = DIR_SRC + 'avi_test/'
video_name = 'input.avi'
sizeRate = 0.8

# Video resource
# cap = cv2.VideoCapture(0)

# 동영상 파일로부터 cv2.VideoCapture 객체 생성
cap = cv2.VideoCapture(dir_avi + video_name)
cap = stop_if_none(cap, message="Camera open failed!")



def getMOG():
    """  """
    mog = cv2.createBackgroundSubtractorMOG2()

    frame2 = None
    while True:
        ret, frame = cap.read()

        frame2 = frame.copy()
        fgmask = mog.apply(frame)
        cv2.imshow('result',fgmask)
        res = cv2.bitwise_and(frame2,frame2,mask=fgmask)
        cv2.imshow('res',res)
        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def getGMG():
    """  """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

    while True:
        ret, frame = cap.read()
        frame2 = frame.copy()

        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask,cv2.MORPH_OPEN, kernel)

        cv2.imshow('mask',fgmask)
        res = cv2.bitwise_and(frame2,frame2,mask=fgmask)
        cv2.imshow('res',res)
        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# getGMG()
getMOG()
