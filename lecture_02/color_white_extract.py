"""
#  https://stackoverflow.com/questions/22588146/tracking-white-color-using-python-opencv
#
"""
import sys
import cv2
import numpy as np
from _path import (DIR_SRC, get_cut_dir, stop_if_none)

dir_avi = DIR_SRC + 'avi_test/'

# video_name = 'input.avi'    # avi, mp4, video_clip
list_video = (
    '201907-03.mp4',
    "2021-08-12-transfer-test-SVGA.mp4",
    "2021-08-27-discharge-all-SVGA.mp4",
    "2021-08-27-rubber-sheet-SVGA.mp4",
)

video_name = list_video[0]    # avi, mp4, video_clip

cap = cv2.VideoCapture(dir_avi + video_name)
cap = stop_if_none(cap, message="Camera open failed!")

while(1):
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of white color in HSV
    # change it according to your need !
    lower_white = np.array([0,0,50], dtype=np.uint8)
    upper_white = np.array([120,120,180], dtype=np.uint8)

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask= mask)

    cv2.imshow('frame',frame)
    # cv2.imshow('mask',mask)
    cv2.imshow('res',res)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()