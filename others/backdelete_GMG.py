"""
# Python OpenCV 시작(2) - GMG
# 영상에서 배경제거하기 : https://bit.ly/2SUdiA7
"""
import cv2
import numpy as np

from _path import (DIR_SRC, stop_if_none)

dir_avi = DIR_SRC + 'avi_test/'
video_name = '201907-01.mp4'
# video_name = 'input.avi'

sizeRate = 0.35

# Video resource
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(dir_avi + video_name)
cap = stop_if_none(cap, message="Camera open failed!")

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
winResize = (int(width * sizeRate), int(height * sizeRate))


def main():
    getGMG(canny=0)


def getGMG(canny=0):
    """  """
    winName0 = 'GMG / canny={}'.format(canny)
    winName1 = 'Bitwise / canny={}'.format(canny)

    gmg = cv2.bgsegm.createBackgroundSubtractorGMG()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    while True:
        ret, frame = cap.read()
        frame2 = frame.copy()

        fgmask = gmg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        bitwise = cv2.bitwise_and(frame2, frame2, mask=fgmask)

        # resize
        frame = cv2.resize(frame, winResize)
        fgmask = cv2.resize(fgmask, winResize)
        bitwise = cv2.resize(bitwise, winResize)

        # canny filter
        if canny:
            fgmask = cv2.Canny(fgmask, 50, 150)
            bitwise = cv2.Canny(bitwise, 50, 150)

        cv2.imshow(winName0, fgmask)
        cv2.imshow(winName1, bitwise)
        cv2.imshow('original', frame)

        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()
