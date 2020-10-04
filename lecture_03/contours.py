"""
# extract out-line of namecard1,2 :
"""
# dest = destination

import sys
import random
import numpy as np
import cv2

from _path import get_cut_dir
dir_home = get_cut_dir('openCV_TAcademy')

# NC1=어두운사진 / NC2=밝은사진
src = cv2.imread(dir_home + '/src/namecard1.jpg')

# 이미지 로딩이 안되었으면 = 실패! 종료
if src is None:
    print('Image load failed!')
    sys.exit()

# 명함 이미지 소스를 만든다 = Resize
src = cv2.resize(src, (0, 0), fx=0.5, fy=0.5)
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

h, w = src_gray.shape[:2]
dst1 = np.zeros((h, w, 3), np.uint8)
dst2 = np.zeros((h, w, 3), np.uint8)

# 이진화
_, src_bin = cv2.threshold(src_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 외곽선 검출 옵션
contours1, _ = cv2.findContours(src_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours2, _ = cv2.findContours(src_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

for i in range(len(contours1)):
    c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    cv2.drawContours(dst1, contours1, i, c, 1)

for i in range(len(contours2)):
    c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    cv2.drawContours(dst2, contours2, i, c, 1)

cv2.imshow('src_resize', src)
cv2.imshow('src_binary', src_bin)
cv2.imshow('destination-01 :', dst1)
cv2.imshow('destination-02 :', dst2)
cv2.waitKey()

cv2.destroyAllWindows()
