"""
# IMAGE LABELLING : COIN-LABELLING -
"""
# dst = destination_image

import sys
import random
import numpy as np
import cv2

from _path import get_cut_dir

dir_src = get_cut_dir('openCV_TAcademy') + '/src/'
src = cv2.imread(dir_src + 'coins.png', cv2.IMREAD_GRAYSCALE)

# 이미지 로딩 실패시 시스템 종료!
if src is None:
    print('Image load failed!')
    sys.exit()

# 원본 이미지의 크기
h, w = src.shape[:2]

# 같은 사이즈의 빈 이미지 2개 생성
dst1 = np.zeros((h, w, 3), np.uint8)
dst2 = np.zeros((h, w, 3), np.uint8)

# 전처리
src = cv2.blur(src, (3, 3))
_, src_bin = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)

# 레이블링
cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(src_bin)
for i in range(1, cnt):
    c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    dst1[labels == i] = c

# 외곽선 검출
contours, _ = cv2.findContours(src_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

for i in range(len(contours)):
    c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    cv2.drawContours(dst2, contours, i, c, 1)

cv2.imshow('src', src)
cv2.imshow('src_bin', src_bin)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()
