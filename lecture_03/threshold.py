import sys
import cv2

from histogram import ( getGrayHistImage, calcGrayHist )
from _path import get_cut_dir
dir_src = get_cut_dir('openCV_TAcademy') + '/src/'

filename = 'namecard1.jpg'

# 실행 시 다른 인자가 있을때는 추가인자를 기준으로 반영한다.
if len(sys.argv) > 1:
    filename = sys.argv[1]

src_gray = cv2.imread(dir_src + filename, cv2.IMREAD_GRAYSCALE)

# 이미지 로딩 실패시 : 시스템 종료
if src_gray is None:
    print('Image load failed!')
    sys.exit()

src_gray_resized = cv2.resize(src_gray, (0, 0), fx=0.5, fy=0.5)
cv2.imshow('src_gray_resized', src_gray_resized)

hist_img = getGrayHistImage(calcGrayHist(src_gray_resized))
hist_img = cv2.cvtColor(hist_img, cv2.COLOR_GRAY2BGR)
cv2.imshow('hist_img', hist_img)


def on_threshold(pos):
    _, dst = cv2.threshold(src_gray_resized, pos, 255, cv2.THRESH_BINARY)
    hist_img2 = hist_img.copy()
    cv2.line(hist_img2, (pos, 0), (pos, 100), (0, 128, 255))
    cv2.imshow('hist_img', hist_img2)
    cv2.imshow('dst', dst)


cv2.namedWindow('dst')
cv2.createTrackbar('Threshold', 'dst', 0, 255, on_threshold)
cv2.setTrackbarPos('Threshold', 'dst', 130)

cv2.waitKey()
cv2.destroyAllWindows()
