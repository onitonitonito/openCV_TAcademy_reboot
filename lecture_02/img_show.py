"""
# cv2 window view :
"""

import sys
import cv2

from skimage import data
from _path import get_cut_dir

# 영상 불러오기
dir_home = get_cut_dir('openCV_TAcademy')

flag = cv2.IMREAD_COLOR        # default
flag = cv2.IMREAD_UNCHANGED    # default
# flag = cv2.IMREAD_GRAYSCALE
# flag = cv2.IMREAD_REDUCED_COLOR_2
# flag = cv2.IMREAD_REDUCED_COLOR_8

#CV2 는 BGR값으로 프린트 한다 = 주의!
# img = cv2.imread(dir_home + '/src/cat.bmp',flags=flag)
imgRGB = data.chelsea()   # nd.array - cat image
img = cv2.cvtColor(src=imgRGB, code=cv2.COLOR_RGB2BGR)


if img is None:
    print('Image load failed!')
    sys.exit()

# 영상 화면 출력
cv2.namedWindow('image')
cv2.moveWindow('image', x=900, y=0)
cv2.resizeWindow('image', width=100, height=80)  # NOT KIC-IN

cv2.imshow('image', img)
cv2.waitKey()

# 창 닫기
cv2.destroyAllWindows()
