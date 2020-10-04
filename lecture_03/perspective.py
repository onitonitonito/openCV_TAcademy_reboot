"""
# perspective arrangement :
"""
print(__doc__)

from _path import (DIR_SRC, echo, get_cut_dir, stop_if_none)

import cv2
import sys
import numpy as np

src = cv2.imread(DIR_SRC + 'namecard1.jpg')
src = stop_if_none(src, message='image loaging failed!')

# 왜곡 보정후 이미지 사이즈
w, h = 720, 400

# 소스 이미지의 pts
src_quards = np.array([[325, 307], [760, 369], [718, 611], [231, 515]], np.float32)
dst_quards = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32)

pers = cv2.getPerspectiveTransform(src_quards, dst_quards)
dst = cv2.warpPerspective(src, pers, (w, h))

cv2.imshow('src', src)
cv2.imshow('dst', dst)

cv2.waitKey()
cv2.destroyAllWindows()
