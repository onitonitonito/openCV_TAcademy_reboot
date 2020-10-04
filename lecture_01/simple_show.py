"""
# IMAGE READ/SHOW : NO USE OF MATPLOTLIB
# BE CAREFUL OF RGB <-> BGR
"""
# src(RGB) -> cv2(BGR) <-> plt(RGB)

print(__doc__)

import cv2
import matplotlib.pyplot as plt
from _path import (DIR_SRC, get_cut_dir, stop_if_none)

# VARIOUS IMAGE READUNG : cv2 <-> plt
# 01 CV2-object
src = cv2.imread(DIR_SRC + 'cat.bmp', cv2.IMREAD_UNCHANGED)
src = stop_if_none(src, message='Image loading failed!')

# 02 PLT-object
src2 = plt.imread(DIR_SRC + 'cat.bmp', format='RGB')
src2 = stop_if_none(src2, message='Image loading failed!')

# 03 CV2-object convert BGR to RGB formand
srcRGB = cv2.cvtColor(src2, cv2.COLOR_BGR2RGB)
srcRGB = stop_if_none(srcRGB, message='Image loading failed!')


# CHECK: img type = ndarray
[print(f"TYPE = {type(obj)}") for obj in [src, src2, srcRGB]]

# cv2 imshow()
cv2.imshow('src', src)   # CV2-object : CV2 -> CV2 = O.K
cv2.imshow('src2', src2) # PLT-object : PLT -> CV2 = N.G : BGR-format

cv2.waitKey()
cv2.destroyAllWindows()

# matplotlib = check image on window
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,7), )
ax.imshow(src)
plt.show()
