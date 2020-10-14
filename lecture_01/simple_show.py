"""
# IMAGE READ/SHOW : NO USE OF MATPLOTLIB
# BE CAREFUL OF RGB <-> BGR
"""
# src(RGB) -> cv2(BGR) <-> plt(RGB)

print(__doc__)

import os
import cv2
import matplotlib.pyplot as plt

from typing import List
from _path import (
                    DIR_SRC,
                    getRGB,
                    getGray,

                    get_cut_dir,
                    stop_if_none,
                )

def get_img_list(
        dir_target:str,
        exts_valid:List=['jpg','png','bmp'],
    ) -> List[str]:
    """# image filenames, like JPG, PNG, BMP"""
    imgs = [file for file in os.listdir(dir_target)
            if len(file.split('.')) > 1 and \
                file.split('.')[-1] in exts_valid]

    [print(img) for img in imgs]
    return imgs

imgs = get_img_list(DIR_SRC)

# VARIOUS IMAGE READUNG : cv2 <-> plt
# 01 CV2-object
src = cv2.imread(DIR_SRC + 'cat.bmp', cv2.IMREAD_UNCHANGED)
src = stop_if_none(src, message='Image loading failed!')

# 02 PLT-object
src2 = plt.imread(DIR_SRC + 'cat.bmp', format='RGB')
src2 = stop_if_none(src2, message='Image loading failed!')

# 03 CV2-object convert BGR to RGB formand
srcRGB = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
srcRGB = stop_if_none(srcRGB, message='Image loading failed!')

# CHECK: img type = ndarray
print('\n\n', '# CHECK: img type = ndarray')
[print(f"TYPE = {type(obj)}") for obj in [src, src2, srcRGB]]

# matplotlib = check image on window
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15,7), )

# matplotlib = check image on window
ax[0][0].imshow(src)      # cv2
ax[0][1].imshow(src2)     # plt
ax[0][2].imshow(srcRGB)   # cv2 -> RGB

# matplotlib = check image on window
ax[1][0].imshow(getRGB(src))             # cv2 -> RGB
ax[1][1].imshow(getGray(src))            # cv2 -> Gray
ax[1][2].imshow(getRGB(getGray(src)))    # cv2 -> Gray -> RGB = Gray!

plt.show()


# cv2 imshow()
cv2.imshow('src', src)   # CV2-object : CV2 -> CV2 = O.K
cv2.imshow('src2', src2) # PLT-object : PLT -> CV2 = N.G : BGR-format

cv2.waitKey()
cv2.destroyAllWindows()
