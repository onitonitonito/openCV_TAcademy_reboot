"""
# IMAGE PROCESSING BEFORE OCR : SELF
"""
print(__doc__)

import PIL
import cv2

from PIL import ImageDraw
from _path import DIR_HOME, get_cut_dir, stop_if_none

dir_read = DIR_HOME + 'src\\readOCR\\'
dir_result = DIR_HOME + 'src\\resultOCR\\'

filename = 'books.jpg'
post_fix = 'GrayResize'
name = filename.split('.')[0]
degrees = [0, 90, 180, 270]

im = cv2.imread(filename=dir_read + filename)
print('ORIGINAL=', im.shape, '\n\n')

im = cv2.cvtColor(src=im, code=cv2.COLOR_BGR2GRAY)
im = cv2.resize(src=im, dsize=(0,0), fx=0.5, fy=0.5)



cv2.imwrite(dir_read + f"{name}{post_fix}{degrees[0]:03}.jpg", im)
print(f"  0 degree-SHAPE=", im.shape)


for deg in degrees[1:]:
    im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)

    # 저장할 파일 Type : JPEG, PNG 등 / 저장할 때 Quality 수준 : 보통 95 사용
    cv2.imwrite(dir_read + f"{name}{post_fix}{deg:03}.jpg", im)
    print(f"{deg:3} degree-SHAPE=", im.shape)


cv2.imshow('im', im)
cv2.waitKey()


cv2.destroyAllWindows()
