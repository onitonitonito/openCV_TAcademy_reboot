"""
# namecard test : namecard1, 2 - what's the best threshold value?
"""
print(__doc__)

import sys
import cv2

from _path import get_cut_dir
dir_home = get_cut_dir('openCV_TAcademy')

img = cv2.imread(filename = dir_home + '/src/namecard1.jpg')

if img is None:
    print('image loaing is failed!')
    sys.exit()


# dsize=destination-size = pixels x,y
# img = cv2.resize(img, (640, 480))   # WHY NOT KICK-IN?
img = cv2.resize(img, (0, 0), fx=0.4, fy=0.4)   # WHY NOT KICK-IN?
winname1 = 'imgRGB'
cv2.imshow(winname1,img)
cv2.moveWindow(winname1, x=0, y=130)


img_gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
winname2 = 'grayScale'
cv2.imshow(winname2, img_gray)
cv2.moveWindow(winname2, x=0, y=500)

# Threashold value = 130, maxVal=255 | OTSU-Algorithm
# _, img_bin = cv2.threshold(img_gray, 130, 255, cv2.THRESH_BINARY)
_, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print(f'threVal = {_}')   # threVal = 132.0 <-- OTSU -Algorithms

winname3 = 'img_bin'
cv2.imshow(winname3, img_bin)
cv2.moveWindow(winname3, x=480, y=500)

# 키입력 값을 기다린다.
while True:
    if cv2.waitKey() == 27:     # timedelay = mili-second
        break
