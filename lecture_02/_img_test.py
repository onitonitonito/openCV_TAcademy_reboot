"""
# simple img type test :
"""
# https: // docs.opencv.org/master/d4/da8/group__imgcodecs.html
# # ga288b8b3da0892bd651fce07b3bbd3a56

print(__doc__)

import cv2
from _path import get_cut_dir

dir_home = get_cut_dir('openCV_TAcademy')
img = cv2.imread(dir_home + '/src/cup.jpg')


# 속성 확인
print('type =', type(img))
print('shape=', img.shape)
print('dtype=', img.dtype)

# 사진 속성설정
img_resize = cv2.resize(src=img, dsize=(0,0), fx=0.5, fy=0.5)
cv2.imshow(winname='image', mat=img_resize)
cv2.moveWindow(winname='image',x=100, y=300)

# cv2.resizeWindow(winname='image', width=200, height=100)

# mil-second = 2000 (2sec), default = 0 : infinire
while True:
  if cv2.waitKey() == 27:    # 27 = ASC(ESC)
    break

# 창 닫기 = 생략해도 된다.
cv2.destroyAllWindows()
