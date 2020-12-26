"""
# Mouse Demo - simple
# https://bit.ly/38iiHcS
"""
# 간단한 Demo
# 아래 Demo는 화면에 Double-Click을 하면 원이 그려지는 예제.

import cv2
import numpy as np


def draw_circle(event, x, y, flags, param):
    """# callback함수"""
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x, y), 100, (255, 0, 0), -1)

# 빈 Image 생성
img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while(1):
    cv2.imshow('image', img)
    if cv2.waitKey(0) & 0xFF == 27:
        break

cv2.destroyAllWindows()
