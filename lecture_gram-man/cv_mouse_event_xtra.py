"""
# Advanced Mouse Demo
# https://bit.ly/38iiHcS
"""
# 다음은 마우스를 누른 상태에서 이동시 원 또는 사각형을 그리는 Demo입니다.
# m = mode change / 사각형 <--> 원 모드 변경

# 이 예제는 향후 대상 추적이나 이미지 Segmentaion시 응용될 수 있습니다.
# (ex; 이미지에서 대상을 마우스로 선택하고 동일한 대상을 찾는 경우)

import cv2
import numpy as np

drawing = False  # Mouse가 클릭된 상태 확인용
mode = True     # True이면 사각형, false면 원
ix, iy = -1, -1


def draw_circle(event, x, y, flags, param):
    """# Mouse Callback함수"""
    global ix, iy, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스를 누른 상태
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:  # 마우스 이동
        if drawing == True:            # 마우스를 누른 상태 일경우
            if mode == True:
                cv2.rectangle(img, (ix, iy), (x, y), (255, 0, 0), -1)
            else:
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False             # 마우스를 때면 상태 변경
        if mode == True:
            cv2.rectangle(img, (ix, iy), (x, y), (255, 0, 0), -1)
        else:
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)


img = np.zeros((512, 512, 3), np.uint8)

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF

    if k == ord('m'):    # 사각형, 원 Mode변경
        mode = not mode

    elif k == 27:        # esc를 누르면 종료
        break

cv2.destroyAllWindows()
