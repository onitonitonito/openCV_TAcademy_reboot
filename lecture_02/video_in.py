"""
# simple saved video input test : '/src/vtest.avi'
"""

import sys
import cv2

from _path import (DIR_SRC, get_cut_dir, stop_if_none)

dir_avi = DIR_SRC + 'avi_test/'

# video_name = 'input.avi'    # avi, mp4, video_clip
list_video = (
    '201907-03.mp4',
    "2021-08-12-transfer-test-SVGA.mp4",
    "2021-08-27-discharge-all-SVGA.mp4",
    "2021-08-27-rubber-sheet-SVGA.mp4",
)

video_name = list_video[0]    # avi, mp4, video_clip

landscape = 0              # 0 = Portrait / 1 = Landscape
sizeRate = 0.7

# 동영상 파일로부터 cv2.VideoCapture 객체 생성

cap = cv2.VideoCapture(dir_avi + video_name)
cap = stop_if_none(cap, message="Camera open failed!")

if not cap.isOpened():
    cap = cv2.VideoCapture(0)     # Device #0 = CAMERA_ON

# 프레임 크기
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)

winResize = (int(width * sizeRate) , int(height * sizeRate))

if landscape:
    moveTo = (0, winResize[1]+30)
else:
    moveTo = (winResize[0]+5, 0)


# 프레임 해상도, 전체 프레임수, FPS 출력
print('Frame width:', width)
print('Frame height:', height)
print('Frame count:', count)
print('FPS:', fps)

delay = round(1000 / fps)

# 매 프레임 처리 및 화면 출력
winName0 = 'Original Image'
winName1 = 'Canny Edge Detection'


while True:
    _, frame = cap.read()
    frame = stop_if_none(frame, message="No Video Input!")
    frame = cv2.resize(frame, winResize)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    edge = cv2.Canny(frame, 50, 150)
    edge = cv2.resize(edge, winResize)

    cv2.imshow(winName0, frame)
    cv2.moveWindow(winName0, *moveTo)

    cv2.imshow(winName1, edge)
    cv2.moveWindow(winName1, 0, 0)

    key = cv2.waitKey(delay)
    if key == 27:   # ESC-key
        break

    #wait until any key is pressed
    if key == ord('p'):
        cv2.waitKey(-1)

# 자원 해제
cap.release()
cv2.destroyAllWindows()
