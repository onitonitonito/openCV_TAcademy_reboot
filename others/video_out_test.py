"""
# simple video writing : '/src/output.avi'
"""
import sys
import cv2

from _path import get_cut_dir

dir_home = get_cut_dir('openCV_TAcademy')


# 카메라로부터 cv2.VideoCapture 객체 생성
cap = cv2.VideoCapture(0)
video_on = cap.isOpened()

if not video_on:
    print("Camera open failed!")
    sys.exit()

# 동영상 저장을 위한 cv2.VideoWriter 객체 생성
w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# *'DIVX' == 'D','I','V','X'
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

out = cv2.VideoWriter(dir_home + '/src/avi/output.avi', fourcc, fps, (w, h))

# 매 프레임 처리 및 화면 출력
while True:
    ret, frame = cap.read()

    if not ret:
        break

    edge = cv2.Canny(frame, 50, 150)
    edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    out.write(edge)

    cv2.imshow('frame', frame)
    cv2.imshow('edge', edge)

    if cv2.waitKey(10) == 27:
        break

# 자원 해제

cap.release()
out.release()
cv2.destroyAllWindows()
