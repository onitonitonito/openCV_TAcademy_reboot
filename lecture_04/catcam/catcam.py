"""
# DNN - FACE DETECTOR -> GIVES EVERY CAT-EARS on the HEAD!
# FACE-DNN = openCV_FACE_DETECTOR
"""
# RUN at CMD-Terminal mode : OK!
# Not Script-Run mode = WARNING appear
#   -> [FIX] : Script-Run = NOW OK!

print(__doc__)

import sys
import cv2
import numpy as np

from _path import get_cut_dir, stop_if_none


dir_src = get_cut_dir('openCV_TAcademy') + 'src\\'
dir_dnn = get_cut_dir('catcam') + 'src_dnn\\'
dir_img = get_cut_dir('catcam') + 'src_img\\'


model = dir_dnn + 'opencv_face_detector_uint8.pb'
config = dir_dnn + 'opencv_face_detector.pbtxt'

# IMAGE OBJECT LOAD! : IF OBJECT == NONE -> ERROR!
cat = cv2.imread(dir_img + 'ears_cat.png', cv2.IMREAD_UNCHANGED)
cat = stop_if_none(cat, message='Image open failed!')

# CAMERA OBJECT LOAD! : IF OBJECT == NONE -> ERROR!
cap = cv2.VideoCapture(0)
cap = stop_if_none(cap, message='Camera open failed!')

# if net.empty():
net = cv2.dnn.readNet(model, config)
net = stop_if_none(net, message='Net open failed!')


def overlay(frame, cat, pos) -> None:
    """# OVER-LAY CAT-EARS ON CAMERA IMAGE REAL-TIME"""
    if pos[0] < 0 or pos[1] < 0:
        return

    if pos[0] + cat.shape[1] > frame.shape[1] or pos[1] + cat.shape[0] > frame.shape[0]:
        return

    sx = pos[0]
    ex = pos[0] + cat.shape[1]
    sy = pos[1]
    ey = pos[1] + cat.shape[0]

    img1 = frame[sy:ey, sx:ex]                  # shape=(h, w, 3)
    img2 = cat[:, :, 0:3]                       # shape=(h, w, 3)
    alpha = 1. - (cat[:, :, 3] / 255.)          # shape=(h, w)
    #ww = np.stack((alpha,)*3, axis=-1)

    img1[:, :, 0] = (img1[:, :, 0] * alpha + img2[:, :, 0] * (1. - alpha)).astype(np.uint8)
    img1[:, :, 1] = (img1[:, :, 1] * alpha + img2[:, :, 1] * (1. - alpha)).astype(np.uint8)
    img1[:, :, 2] = (img1[:, :, 2] * alpha + img2[:, :, 2] * (1. - alpha)).astype(np.uint8)
    #img1 = (img1 * ww + img2 * (1. - ww)).astype(np.uint8)


while True:
    """# net = cv2.DNN.readNet(model, config)"""
    _, frame = cap.read()
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123))
    net.setInput(blob)

    detect = net.forward()
    detect = detect[0, 0, :, :]

    for i in range(detect.shape[0]):
        confidence = detect[i, 2]
        if confidence < 0.5:
            break

        x1 = int(detect[i, 3] * w + 0.5)
        y1 = int(detect[i, 4] * h + 0.5)
        x2 = int(detect[i, 5] * w + 0.5)
        y2 = int(detect[i, 6] * h + 0.5)

        fx = (x2 - x1) / cat.shape[1]
        cat2 = cv2.resize(cat, (0, 0), fx=fx, fy=fx)
        pos = (x1, y1 - (y2 - y1) // 4)
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))

        overlay(frame, cat2, pos)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
