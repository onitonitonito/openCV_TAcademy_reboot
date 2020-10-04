"""
# DNN FACE DETECTOR : CAMERA REAL-TIME
# FACE-DNN = RES10_300x300_SSD_fp16
"""
# learnopencv/res10_300x300_ssd_iter_140000_fp16.caffemodel
#  https://bit.ly/3b5ZJ8Z

print(__doc__)

import cv2
from _path import get_cut_dir, stop_if_none

dir_src = get_cut_dir('openCV_TAcademy') + 'src\\'
dir_dnn = get_cut_dir('classify') + 'src_dnn\\'
dir_img = get_cut_dir('classify') + 'src_img\\'

# model = dir_dnn + 'opencv_face_detector_uint8.pb'
# config = dir_dnn + 'opencv_face_detector.pbtxt'

model = dir_dnn + 'res10_300x300_ssd_iter_140000_fp16.caffemodel'
config = dir_dnn + 'deploy.prototxt'


# if not cap.isOpened():
cap = cv2.VideoCapture(0)
cap = stop_if_none(cap, message='Camera open failed!')

# if net.empty():
net = cv2.dnn.readNet(model, config)
net = stop_if_none(net, 'Net open failed!')

while True:
    _, frame = cap.read()
    if frame is None:
        break

    blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123))
    net.setInput(blob)
    detect = net.forward()

    detect = detect[0, 0, :, :]
    (h, w) = frame.shape[:2]

    for i in range(detect.shape[0]):
        confidence = detect[i, 2]
        if confidence < 0.5:
            break

        x1 = int(detect[i, 3] * w)
        y1 = int(detect[i, 4] * h)
        x2 = int(detect[i, 5] * w)
        y2 = int(detect[i, 6] * h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))

        label = 'Face: %4.3f' % confidence
        cv2.putText(frame, label, (x1, y1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
