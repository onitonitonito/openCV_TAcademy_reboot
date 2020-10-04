"""
# openCV CLOSE-EYE DETECTION : Haar-Cascade Classifier
#  - IF SCORE over 15: CLOSE
#   * ALARM RING!                   = DISABLED!
#   * CAPTURE SCENE                 = DISABLED!
#   * FRAME TURN RED! AND BLINKING! = ENABLE
# need tensorflow==2.0
"""
# [OpenCV] putText 폰트 c++ = https://bit.ly/34IOQc3
# TF higher version occure DICT issue : refer below
# https://github.com/tensorflow/tensorflow/issues/38135


import cv2
import numpy as np
import tensorflow as tf

from _path import (get_cut_dir, stop_if_none)

dir_dnn = get_cut_dir('drowsiness_detect') + 'src_dnn\\'


face = cv2.CascadeClassifier(dir_dnn + 'haarcascade_frontalface.xml')
leye = cv2.CascadeClassifier(dir_dnn + 'haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier(dir_dnn + 'haarcascade_righteye_2splits.xml')


lbl = ['CLOSED', 'OPEN']
face_x, face_y = 0, 0
detect_color=(30, 182, 15)   # BGR <- rgb(15, 182, 20)

model = tf.keras.models.load_model(dir_dnn + 'cnnCat2.h5')

count = 0
score = 0
thicc = 2

rpred, lpred = [-1], [-1]

cap = cv2.VideoCapture(0)
cap = stop_if_none(cap, "VIDEO LOAD FAILED!")

# 프레임 크기
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)


# 프레임 해상도, 전체 프레임수, FPS 출력
print('Frame width:', width)
print('Frame height:', height)
print('Frame count:', count)
print('FPS:', fps)

# font = cv2.FONT_HERSHEY_COMPLEX_SMALL
font = cv2.FONT_HERSHEY_SIMPLEX


while(True):
    _, frame = cap.read()
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(
                gray,
                minNeighbors=5,
                scaleFactor=1.1,
                minSize=(25, 25))

    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(                      # BLACK FILLED RACT.
                img=frame,
                pt1=(0, height - 50),
                pt2=(220, height),
                color=(0, 0, 0),
                thickness=cv2.FILLED)

    for (face_x, face_y, face_w, face_h) in faces:
        cv2.rectangle(
                img=frame,
                pt1=(face_x, face_y),
                pt2=(face_x + face_w, face_y + face_h),
                color=detect_color,
                thickness=4)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        count = count + 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)

        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)

        rpred = model.predict_classes(r_eye)

        lbl = 'OPEN' if(rpred[0] == 1) else 'CLOSED'
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        count = count + 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)

        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)

        lpred = model.predict_classes(l_eye)

        lbl = 'OPEN' if(lpred[0] == 1) else 'CLOSED'
        break

    if(rpred[0] == 0 and lpred[0] == 0):
        detect_color = (27, 27, 224)        # BGR <- rgb(224, 27, 27)
        detect_text = 'CLOSED'
        if score <= 30:
            score += 1
        else:
            score = 40
    else:
        detect_color=(30, 182, 15)          # BGR <- rgb(15, 182, 30)
        detect_text = 'OPEN'
        if score >= 1:
            score -= 1
        else:
            score = 0

    cv2.putText(
                img=frame,
                text=detect_text,
                org=(face_x, face_y-10),      # Bottom-left coord
                fontFace=font,
                fontScale=0.9,
                color=detect_color,
                thickness=2,
                lineType=cv2.LINE_AA)

    cv2.putText(
                img=frame,
                text="Warn Level: "+ str(score),
                org=(10, height - 15),
                fontFace=font,
                fontScale=0.7,
                color=(255, 255, 255),       # BGR <-
                thickness=2,
                lineType=cv2.LINE_AA)

    if(score > 15):                   # IF SLEEPY
        # FRAME BLINKING!
        if(thicc < 25):
            thicc = thicc + 4
        else:
            thicc = thicc - 4
            if(thicc < 2):
                thicc = 2

        cv2.rectangle(
                    img=frame,
                    pt1=(0, 0),
                    pt2=(width, height),
                    color=(226, 0, 238),       # BGR <- rgb(238, 0, 226)
                    thickness=thicc)

    cv2.imshow('frame : Haar-Casecade Detect', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# CLEAN & REMOVE DEVICE
cap.release()
cv2.destroyAllWindows()
