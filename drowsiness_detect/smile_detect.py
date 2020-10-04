"""
# How to Build a Smile Detector. Detect Happiness in Python
"""
# by Rohan Gupta | Towards Data Science = https://bit.ly/3lRKup2
# https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
print(__doc__)

import cv2

from typing import (List, Dict)
from _path import (get_cut_dir, stop_if_none)

dir_dnn = get_cut_dir('drowsiness_detect') + 'src_dnn\\'

cascade_face = cv2.CascadeClassifier(dir_dnn + 'haarcascade_frontalface_default.xml')
cascade_eye = cv2.CascadeClassifier(dir_dnn + 'haarcascade_eye.xml')
cascade_smile = cv2.CascadeClassifier(dir_dnn + 'haarcascade_smile.xml')


def detect_face(img_gray, img_RGB) -> object:
    return img_RGB

def detect_eye(img_gray:object, img_RGB:object) -> object:
    # eyd detection &
    eyes = cascade_eye.detectMultiScale(
                    image=img_gray,
                    scaleFactor=1.2,
                    minNeighbors=18,
                )

    for (x_eye, y_eye, w_eye, h_eye) in eyes:
        cv2.rectangle(
                    img=img_RGB,
                    pt1=(x_eye, y_eye),
                    pt2=(x_eye+w_eye, y_eye+h_eye),
                    color=(0, 180, 60),     # BGR <-- rgb(60, 180, 0)
                    thickness=2,
                )
    return img_RGB

def detect_smile(img_gray:object, img_RGB:object) -> object:
    # smile detection
    smiles = cascade_smile.detectMultiScale(
                    image=img_gray,
                    scaleFactor=1.7,
                    minNeighbors=20,
                )

    for (x_smile, y_smile, w_smile, h_smile) in smiles:
        cv2.rectangle(
                    img=img_RGB,
                    pt1=(x_smile, y_smile),
                    pt2=(x_smile+w_smile, y_smile+h_smile),
                    color=(255, 0, 130),    # BGR <-- rgb(130, 0, 255)
                    thickness=2,
                )
    return img_RGB


def detection(img_gray, img_RGB):
    """#  detect faces & set rectangles on positions"""
    # face detecttion & set squares on them
    faces = cascade_face.detectMultiScale(
                    image=img_gray,
                    scaleFactor=1.3,
                    minNeighbors=5,
                )

    for (x_face, y_face, w_face, h_face) in faces:
        cv2.rectangle(
                    img=img_RGB,
                    pt1=(x_face, y_face),
                    pt2=(x_face+w_face, y_face+h_face),
                    color=(255, 130, 0),    # BGR <-- rgb(0, 130, 255)
                    thickness=2,
                )

        ri_gray = img_gray[
                    y_face:y_face + h_face,
                    x_face:x_face + w_face,]

        ri_RGB = img_RGB[
                    y_face:y_face + h_face,
                    x_face:x_face + w_face,]

        detect_eye(ri_gray, ri_RGB)
        detect_smile(ri_gray, ri_RGB)

    return img_RGB

vc = cv2.VideoCapture(0)

while True:
    _, img_RGB = vc.read()
    img_gray = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2GRAY)

    final = detection(img_gray, img_RGB)
    cv2.imshow('Video', final)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()
