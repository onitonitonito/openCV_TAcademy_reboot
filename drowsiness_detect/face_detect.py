"""
# Face Recognition by OpenCV-Python (py-2.7)
# urllib.request.urlopen('http://216.58.192.142',timeout=1)
# https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml = https://bit.ly/3jFEy0v
#"""
print(__doc__)

import os
import cv2
import urllib.request
import matplotlib.pyplot as plt
import numpy as np

from _path import (get_cut_dir, stop_if_none)

dir_dnn = get_cut_dir('drowsiness_detect') + 'src_dnn\\'
describe_str = "Here, 'Haar Cascade' Founds '{0}' faces!"

model = 'haarcascade_frontalface.xml'
model_dir = dir_dnn + model

IMG_URL = "https://bit.ly/3hYctB3"    # telegraph - ski resort
IMG_URL = 'https://bit.ly/3509nsL'    # a baby with beard
IMG_URL = 'https://bit.ly/3gUZIWJ'    # many faces inc'l a beard man


def main():
    check_model_exist(model, dir_dnn)
    find_faces(IMG_URL)

def check_model_exist(model, dir_dnn, echo=1):
    if model in os.listdir(dir_dnn):
        download = False
    else:
        # download harr-cascade.xml from githib.com - haarcascades
        download = True

        xmldata = urllib.request.urlopen('https://bit.ly/3jFEy0v').read()
        with open(model_dir,'wb') as f:
            f.write(xmldata)

    if download:
        if echo:
            print("Haar Cascade.xml download = O.K!")
            print(f"designated dir : {dir_dnn}")
        return True
    else:
        if echo:
            print(f"{model} is already exist on dir_dnn")
        return False

def find_faces(full_url_to_image):
    f = urllib.request.urlopen(full_url_to_image)

    face_cascade = cv2.CascadeClassifier(model_dir)

    image = np.asarray(bytearray(f.read()), dtype="uint8")
    image = stop_if_none(image, message='image loading error!')

    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
                                gray,
                                scaleFactor=1.1,
                                minNeighbors=5,
                                minSize=(30, 30),
                                flags=cv2.CASCADE_SCALE_IMAGE
                            )

    print(describe_str.format(len(faces)))

    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(
                        img=image,
                        pt1=(x, y),
                        pt2=(x + w, y + h),
                        color=(9, 245, 50),   # BGR <-- rgb(50, 245, 9)
                        thickness=3,
                    )

        cv2.putText(
                        img=image,
                        text=f"{i+1}",
                        org=(x , y-10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(9, 102, 249), # BGR = rgb(249, 102, 9)
                        thickness=3,
                        )

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image, extent=[300, 500, 0, 1], aspect='auto')
    plt.title(describe_str.format(len(faces)), size=20, fontweight=10)
    plt.grid(False)
    plt.axis('off')
    plt.show()




if __name__ == '__main__':
    main()
