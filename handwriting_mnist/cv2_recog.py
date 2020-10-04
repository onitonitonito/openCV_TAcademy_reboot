"""
# Keras와 OpenCV를 사용하여 손글씨 숫자 인식하기
# 손글씨 숫자 인식하기 ::
# 멈춤보단 천천히라도 - 작성 2020. 3. 30
"""
# = https://bit.ly/3lAYL9u

print(__doc__)

import cv2
import numpy as np

from tensorflow.keras.models import load_model
from _path import (get_cut_dir, stop_if_none)

# filename = 'hand-writing-00.jpg'     # orig - 1,083 kb
filename = 'hand-writing-01.jpg'     # LOW  -   256 kb
# filename = 'hand-writing-02.jpg'     # HIGH -   858 kb

model_file = 'model.h5'

dir_src = get_cut_dir('handwriting_mnist') + 'src\\'


img_color = cv2.imread(dir_src + filename, cv2.IMREAD_COLOR)
img_color = stop_if_none(img_color, message='image loading is failed!')

img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)


_,img_binary = cv2.threshold(
                            img_gray,
                            0,
                            255,
                            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
                        )

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ( 5, 5 ))

img_binary = cv2.morphologyEx(
                            img_binary,
                            cv2.MORPH_DILATE,
                            kernel,
                        )

cv2.imshow('digit', img_binary)
cv2.imwrite(dir_src + 'binary-' + filename, img_binary)

cv2.waitKey()

contours, hierarchy = cv2.findContours(
                            img_binary,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )

for contour in contours:
    # 너무 작은 객체는 제외 = 면적이 1,000픽셀보다 작은 객체는 제외
    if cv2.contourArea(contour) < 1000:
        continue

    x, y, w, h = cv2.boundingRect(contour)

    length = max(w, h) + 60
    img_digit = np.zeros((length, length, 1), np.uint8)

    new_x = x-(length - w)//2
    new_y = y-(length - h)//2


    img_digit = img_binary[new_y:new_y+length, new_x:new_x+length]

    kernel = np.ones((5, 5), np.uint8)

    img_digit = cv2.morphologyEx(
                            img_digit,
                            cv2.MORPH_DILATE,
                            kernel,
                        )

    # cv2.imshow('digit-1', img_digit)
    # cv2.waitKey()

    model = load_model(dir_src + model_file)

    img_digit = cv2.resize(
                            img_digit,
                            (28, 28),
                            interpolation=cv2.INTER_AREA,
                        )

    img_digit = img_digit / 255.0

    img_input = img_digit.reshape(1, 28, 28, 1)

    predictions = model.predict(img_input)
    number = np.argmax(predictions)

    print(number)

    cv2.rectangle(
                    img_color,
                    (x, y),
                    (x+w, y+h),
                    (255, 255, 0),
                    2,
                )


    location = (x + int(w *0.5), y - 10)
    font = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 1.2
    cv2.putText(
                    img_color,
                    str(number),
                    location,
                    font,
                    fontScale,
                    (0,255,0),
                    2
                )


    # cv2.imshow('digit-2', img_digit)
    # cv2.waitKey()


cv2.imshow('result', img_color)
cv2.imwrite(dir_src + 'result-' + filename, img_color)
cv2.waitKey()
