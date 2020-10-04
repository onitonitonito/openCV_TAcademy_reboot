"""
# ■ 미리 학습된 GoogLeNet 학습 모델 및 구성 파일 다운로드
#  모델 파일 (Size = 52 MB)
#   http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
"""
# TODO: to be modified : NOT MODIF YET,

#  구성 파일 : 아래 URL 에서 deploy.prototxt 파일 다운로드
#   https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet

#  클래스 이름 파일(1~1000 번 클래스에 대한 설명을 저장한 텍스트 파일)
#   https://github.com/opencv/opencv/blob/4.1.0/samples/data/dnn/classification_classes_ILSVRC2012.txt

import sys
import cv2
import numpy as np


filename = 'space_shuttle.jpg'

if len(sys.argv) > 1:
    filename = sys.argv[1]

img = cv2.imread(filename)

if img is None:
    print('Image load failed!')
    exit()

# Load network

net = cv2.dnn.readNet('bvlc_googlenet.caffemodel', 'deploy.prototxt')

if net.empty():
    print('Network load failed!')
    exit()

# Load class names

classNames = None
with open('classification_classes_ILSVRC2012.txt', 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Inference

inputBlob = cv2.dnn.blobFromImage(img, 1, (224, 224), (104, 117, 123))
net.setInput(inputBlob, 'data')
prob = net.forward()

# Check results & Display

out = prob.flatten()
classId = np.argmax(out)
confidence = out[classId]

text = '%s (%4.2f%%)' % (classNames[classId], confidence * 100)
cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()
