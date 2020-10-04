"""
# GOOGLE-NET Image CLASSIFICATION :
"""
print(__doc__)

import os
import sys
import cv2
import numpy as np

from _path import get_cut_dir, stop_if_none

dir_dnn = get_cut_dir('classify') + 'src_dnn\\'
dir_img = get_cut_dir('classify') + 'src_img\\'

font_color = (0, 51, 249) # BGR <- rgb(249, 51, 0)

files = [ filename
            for filename in os.listdir(dir_img)
            if filename.split('.')[-1] == 'jpg']

ask_sheets = {}
for i, filename in enumerate(files):
    ask_sheets[str(i)] = ['id', 0, filename]


model = dir_dnn + 'bvlc_googlenet.caffemodel'
config = dir_dnn + 'deploy.prototxt'
classes = dir_dnn + 'classification_classes_ILSVRC2012.txt'

# Load class names
classNames = None
with open(classes, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

def get_id_CNN(filename, model, config):

    img = cv2.imread(filename)
    if img is None:
        print("Null Object!")
        raise EnvironmentError

    # LOAD CLASS NAMES -> if net.empty():
    net = cv2.dnn.readNet(model, config)
    if net.empty():
        print('Network load failed!')
        raise EnvironmentError

    inputBlob = cv2.dnn.blobFromImage(img, 1, (224, 224), (104, 117, 123))
    net.setInput(inputBlob)
    prob = net.forward()
    out = prob.flatten()
    return img, out

def put_result(img, out, font_color=(0, 51, 249), echo=True):
    classId = np.argmax(out)
    confidence = out[classId]

    text = f"Id={classNames[classId]:} [{confidence * 100:5.2f}%]"

    cv2.putText(
                img=img,
                text=text,
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=font_color,
                thickness=2,
                # lineType=cv2.LINE_AA,
            )
    if echo:
        print(text, flush=1)
        cv2.imshow('img', img)

    cv2.waitKey()
    return classNames[classId], confidence


for key in ask_sheets.keys():
    filename = dir_img + ask_sheets[key][-1]

    img, out = get_id_CNN(filename, model, config)
    answer, conf = put_result(img, out, echo=1)
    ask_sheets[key][0], ask_sheets[key][1] = answer, conf

print('\n\n')

for idx, vals in enumerate(ask_sheets.values()):
    vals = list(vals)
    id, conf, filename = vals[0], vals[1], vals[2]

    print(f"{idx:02}. {filename:18} : {id[:29]:30} ... | {conf*100:>0.2f} %")

cv2.destroyAllWindows()




import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame.from_dict(ask_sheets).T
df.columns = ['id','conf','filename']

print('\n\n', df.describe())
print('\n\n', df.info())

# BAR 플롯이 필요하면, plot.barh() / .bar() 사용!

fig, axes = plt.subplots(2,2, figsize=(12,6))
df.conf.plot(ax=axes[0,0])
df.conf.hist(bins=20, ax=axes[0,1])
df.conf.plot.bar(ax=axes[1,0])
sns.boxplot(df.conf, ax=axes[1,1])
plt.show()

input('\n\nCONTINUE?')
