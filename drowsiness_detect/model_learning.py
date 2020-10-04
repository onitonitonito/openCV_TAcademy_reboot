"""
# model training! 
# tensor_flow version = keras -> 2.2 over
"""

import os
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

from keras.models import load_model
from keras.models import Sequential
from keras.layers import (
                            Dropout,
                            Conv2D,
                            Flatten,
                            Dense,
                            MaxPooling2D,
                            BatchNormalization,
                        )

from _path import (get_cut_dir, stop_if_none)

dir_dnn = get_cut_dir('drowsiness_detect') + 'src_dnn\\'


def generator(
        dir,
        gen=image.ImageDataGenerator(rescale=1./255),
        shuffle=True,
        batch_size=1,
        target_size=(24,24),
        class_mode='categorical',
    ):

    return gen.flow_from_directory(
                    dir,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    color_mode='grayscale',
                    class_mode=class_mode,
                    target_size=target_size,
                )

BS, TS = 32, (24,24)

train_batch= generator(
                        'data/train',
                        shuffle=True,
                        batch_size=BS,
                        target_size=TS,
                    )

valid_batch= generator(
                        'data/valid',
                        shuffle=True,
                        batch_size=BS,
                        target_size=TS
                    )

SPE= len(train_batch.classes)//BS
VS = len(valid_batch.classes)//BS

print(SPE,VS)


# img,labels= next(train_batch)
# print(img.shape)


#32-conv.filters to each 3x3 again
#64-conv-filters to each 3x3, pick best feat. via pooling
# many dimensions, we only want a classification output
#fully connected to get all relevant data
#one more dropout for convergence' sake :)
#output a softmax to squash the matrix into output probabilities

model = Sequential([
        Conv2D(
                32,
                kernel_size=(3, 3),
                activation='relu',
                input_shape=(24,24,1)),
        MaxPooling2D(pool_size=(1,1)),

        Conv2D(
                32,
                (3,3),
                activation='relu'),
        MaxPooling2D(pool_size=(1,1)),

        Conv2D(
                64,
                (3, 3),
                activation='relu'),
        MaxPooling2D(pool_size=(1,1)),

        Dropout(0.25),
        Flatten(),

        Dense(128, activation='relu'),
        Dropout(0.5),

        Dense(2, activation='softmax')
    ])

model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

model.fit_generator(
                        train_batch,
                        validation_data=valid_batch,
                        epochs=15,
                        steps_per_epoch=SPE ,
                        validation_steps=VS
                    )

model.save(dir_dnn + 'cnnCat2-1.h5', overwrite=True)
