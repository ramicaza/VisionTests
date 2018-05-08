#!/usr/bin/env python3

import cv2
import sys
import os
import keras
from keras.models import load_model
import numpy as np
np.set_printoptions(suppress=True)

model_name = 'keras_lane_v3.h5'
model = load_model(os.path.join(os.getcwd(),'saved_models',model_name))

def moused(event, x, y, flags, param):
    global img
    s = 64//2
    if event == cv2.EVENT_MOUSEMOVE:
        cpy = img.copy()
        # cv2.rectangle(cpy, (x-s, y-s), (x+s, y+s), (255,0,0), 2)
        # cv2.imshow("image",cpy)

        cropped = img[y-s:y+s,x-s:x+s]
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        cropped = cv2.resize(cropped,(64, 64), interpolation=cv2.INTER_CUBIC)
        cropped = cropped.astype('float32') - 127.5
        cropped /= 127.5
        cropped = np.expand_dims(cropped, axis=0)
        pred = model.predict(cropped)[0]
        print(pred[1] > pred[0])

img = cv2.imread(sys.argv[1],cv2.IMREAD_COLOR)
cv2.imshow("image",img)

cv2.setMouseCallback("image", moused)
cv2.waitKey(0)


# import data
# d = data.Data().get_partitions(0.7,0.2,0.1)
# (x_train, y_train), (x_valid, y_valid), (x_test,y_test) = d
# y_test = keras.utils.to_categorical(y_test, 2)
# x_test = x_test.astype('float32')
# x_test /= 255
# print(x_test.shape)
# print(y_test.shape)
# scores = model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])
