#!/usr/bin/env python3

import cv2
import sys
import os
import keras
from keras.models import load_model
import json
import numpy as np
np.set_printoptions(suppress=True)

ws = 48
model_name = 'keras_obstacle.h5'

stats = json.load(open(os.path.join(os.getcwd(),'stats.json'), 'r'))
means = np.array(stats['means'])
stddevs = np.array(stats['stddevs'])

model = load_model(os.path.join(os.getcwd(),'saved_models',model_name))

img = cv2.imread(sys.argv[1],cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

top = 0
left = 0
occupied = np.zeros((img.shape[0] // ws,img.shape[1] // ws),dtype=np.uint8)

print('starting cn runs')
while top + ws < img.shape[0]:
    while left + ws < img.shape[1]:
        cropped = rgb[top:top + ws,left:left + ws]
        cropped = cv2.resize(cropped,(32, 32), interpolation=cv2.INTER_CUBIC)

        cropped = cropped.astype('float32') - 127.5
        cropped /= 127.5
        cropped -= means
        cropped /= stddevs
        cropped = np.expand_dims(cropped, axis=0)
        pred = model.predict(cropped)[0][0]
        pred = 0 if pred > 0.5 else 1
        occupied[top // ws][left // ws] = pred

        color = ((1-pred)*255,0,pred*255)
        cv2.rectangle(img,
        (left, top), (left + ws-2, top + ws-2),
        color, 2)

        left += ws #shift window right
    left = 0
    top += ws #shift window down
print('done cning')

cv2.imshow("image",img)
cv2.waitKey(0)
