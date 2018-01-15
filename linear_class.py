#!/usr/bin/env python

import cv2
import numpy as np
from pprint import pprint
from sklearn.linear_model import SGDClassifier

X = []
Y = []
in_train = True
in_pos = True
clf = SGDClassifier()

def add_instance(features):
    X.append(features)
    Y.append(in_pos)

if __name__ == "__main__":
    def moused(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            features = hsv[y][x]
            if in_train:
                add_instance(np.copy(features))
            else:
                print(features)
                print(clf.predict(features.reshape(1,3)))
            cv2.circle(img, (x,y), 5, (0,0,255), -1)
            cv2.imshow("image", img)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", moused)
    camera = cv2.VideoCapture(0)
    while True:
        ret,img = camera.read()
        text = "POS" if in_pos else "NEG"
        cv2.putText(img,text,(20,40),cv2.FONT_HERSHEY_SIMPLEX,
        1,(0,0,255),2,cv2.LINE_AA)
        cv2.imshow("image", img)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            in_pos = False
        elif key == ord('p'):
            in_pos = True
        elif key == ord('t'):
            #pprint(zip(X,Y))
            pprint(clf.fit(X,Y).score(X,Y))
            print('COEFFICIENT MTX:')
            pprint(clf.coef_)
            in_train = False
