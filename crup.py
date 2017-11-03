#!/usr/bin/env python

# Cutting edge CV research performed in this file

import cv2
import numpy as np

def transformBound(rect,xs,ys):
    x,y,w,h = rect
    xt = int(round(x + (w-w*xs)/2))
    wt = int(round(w*xs))
    yt = int(round(y + (h-h*ys)/2))
    ht = int(round(h*ys))
    return (xt,yt,wt,ht)

faceCascade = cv2.CascadeClassifier("/home/nico/OpenCV/data/haarcascades\
/haarcascade_frontalface_default.xml")

camera = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    if len(faces) > 0:
        xt,yt,wt,ht = transformBound(faces[0],0.6,0.7)
        #cv2.rectangle(frame, (xt, yt), (xt+wt, yt+ht), (0, 255, 0), 2)
        cropped = frame[yt:yt+ht, xt:xt+wt]
        sized = cv2.resize(cropped,(2500,1000))
        sized = cv2.flip(sized,1)
        cv2.imshow('Video', sized)
    # Display the resulting

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
