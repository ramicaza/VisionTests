#!/usr/bin/env python3

# Detects potholes using umigv™ custom opencv fitEllipseWithError func

import cv2
import numpy as np
from time import time
import transform
from pprint import pprint

AREA_THRESH = 500
# TODO: determine if this depends on ellipse size
ELLIPSENESS_THRESH = 12
# These constants were learned using linear_class.py
COLOR_FILTER = np.array([0.00626744, -0.00635294,  0.0077638 , -1.122643662730205])

def filter_by_area(contours,thresh):
	ret = []
	for contour in contours:
		if cv2.contourArea(contour) > thresh:
			ret.append(contour)
	return ret

def filter_by_ellipseness(contours,thresh):
	ret = []
	for contour in contours:
		real_area = cv2.contourArea(contour)
		if real_area < 5: continue
		ellipse, dist = cv2.fitEllipseWithError(contour)
		if dist < thresh:
			ret.append(contour)
	return ret

"""
The parameters in this fxn should be tweaked in order
to optimally get all shades of white... Currently the approach is
to convert to hsv color space and filter by saturation and brightness
"""
def get_white_contours(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	biased = np.ones(shape=(hsv.shape[0],hsv.shape[1],4),dtype=np.uint8)
	biased[:,:,:3] = hsv
	dot = biased.dot(COLOR_FILTER)
	dot = np.clip(dot,0,1)
	dot = dot.astype(int)
	mask = cv2.inRange(dot,1,1)
	cv2.imshow("mask", mask)
	#print(hsv[240][320])
	#cv2.circle(img,(320,240),5,(0,0,255),-1)

	_, contours, _ = cv2.findContours(mask,
	cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	return contours

if __name__ == "__main__":
	camera = cv2.VideoCapture(0)
	start = time()

	#This is an example of how to make and use the transform api/functor
	t = transform.Transform(69)
	print("Transformed point: ",t.transform(100,50))

	while True:
		ret,img = camera.read()
		contours = get_white_contours(img)
		contours = filter_by_area(contours,AREA_THRESH)
		contours = filter_by_ellipseness(contours,ELLIPSENESS_THRESH)
		fps = 1.0/(time()-start)
		print("{} potholes found - {} fps".format(len(contours),fps))

		if len(contours) > 0:
			for contour in contours:
				M = cv2.moments(contour)
				center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
				print(center)
				cv2.circle(img, center, 5, (0,0,255), -1)

		cv2.imshow("blobs", img)
		start = time()
		if cv2.waitKey(5) & 0xFF == ord('q'):
			break
