#!/usr/bin/env python

# Detects potholes using umigv™ custom opencv fitEllipseWithError func

import cv2
import numpy as np
from time import time
import transform

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
	lower_thresh = np.array([0,0,110], dtype = "uint8")
	upper_thresh = np.array([179,30,255], dtype="uint8")
	mask = cv2.inRange(hsv, lower_thresh, upper_thresh)
	cv2.imshow("mask", mask)
	#print(hsv[240][320])
	#cv2.circle(img,(320,240),5,(0,0,255),-1)

	_, contours, _ = cv2.findContours(mask,
	cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	return contours

if __name__ == "__main__":
	AREA_THRESH = 500
	# TODO: determine if this depends on ellipse size
	ELLIPSENESS_THRESH = 10
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
