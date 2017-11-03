#!/usr/bin/env python

# Detects potholes using umigv™ custom opencv fitEllipseWithError func

import cv2
import numpy as np

def filter_by_area(contours,thresh):
	ret = []
	for contour in contours:
		if cv2.contourArea(contour) > thresh:
			ret.append(contour)
	return ret

def filter_by_solidity(contours,thresh):
	ret = []
	for contour in contours:
		area = cv2.contourArea(contour)
		hull = cv2.convexHull(contour)
		hull_area = cv2.contourArea(hull)
		if area == 0 or hull_area == 0: continue 
		solidity = float(area)/hull_area
		if solidity > thresh:	
			ret.append(contour)
	return ret

def filter_by_ellipseness(contours,thresh):
	ret = []
	for contour in contours:
		real_area = cv2.contourArea(contour)
		if real_area < 5: continue
		ellipse, dist = cv2.fitEllipseWithError(contour)
		#ellipse_area = 3.14159*ellipse[1][0]*ellipse[1][1]/4
		#dist = abs(ellipse_area/real_area - 1)
		if dist < thresh:
			ret.append(contour)
	return ret

camera = cv2.VideoCapture(0)
while True:
	ret,img = camera.read()
	#grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	
	lower_thresh = np.array([0,0,110], dtype = "uint8")
	upper_thresh = np.array([179,30,255], dtype="uint8")

	mask = cv2.inRange(hsv, lower_thresh, upper_thresh)
	#print(hsv[240][320])
	#cv2.circle(img,(320,240),5,(0,0,255),-1)
	
	_, contours, _ = cv2.findContours(mask,
		cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	contous = filter_by_area(contours,20)
	contours = filter_by_solidity(contours,0.9)
	contours = filter_by_ellipseness(contours,20)

	if len(contours) > 0:
		contour = max(contours, key=lambda el: cv2.contourArea(el))
		M = cv2.moments(contour)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		cv2.circle(img, center, 5, (0,0,255), -1)
	
	cv2.imshow("blobs", img)
	if cv2.waitKey(5) & 0xFF == ord('q'):
		break
