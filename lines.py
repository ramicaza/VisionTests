#!/usr/bin/env python

# This was the original test in this repo

import numpy as np
import cv2

cap = cv2.VideoCapture("Sample.mp4")

while(True):
    # Capture frame-by-frame
	ret, frame = cap.read()

	frame = frame[100:1200, 100:1100]
	
	# Blur the frame
	blur = cv2.GaussianBlur(frame,(11,11),0)

	# Convert to HSV colorspace for thresholding
	hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
   
	lower_white = np.array([0,0,140], np.uint8)
	upper_white = np.array([180,70,255], np.uint8)

	mask = cv2.inRange(hsv, lower_white, upper_white)
	
	kernel = np.ones((5,5), np.uint8)

	img_erosion = cv2.erode(mask, kernel, iterations=1)
	img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
	

	""" 
	size = np.size(img_dilation)
	skel = np.zeros(img_dilation.shape,np.uint8)

	element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	done = False

	while(not done):
		eroded = cv2.erode(img_dilation, element)
		temp = cv2.dilate(eroded,element)
		temp = cv2.subtract(img_dilation, temp)
		skel = cv2.bitwise_or(skel,temp)
		img_dilation = eroded.copy()

		zeros = size - cv2.countNonZero(mask)
		if zeros==size:
			done = True
	"""

	canny = cv2.Canny(img_dilation, 50, 150,apertureSize = 3)	

	(mu, sigma) = cv2.meanStdDev(canny)
	edges = cv2.Canny(canny, mu - sigma, mu + sigma)
	lines = cv2.HoughLines(edges, 1, np.pi / 180, 70)
	
	if lines is not None:
		print len(lines[0])

		for rho,theta in lines[0]:
			a = np.cos(theta)
 			b = np.sin(theta)
			x0 = a*rho
			y0 = b*rho
			x1 = int(x0 + 1000*(-b))
			y1 = int(y0 + 1000*(a))
			x2 = int(x0 - 1000*(-b))
			y2 = int(y0 - 1000*(a))

			cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
	
	# Display the resulting frame	
	cv2.imshow('frame',canny)
		
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
