#This shows how to import opencv in non-standard path
import sys
sys.path.insert(1,'/usr/local/opencv/opencv-2.4.11/lib/python2.7/site-packages')
sys.maxsize

from cv2 import __version__
print "successful loading opencv module with version"
print __version__

#Basic image
import cv2
import os
import glob

import numpy as np
import matplotlib.pyplot as plt

currentBarcodeNum = "0"

#Purpose:	Process image (Rotation and Scaling) before detection or extraction
#Import:	Angle for rotation if needed, image being read from input
#Export: 	Processed image
def processImage(angle, image):
	
	#Scaling the image into a smaller one
	image = cv2.resize(image, (0,0), fx=0.18, fy=0.18) 
	
	#Get the height and width from the scaled image
	(h,w) = image.shape[:2]
	
	#Do rotation 
	M = cv2.getRotationMatrix2D((w/2,h/2),angle,1)
	image = cv2.warpAffine(image,M,(w,h))

	return image

#Purpose:	Find the highest gradient in the picture using Sobel and Morphology
#Import:	Gray image
#Export: 	Processed image after morphology
def featureDetection(gray):
	
	#Compute the Sobel of the image in both the x and y direction
	gradX = cv2.Sobel(gray, ddepth = cv2.cv.CV_32F, dx = 1, dy = 0, ksize = -1)
	gradY = cv2.Sobel(gray, ddepth = cv2.cv.CV_32F, dx = 0, dy = 1, ksize = -1)
	 
	#Subtract the y-gradient from the x-gradient
	gradient = cv2.subtract(gradX, gradY)
	gradient = cv2.convertScaleAbs(gradient)

	#Blur and threshold the image using binary
	blurred = cv2.blur(gradient, (7, 7))
	(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

	#Apply closing on the image
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
	closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

	#Perform a series of erosions and dilations to get the right gradient 
	newK = np.ones((5,5), np.uint8)
	closed = cv2.erode(closed, newK, iterations = 2)
	closed = cv2.dilate(closed, newK, iterations = 9)
	
	return closed

#Purpose:	Extract barcode area using countours
#Import:	x, y, w, h and the image
#Export: 	Extracted barcode area
def extractBarcodeArea(x, y, w, h, image):
	
	#Crop barcode area
	ori = image[y:y+h, x:x+w]
	
	return ori

#Purpose:	Save barcode number area to directory
#Import:	Barcode number area image
#Export: 	None
def saveFinalImage(finalImg):
	
	global currentBarcodeNum
	
	#Saving to this directory
	path = "/home/student/tan_xhienyi_18249833/output/task1"
	finalImg = cv2.resize(finalImg, (0,0), fx=3, fy=3) 
	
	#Save image to directory according to name
	cv2.imwrite(os.path.join(path, "barcode" + currentBarcodeNum + ".png"), finalImg)

#Purpose:	Refine barcode area again, to find the barcode area more precisely
#Import:	Barcode area image
#Export: 	Refined barcode area image
def processROI(roi):
	
	#Apply gradient morphology
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
	gradient = cv2.morphologyEx(roi, cv2.MORPH_GRADIENT, kernel)
	
	#Apply threshold using binary and otsu
	(_, thresh) = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	
	#Apply closing and erosion 
	closingKernel = np.ones((3,3), np.uint8)
	closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, closingKernel)
	
	erodeKernel = np.ones((5,5), np.uint8)
	closing = cv2.erode(thresh, erodeKernel, iterations = 2)
	
	return closing

#Purpose:	Crop barcode number area using percentage 
#Import:	Height percantage that needed for cropping, barcode area image, x, y, h, w
#Export: 	Barcode number image	
def cropImagePercentage(percentage, roi, x, y, h, w):
	
	#For example, if h = 100 and percentage = 0.20, h_part is 20	
	h_part = int(h * percentage)

	#Use original height minus to h_part to get the height needed for cropping
	cropped_h = h - h_part

	#Crop image resulting in barcode number only
	croppedArea = roi[y+cropped_h:y+h+3, x:x+w]
	
	return croppedArea, cropped_h
	
#Purpose:	Get current barcode file name e.g. barcode1 then return 1
#Import:	Current read barcode number e.g. barcode1
#Export: 	Barcode file number
def getCurrentBarcodeNumber(currentBarcodeName):
	
	#Get the last fifth char from image name, which gets us the image number 
	barcodeNum = ( currentBarcodeName[-5:] )[:1]
	
	return barcodeNum

#Purpose:	This method runs all the operations from start to finish
#Import:	Angle needed for rotation, image that is read from input
#Export: 	None
def runDetector(angle, image):
	
	#First pre-process the image
	image = processImage(angle, image)
	
	#Make it into gray scale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	#Find the greatest gradient from the gray scale image
	closed = featureDetection(gray)
	 
	#Find the contours in the thresholded image, then sort the contours, keeping only the largest one area
	#This area should be the barcode area
	(cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
	
	#Get a rectangle for the detected area
	rect = cv2.minAreaRect(c)
	box = np.int0(cv2.cv.BoxPoints(rect))

	#Points of the reactangle
	x,y,w,h = cv2.boundingRect(c)
	
	#Extracted area called ori
	ori = extractBarcodeArea(x,y,w,h,image)
	
	#Save the the points x and y from the rectangle for later use
	savedX = x
	savedY = y
	
	#Process the extracted area make it call roi (Region of Interest)
	roi = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)

	#refine roi with closing and eroding to get the barcode area more precisely
	closing = processROI(roi)
	
	#Again find the countours for roi
	(cnts, _) = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
	c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

	#Get a rectangle for the detected area in roi
	rect = cv2.minAreaRect(c)
	box2 = np.int0(cv2.cv.BoxPoints(rect))

	#Draw contours on the detected area
	cv2.drawContours(roi, [box2], -1, (0, 255, 0), 2)
	
	#Copy roi for furthur processing
	copyOfROI = roi.copy()
	
	#Points of rectangle from roi
	x,y,w,h = cv2.boundingRect(c)

	#Crop roi to only the numbers using percantage
	croppedArea,  cropped_h = cropImagePercentage(0.18, roi, x, y, h , w)
	
	#Before cropping the barcode numbers, check make sure the image has enough size to be cropped
	if ( x+savedX-10 ) <= 0:
		final = image[y+cropped_h+savedY:y+h+savedY, x+savedX:x+w+savedX]
	else:	
		final = image[y+cropped_h+savedY:y+h+savedY+5, x+savedX-10:x+w+savedX]
	
	#Saves the barcode number to directory
	saveFinalImage(final)

#Purpose:	A method that starts taking in image and then run detection, rotate the image when needed
#Import:	Image that is read from input
#Export: 	None
def startReading(image):	
	
	angle = 0
	
	try:	
		runDetector(angle,image)
	except IndexError:
		try:
			if angle != 360 : 
				angle = angle + 45
				runDetector(angle,image)
		except IndexError:
			try:
				if angle != 360 : 
					angle = angle + 45
					runDetector(angle,image)
			except IndexError:
				try:
					if angle != 360 : 
						angle = angle + 45
						runDetector(angle,image)
				except IndexError:
					try:
						if angle != 360 : 
							angle = angle + 45
							runDetector(angle,image)
					except IndexError:
						try:
							if angle != 360 : 
								angle = angle + 45
								runDetector(angle,image)
						except IndexError:
							try:
								if angle != 360 : 
									angle = angle + 45
									runDetector(angle,image)
							except IndexError:
								try:
									if angle != 360 : 
										angle = angle + 45
										runDetector(angle,image)
								except IndexError:
									try:
										if angle != 360 : 
											angle = angle + 45
											runDetector(angle,image)
									except IndexError:
											print "done"

def main():
	global currentBarcodeNum
	
	#For loop to get all images from the directory
	for img in glob.glob("/home/student/test/task1/*.jpg"):
		
		#Read image using image name	
		cv_img = cv2.imread(img,1)
		
		#Get the image name for later use, e.g. img1.jpg returns 1
		currentBarcodeNum = getCurrentBarcodeNumber(img)
		
		print "Reading input image name: "
		print img
		
		startReading(cv_img)
	        

main()	
