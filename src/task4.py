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


newSamples = np.empty((0,100))
index = 0

#Purpose:	Save images in order
#Import:	Current read barcode number e.g. barcode1, image to be saved, index in order
#Export: 	None
def saveFinalImageInOrder(barcodeNum, img, finalIndex):
	
	#Path for saving
	path = "/home/student/tan_xhienyi_18249833/output/task2/barcode" + str(barcodeNum) + "/"
	
	#Add 0 infront of digits which are less than 10 for saving purposes
	if finalIndex < 10:
		cv2.imwrite(os.path.join(path, "d" + str(0) + str(finalIndex) + ".png"), img)
	else:
		cv2.imwrite(os.path.join(path, "d" + str(finalIndex) + ".png"), img)	

#Purpose:	Save temporary crop region to directory 
#Import:	Current read barcode number e.g. barcode1, temporary image to be saved
#Export: 	None	
def saveTempImage(barcodeNum, img):
	
	global index
	
	#Path for saving					
	path = "/home/student/tan_xhienyi_18249833/output/task2/barcode" + str(barcodeNum) + "/"
	
	#Save image and increase the index
	index += 1
	cv2.imwrite(os.path.join(path, "a" + str(index) + ".png"), img)

#Purpose:	Sort images according to their x value, then delete the temporary images
#Import:	Current read barcode number e.g. barcode1, dictionary that has image name and their x value
#Export: 	None
def sortImage(barcodeNum, x_dict):
	
	#Get the list of x values and sort them in order
	keylist = x_dict.keys()
	keylist.sort()
	
	indexNum = 0
	
	#For even x value in the list
	for key in keylist:

		for img in glob.glob("/home/student/tan_xhienyi_18249833/output/task2/barcode"+ str(barcodeNum) + "/" + x_dict[key]):
			
			indexNum += 1
			
			#Read the temp image and change the name and save it as final image
			cv_img = cv2.imread(img,1)
			saveFinalImageInOrder(barcodeNum ,cv_img, indexNum)
			
			#Remove the temp images once is saved as final image
			os.remove(img)

#Purpose:	Crop digit
#Import:	image, x, y, w, h
#Export: 	Cropped area
def cropDigit(im, x, y, w, h):
	
	increaseCropSize = 4
	if  ( (y- increaseCropSize) < 0 ) or ( (x-increaseCropSize) < 0 ):
		return im
	else:
		digit = im[y-increaseCropSize:y+h+increaseCropSize,x-increaseCropSize:x+w+increaseCropSize]
		return digit
	
	
#Purpose:	Detect number using contour
#Import:	Current image read from directory, responses for training data, Current read barcode number e.g. barcode1
#Export: 	None
def detectNumber(im, newResponses, barcodeNum):
	
	global newSamples
	global index
	
	x_array = []
	x_dict = {}
	overlap = False
	
	gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

	#Find contours using external so it draws a box on each digit 
	contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	
	samples =  np.empty((0,100))
	responses = []
	keys = [i for i in range(48,58)]
	
	#For each countour in this image and if the area is greater than 50
	for cnt in contours:
		
		if cv2.contourArea(cnt)>50:
			
			[x,y,w,h] = cv2.boundingRect(cnt)
			
	        ## Set some constraint to avoid detecting none digit area
			if  h>30 and w>10 and abs(w-h) > 10:
				
				#Check for overlap rectangles, e.g, Zero may be detected more than once
				for stored_x in x_array:
					
					#If a detected area is less or equal to 10 than an already detected area, it means the both are too close
					#Which results in overlapping rectangles
					#If overlaps then break out of this loop and try next contours
					if abs(stored_x - x) <= 10:
						
						overlap = True
						break
						
				#If no overlap is occured		
				if overlap == False:
					
					#Crop digit into a roi
					roi = cropDigit(thresh, x, y, w, h)
					ori = cropDigit(im, x, y, w, h)
					
	            
					#Denoise image if blurred
					ori = cv2.fastNlMeansDenoisingColored(ori,None,10,10,7,21)
					
					#Save temp image 
					saveTempImage(barcodeNum, ori)
					
					#Save temp image's x value as a key to a dictionary along with its name
					x_dict[x] = "a" + str(index) + ".png"
					
					#Store x value into an array that is used for checking overlapping
					x_array.append(x)
					
				else:
					overlap = False
	
	#Sort image in order using their x values
	
	sortImage(barcodeNum, x_dict)


#Purpose:	Get current barcode file name e.g. barcode1 then return 1
#Import:	Current read barcode number e.g. barcode1
#Export: 	Barcode file number
def getCurrentBarcodeNumber(currentBarcodeName):
	
	barcodeNum = ( currentBarcodeName[-5:] )[:1]
	return barcodeNum
	
indexTask3 = 0
recognizedNumber = "0"
outputArray = []


#Purpose:	Read images and sort them in order by their name
#Import:	Current barcode number image name
#Export: 	Filenames
def readImagesInOrder(barcodeNum):
	
	#Filename for this path, "/home/student/tan_xhienyi_18249833/output/task1/*.png
	filename = [img for img in glob.glob("/home/student/tan_xhienyi_18249833/output/task2/barcode" + str(barcodeNum) + "/*.png")]
	
	#Sort file name is order by their name
	filename.sort()
	
	return filename
	
#Purpose:	Pre-process image
#Import:	Current read barcode number e.g. barcode1
#Export: 	Barcode file number
def imageProccessing(im):
	
	#Convert to gray scale image
	gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	
	#Apply threshhold
	thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
	
	return thresh

#Purpose:	Loads my predictive model from files
#Import:	None
#Export: 	Samples and Responses from my predictive model
def loadPredictiveModelSamples():
	
	#Loads my model
	samples = np.loadtxt('/home/student/tan_xhienyi_18249833/src/generalsamples.data',np.float32)
	responses = np.loadtxt('/home/student/tan_xhienyi_18249833/src/generalresponses.data',np.float32)
	responses = responses.reshape((responses.size,1))
	
	return samples, responses

#Purpose:	Initialise KNearest Classifier and train my data
#Import:	Samples and Responses from my predictive model
#Export: 	KNearest model ready for classification
def initKNN(samples, responses):
	
	#Initialise KNearest
	model = cv2.KNearest()
	
	#Train KNearest with my data
	model.train(samples,responses)
	
	return model

#Purpose:	Saves each digit into a text file
#Import:	Current barcode number image name
#Export: 	None
def saveTxtFile(currentBarcodeNum):
	
	global indexTask3
	global recognizedNumber
	
	#Path for saving
	path = "/home/student/tan_xhienyi_18249833/output/task3/barcode" + str(currentBarcodeNum) + "/"
	
	#Name of file
	if indexTask3 < 10:
		name_of_file = "d" + str(0) + str(indexTask3)
	else:
		name_of_file = "d" + str(indexTask3)	

	#Complete file name by concatenating the file type
	completeName = os.path.join(path, name_of_file + ".txt")
	
	#Open that file with write permission
	txtFile = open(completeName,"w")
	
	#Write recognised digit to file
	txtFile.write(recognizedNumber)
	
	#Close file
	txtFile.close()
	
	return outputArray

#Purpose:	Save all numbers into a single txt
#Import:	Numbers array and current barcode name
#Export: 	None	
def combineNumberIntoFile(outputArray, currentBarcodeNum):
	
	
	numbers = ""
	for i in outputArray:
		numbers = numbers + i
	
	#Path for saving
	path = "/home/student/tan_xhienyi_18249833/output/task4/"
	

	name_of_file = "img" + str(currentBarcodeNum)	

	#Complete file name by concatenating the file type
	completeName = os.path.join(path, name_of_file + ".txt")
	
	#Open that file with write permission
	txtFile = open(completeName,"w")
	
	#Write recognised digit to file
	txtFile.write(numbers)
	
	#Close file
	txtFile.close()
	
		
#Purpose:	Recognise digits
#Import:	Filename from a directory
#Export: 	None	
def recognizeDigits(filename, currentBarcodeNum):
	
	global indexTask3
	global recognizedNumber
	
	#Load my predictive model
	samples, responses = loadPredictiveModelSamples()
	
	#Train my data using KNearest
	model = initKNN(samples, responses)
	
	#For each digit in file
	for img in filename:
		
		indexTask3 = indexTask3 + 1 
		
		#Read digit image
		im = cv2.imread(img,1)
		
		#Pre-process image
		thresh = imageProccessing(im)
		
		#Find contours for each digtis
		contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		
		#For each countours
		for cnt in contours:
			
			#If the area is greater than 50
		    if cv2.contourArea(cnt)>50:
				
		        [x,y,w,h] = cv2.boundingRect(cnt)
		        
		        #Set some constraints to avoid detecting non digits
		        if  h>30 and w>10 and abs(w-h) > 10:
					
		            #Resize the digit into 10x10 and make it into float points for KNN
		            roi = thresh[y:y+h,x:x+w]
		            roismall = cv2.resize(roi,(10,10))
		            roismall = roismall.reshape((1,100))
		            roismall = np.float32(roismall)
		            
		            #Find nearest similar digit from sample using k = 1
		            retval, results, neigh_resp, dists = model.find_nearest(roismall, k = 1)
		            
		            #String recognised
		            string = str(int((results[0][0])))
		            
		            #Assign recognizedNumber to String recognised
		            recognizedNumber = string
		
		#Save recognizedNumber to text file
		saveTxtFile(currentBarcodeNum)
		
		#Append each recognizedNumber to an array for showing the result on terminal
		outputArray.append(recognizedNumber)
		
		#Reset recognizedNumber
		recognizedNumber = "0"
	
	combineNumberIntoFile(outputArray, currentBarcodeNum)	
	

#task1	
def task1():
	global currentBarcodeNum
	print "Running Task 1 ..."
	#For loop to get all images from the directory
	for img in glob.glob("/home/student/test/task1/*.jpg"):
		
		#Read image using image name	
		cv_img = cv2.imread(img,1)
		
		#Get the image name for later use, e.g. img1.jpg returns 1
		currentBarcodeNum = getCurrentBarcodeNumber(img)
		
		print "Reading input image name: "
		print img
		
		startReading(cv_img)

#task2
newResponses = []

def task2():
	print "Running Task 2 ..."
	for imgName in glob.glob("/home/student/tan_xhienyi_18249833/output/task1/*.png"):
	
		index = 0
		currentNumber = getCurrentBarcodeNumber(imgName)
		print "Reading each image: barcode", str(currentNumber)
		cv_img = cv2.imread(imgName,1)
		detectNumber(cv_img, newResponses, currentNumber)
		
		
def task3():

	print "Running Task 3 ..."
	global indexTask3
	global outputArray

	#Read barcode from 1 to 5
	for i in range(1,6):
		
		print "Reading Barcode", i
		
		#Get digits from barcode i
		filename = readImagesInOrder(i)
		
		#Start recognising
		recognizeDigits(filename, i)
		
		indexTask3 = 0
		outputArray = []
				        

def main():
	task1()
	task2()
	task3()
	
main()				        
