#This shows how to import opencv in non-standard path
import sys
sys.path.insert(1,'/usr/local/opencv/opencv-2.4.11/lib/python2.7/site-packages')
sys.maxsize

from cv2 import __version__
print "successful loading opencv module with version"
print __version__

#Basic image
import cv2
import glob
import os

import numpy as np
import matplotlib.pyplot as plt
  
newSamples = np.empty((0,100))
index = 0

#Purpose:	Save images in order
#Import:	Current read barcode number e.g. barcode1, image to be saved, index in order
#Export: 	None
def saveFinalImage(barcodeNum, img, finalIndex):
	
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
			saveFinalImage(barcodeNum ,cv_img, indexNum)
			
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

newResponses = []

def main():
	
	for imgName in glob.glob("/home/student/test/task2/*.png"):
	
		index = 0
		currentNumber = getCurrentBarcodeNumber(imgName)
		print "Reading each image: barcode", str(currentNumber)
		cv_img = cv2.imread(imgName,1)
		detectNumber(cv_img, newResponses, currentNumber)
		
main()



	
