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

index = 0
recognizedNumber = "0"
outputArray = []


#Purpose:	Read images and sort them in order by their name
#Import:	Current barcode number image name
#Export: 	Filenames
def readImagesInOrder(barcodeNum):
	
	#Filename for this path, "/home/student/tan_xhienyi_18249833/output/task1/*.png
	filename = [img for img in glob.glob("/home/student/test/task3/barcode" + str(barcodeNum) + "/*.png")]
	
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
	
	global index
	global recognizedNumber
	
	#Path for saving
	path = "/home/student/tan_xhienyi_18249833/output/task3/barcode" + str(currentBarcodeNum) + "/"
	
	#Name of file
	if index < 10:
		name_of_file = "d" + str(0) + str(index)
	else:
		name_of_file = "d" + str(index)	

	#Complete file name by concatenating the file type
	completeName = os.path.join(path, name_of_file + ".txt")
	
	#Open that file with write permission
	txtFile = open(completeName,"w")
	
	#Write recognised digit to file
	txtFile.write(recognizedNumber)
	
	#Close file
	txtFile.close()
	
	return outputArray
	
#Purpose:	Recognise digits
#Import:	Filename from a directory
#Export: 	None	
def recognizeDigits(filename, currentBarcodeNum):
	
	global index
	global recognizedNumber
	
	#Load my predictive model
	samples, responses = loadPredictiveModelSamples()
	
	#Train my data using KNearest
	model = initKNN(samples, responses)
	
	#For each digit in file
	for img in filename:
		
		index = index + 1 
		
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
		
	print outputArray

def main():

	global index
	global outputArray

	#Read barcode from 1 to 5
	for i in range(1,6):
		
		print "Reading Barcode", i
		
		#Get digits from barcode i
		filename = readImagesInOrder(i)
		
		#Start recognising
		recognizeDigits(filename, i)
		
		index = 0
		outputArray = []
	
main()
