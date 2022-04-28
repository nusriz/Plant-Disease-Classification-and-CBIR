import numpy as np
import cv2
import os





read_path 	   = '/home/abishek/FYP/Final-Year-Project/Dataset/lblight'
bin_write_path = os.path.join(read_path,'Binary Image')
ext_write_path = os.path.join(read_path,'Extracted Image')

def image_extractor(img): # K Means CLustering
	LOWER_GREEN = np.array([0,10,10])
	UPPER_GREEN = np.array([50,255,255])
	img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	height,width,channel = img.shape
    ############## Segmentation using K Means Clustering ################
	Z = img_hsv.reshape((-1,3))
	Z = np.float32(Z)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
	K = 2
	ret,label, center = cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	center = np.uint8(center)
	res  = center[label.flatten()]
	seg = res.reshape((img_hsv.shape)) #contains segmented image
	#cv2.imshow('seg',seg)
	print(seg[400,250])
	print(seg[2,2])
    ############## Creating Contour for shape feature extraction #########
	masked = cv2.inRange(seg,LOWER_GREEN,UPPER_GREEN)
	kernel = np.ones((7,7),np.uint8)
	ret, thresh = cv2.threshold(masked,110,255,cv2.THRESH_BINARY)
	opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)
	closing = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel)
	#cv2.imshow('closing',closing)
	_,contours,hierarchy = cv2.findContours(closing,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	ab = 0
	if len(contours) >1 :
		for index in range(len(contours)):
			maxarea = cv2.contourArea(contours[0])
			if cv2.contourArea(contours[index]) > maxarea:
				maxarea = cv2.contourArea(contours[index])
				ab = index    

	black = np.zeros((height,width),np.uint8)
	cv2.drawContours(black,contours,ab,(255,255,255),-1)
	blur = cv2.GaussianBlur(black,(7,7),0)
	#cv2.imshow('blur',blur)
	ret, thresh2 = cv2.threshold(blur,127,255,0) 
	


	#cv2.imshow('thresh2',thresh2)
	#cv2.imshow('binary_image',black)
	extract = cv2.bitwise_and(img,img, mask= thresh2)

	#cv2.imshow('extracted_image',extract)
	return thresh2,extract


def normal_extract(img):
	height,width,channel = img.shape	
	img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	h,s,v = cv2.split(img_hsv)
	ret,thresh = cv2.threshold(s,20,255,cv2.THRESH_BINARY)
	kernel = np.ones((7,7),np.uint8)
	opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)
	closing = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel)
	#cv2.imshow('closing',closing)
	_,contours,hierarchy = cv2.findContours(closing,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	ab = 0
	if len(contours) >1 :
		for index in range(len(contours)):
			maxarea = cv2.contourArea(contours[0])
			if cv2.contourArea(contours[index]) > maxarea:
				maxarea = cv2.contourArea(contours[index])
				ab = index    

	black = np.zeros((height,width),np.uint8)
	cv2.drawContours(black,contours,ab,(255,255,255),-1)
	blur = cv2.GaussianBlur(black,(7,7),0)
	#cv2.imshow('blur',blur)
	ret, thresh2 = cv2.threshold(blur,127,255,0) 


	#cv2.imshow('thresh2',thresh2)
	#cv2.imshow('binary_image',black)
	extract = cv2.bitwise_and(img,img, mask= thresh2)
	#cv2.imshow('extracted_image',extract)
	return thresh2,extract





i=0
lst = os.listdir(read_path)
lst.sort()


for file in lst:
	if file.endswith('.png'):
		i = i+1
		img = cv2.imread(os.path.join(read_path,file))
		#cv2.imshow('orig',img)
		# img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
		binary , extracted = normal_extract(img)
		#binary , extracted = image_extractor(img)
		#cv2.imshow('bin',binary)
		#cv2.imshow('ext',extracted)
	
		#binary , extract = image_extractor(img)
		#binary , extract = image_extractor(extract)
		#binary , extract = image_extractor(extract)
		#binary , extract = image_extractor(extract) # Operated multiple times to segment better
		bin_name = 'binary' + str(i) + '.png'
		ext_name = 'extract' + str(i) + '.png'
		cv2.imwrite(os.path.join(bin_write_path,bin_name),binary)
		cv2.imwrite(os.path.join(ext_write_path,ext_name),extracted)
		print(i)
		print(file)
		# cv2.imshow('orig',img)
		# cv2.imshow('extracted_image',extracted)
		# cv2.imshow('binary',binary)
		# cv2.imshow('extracted_image2',extracted2)
		# cv2.imshow('binary2',binary2)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()