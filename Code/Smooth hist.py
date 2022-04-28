import cv2
from matplotlib import pyplot as plt
import numpy as np 
import os
from numpy import savetxt,loadtxt
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage.filters import gaussian_filter1d

read_path = '/home/abishek/FYP/Final-Year-Project/Tutorial'#Dataset/bacterialspot/Extracted Image'
#write_path = '/home/abishek/FYP/Final-Year-Project/Dataset/bacterialspot/features'
X_bgr = []
X_hsv = []




for file in os.listdir(read_path):
	if file.endswith('lblight.png'):
		bgr_img = cv2.imread(os.path.join(read_path,file))
		#bgr_img = cv2.cvtColor(bgr_img,cv2.COLOR_BGR2HSV)	
		#bgr_hsv = cv2.cvtColor(bgr_img,cv2.COLOR_BGR2HSV)
		hist_b = cv2.calcHist([bgr_img],[0],None,[255],[1,256])
		hist_g = cv2.calcHist([bgr_img],[1],None,[255],[1,256])
		hist_r = cv2.calcHist([bgr_img],[2],None,[255],[1,256])
	

	
		bins = np.asarray([x for x in range(1,256)])
		xnew = np.linspace(bins.min(), bins.max(), 50) 
		print(bins)
		spl1 = make_interp_spline(bins,hist_b,k=3)
		smooth1 = spl1(xnew)
		spl2 = make_interp_spline(bins,hist_g,k=3)
		smooth2 = spl2(xnew)
		spl3 = make_interp_spline(bins,hist_r,k=3)
		smooth3 = spl3(xnew)
		smooth1 = smooth1.T[0]
		smooth2 = smooth2.T[0]
		smooth3 = smooth3.T[0]
		
		smooth1 = gaussian_filter1d(smooth1, sigma=1.5)
		smooth2 = gaussian_filter1d(smooth2, sigma=1.5)
		smooth3 = gaussian_filter1d(smooth3, sigma=1.5)
		
		plt.fill_between(xnew,0,smooth1,color = 'b', alpha = 0.3)
		plt.fill_between(xnew,0,smooth2,color = 'g', alpha = 0.3)
		plt.fill_between(xnew,0,smooth3,color = 'r', alpha = 0.3)
		# plt.plot(xnew,smooth2,color = 'g' ,alpha = 0.3)
		# plt.plot(xnew,smooth3,color = 'r',alpha = 0.3)

		plt.xlim([1,256])
		#plt.ylim([0,4000])
		#plt.fill(hist_b, alpha = 0.5)
		#plt.ylim([0,4000])
		plt.xlabel('Bins')
		plt.ylabel('Frequency')
		plt.title('HSV histogram of a late blight affected leaf ')
		# ax2.plot(hist_h)

		plt.show()
		# hist_b = hist_b.astype('int')
		# hist_g = hist_g.astype('int')
		# hist_r = hist_r.astype('int')
		# #hist_h = hist_h.astype('int')
		# hist_b = hist_b.T[0]
		# hist_g = hist_g.T[0]
		# hist_r = hist_r.T[0]
		# #hist_h = hist_h.T[0]
		# hist = np.concatenate((hist_b,hist_g))
		# hist = np.concatenate((hist,hist_r))
		# X_bgr.append(hist)
		#X_hsv.append(hist_h)




#features1 = np.asarray(X_bgr)
#features2 = np.asarray(X_hsv)
#print(features1.shape)
#print(features2.shape)

#savetxt('')

#savetxt(os.path.join(write_path,'bacterialspot_bgr_hist.csv'), features1, delimiter=',',fmt='%8.0u')
#savetxt(os.path.join(write_path,'bacterialspot_h_hist.csv'), features2, delimiter=',',fmt='%8.0u')
#data = loadtxt('bacterialspot_bgr_hist.csv', delimiter=',')
#print(data.shape)



