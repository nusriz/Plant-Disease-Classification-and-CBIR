import cv2
import numpy as np 


MIN_DESCRIPTOR = 30

def findDescriptor(img):
	contour = []
	_,contour, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	contour_array = contour[0][:,0,:]
	contour_complex = np.empty(contour_array.shape[:-1],dtype = complex)
	contour_complex.real = contour_array[:, 0]
	contour_complex.imag = contour_array[:,-1]
	fourier_result = np.fft.fft(contour_complex)
	return fourier_result
def truncate_descriptor(descriptors, degree):
    """this function truncates an unshifted fourier descriptor array
    and returns one also unshifted"""
    descriptors = np.fft.fftshift(descriptors)
    center_index = len(descriptors) / 2
    #center_index = int(center_index)

    descriptors = descriptors[int(center_index - degree / 2):int(center_index + degree / 2)]
    descriptors = np.fft.ifftshift(descriptors)
    return descriptors


def reconstruct(descriptors, degree):
    """ reconstruct(descriptors, degree) attempts to reconstruct the image
    using the first [degree] descriptors of descriptors"""
    # truncate the long list of descriptors to certain length
    descriptor_in_use = truncate_descriptor(descriptors, degree)
    contour_reconstruct = np.fft.ifft(descriptor_in_use)
    contour_reconstruct = np.array(
        [contour_reconstruct.real, contour_reconstruct.imag])
    contour_reconstruct = np.transpose(contour_reconstruct)
    contour_reconstruct = np.expand_dims(contour_reconstruct, axis=1)
    # make positive
    if contour_reconstruct.min() < 0:
        contour_reconstruct -= contour_reconstruct.min()
    # normalization
    contour_reconstruct *= 800 / contour_reconstruct.max()
    # type cast to int32
    contour_reconstruct = contour_reconstruct.astype(np.int32, copy=False)
    black = np.zeros((800, 800), np.uint8)
    # draw and visualize
    cv2.drawContours(black, contour_reconstruct, -1, 255, thickness=10)
    cv2.imshow("black", black)
    cv2.waitKey(1000)
    cv2.imwrite("reconstruct_result.jpg", black)
    cv2.destroyAllWindows()
    return descriptor_in_use


img = cv2.imread('cnt.png',0)
cv2.imshow('img',img)
print(findDescriptor(img))
fourier_result = findDescriptor(img)
contour_reconstruct = reconstruct(fourier_result, MIN_DESCRIPTOR)