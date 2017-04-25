import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Read in an image
image = cv2.imread('signs_vehicles_xygrad.png')
print()
# Function to calc gradient direction between given thresholds

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel) # Apply Sobel
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)

    abs_sobelx = np.absolute(sobelx) # Absolute values
    abs_sobely = np.absolute(sobely)

    grad_dir = np.arctan2(abs_sobely, abs_sobelx) # Calc grad direction

    sxbinary = np.zeros_like(grad_dir) # Mask for thresholded directions
    sxbinary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1

    return sxbinary

# Function to calc gradient magnitude between given thresholds

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Grayscale

    if(sobel_kernel % 2 == 1): # Test kernel values to be odd (optional?)
        k = sobel_kernel
    else:
        k = 3

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = k) # Apply Sobel
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = k)

    abs_sobelxy = np.sqrt(np.square(sobelx) + np.square(sobely)) # Calc magnitude

    scaled_sobelxy = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy)) # Scale to 255 and convert to uint8

    sxbinary = np.zeros_like(scaled_sobelxy) # Mask for thresholded magnitudes
    sxbinary[(scaled_sobelxy >= mag_thresh[0]) & (scaled_sobelxy <= mag_thresh[1])] = 1

    return sxbinary

# Calc x or y gradient and applies thresholds
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Grayscale

    if (orient == 'x'): # Absolute Sobel in requested orientation, scaled, in uint8
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    elif(orient == 'y'):
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
        abs_sobely = np.absolute(sobely)
        scaled_sobel = np.uint8(255*abs_sobely/np.max(abs_sobely))

    sxbinary = np.zeros_like(scaled_sobel) # Threshold mask
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return sxbinary

ksize = 3

'''
gradx = abs_sobel_thresh(image, orient='x',sobel_kernel=ksize, thresh=(25,100))
grady = abs_sobel_thresh(image, orient='y',sobel_kernel=ksize, thresh=(25,100))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(50,100))
dir_binary_r = dir_threshold(image, sobel_kernel=ksize, thresh=(np.pi/6,np.pi/3))
dir_binary_l = dir_threshold(image, sobel_kernel=ksize, thresh=(2*np.pi/3,5*np.pi/6))
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0,np.pi/2))


combined = np.zeros_like(mag_binary)
combined_sep_l_r = np.zeros_like(mag_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
combined_sep_l_r[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & ((dir_binary_r == 1) | (dir_binary_l == 1)))] = 1

mpimg.imsave('./combined.png',combined,cmap='gray')
mpimg.imsave('./combined_sep_l_r.png',combined_sep_l_r,cmap='gray')
'''
