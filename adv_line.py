import cv2
import calib
import mult_thresh as mt
import os
import numpy as np

# TODO: 1. Camera calibration
mtx, dist = calib.calibrate()

# TODO: 2. Distortion correction
test_dir = './test_images/'
test_fnames = os.listdir(path=test_dir)
for fname in test_fnames:
    img = cv2.imread(test_dir + fname)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    #cv2.imwrite(test_dir + 'undist_' + fname, undist)

# TODO: 3. Color/gradient threshold
# TODO: Test other gradient thresholds to see what is going on
#dir_grad_r = mt.dir_threshold(undist,sobel_kernel=3, thresh=(np.pi/6,np.pi/3))
#dir_grad_l = mt.dir_threshold(undist,sobel_kernel=3, thresh=(2*np.pi/3,5*np.pi/6))
#dir_grad_r = mt.dir_threshold(undist,sobel_kernel=3, thresh=(0,np.pi/2))
abs_sob = mt.mag_thresh(undist, sobel_kernel=3, mag_thresh=(10,100))
print(abs_sob)
#combined = np.zeros_like(dir_grad_r)
#combined[(dir_grad_r == 1) | (dir_grad_l == 1)] = 1
combined = np.copy(abs_sob)

ret = cv2.imwrite(test_dir + 'dir_grad_' + fname, abs_sob)
print(ret)
# TODO: 4. Perspective transform

# TODO: Detect lane lines

# TODO: Determine lane curvature
