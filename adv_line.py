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
#fname = 'test1.jpg'
#print(fname)
    img = cv2.imread(test_dir + fname)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    #cv2.imwrite(test_dir + 'undist_' + fname, undist)

    # TODO: 3. Color/gradient threshold
    # TODO: Test other gradient thresholds to see what is going on

    # Threshold gradients with angles between 30 and 60, and between 120 and 150
    dir_grad_r = mt.dir_threshold(undist,sobel_kernel=3, thresh=(np.pi/6,np.pi/3))
    dir_grad_l = mt.dir_threshold(undist,sobel_kernel=3, thresh=(2*np.pi/3,5*np.pi/6))

    # Threshold gradients by magnitude
    abs_sob_binary = mt.mag_thresh(undist, sobel_kernel=3, mag_thresh=(5,30))

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]
    color_binary = np.zeros_like(dir_grad_r)

    #color_binary[(L >= 60) & (L <= 120)] = 1
    color_binary[(S >= 160) & (S <= 255)] = 1
    #ret = cv2.imwrite(test_dir + 'h_' + fname, H)
    #ret = cv2.imwrite(test_dir + 'l_' + fname, L)
    #ret = cv2.imwrite(test_dir + 's_' + fname, S)

    combined = np.zeros_like(dir_grad_r)
    combined[((dir_grad_r == 1) | (dir_grad_l == 1)) & (abs_sob_binary == 1) & (color_binary == 1)] = 1

    undist[combined == 1] = 255
    undist[combined == 0] = 0

    ret = cv2.imwrite(test_dir + 'dir_grad_color_' + fname, undist)
    print(ret)
# TODO: 4. Perspective transform



# TODO: Detect lane lines

# TODO: Determine lane curvature
