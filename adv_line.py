import cv2
import calib
import mult_thresh
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
    cv2.imwrite(test_dir + 'undist_' + fname, undist)

# TODO: 3. Color/gradient threshold

# TODO: 4. Perspective transform

# TODO: Detect lane lines

# TODO: Determine lane curvature
