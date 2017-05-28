import cv2
import calib
import warp_img
import mult_thresh as mt
import window_search as ws
import region_of_interest as roi
import os
import numpy as np
import tune_threshold as tt

test_dir = './video_imgs/'
save_dir = './binary/'
test_fnames = os.listdir(test_dir)
mtx, dist = calib.calibrate()

for fname in test_fnames:
    img = cv2.imread(test_dir + fname)
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    HSV = cv2.cvtColor(undist, cv2.COLOR_RGB2HSV)
    YUV = cv2.cvtColor(undist, cv2.COLOR_RGB2YUV)
    HLS = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS)

    # Yellow threshold
    abs_UV = np.absolute(YUV[:,:,1] - YUV[:,:,2])
    thresh_UV = cv2.inRange(abs_UV, 30,140).astype(np.uint8)
    yellow = cv2.inRange(HSV, (100, 100, 200), (255, 255, 255))

    yellow2 = cv2.inRange(undist, (225, 180, 0), (255, 255, 170))
    yellow3 = cv2.inRange(HLS, (14, 160, 110), (45, 200, 255))

    # White threshold
    sensitivity_1 = 40
    white = cv2.inRange(HSV, (0, 0, 255 - sensitivity_1), (255, 20, 255))

    white_2 = cv2.inRange(undist, (200, 200, 200), (255, 255, 255))

    color_binary = yellow | white | white_2 | thresh_UV | yellow2 | yellow3

    offset_x_l = 125
    offset_x_r = 0
    bottomLeft = [offset_x_l, color_binary.shape[0]]
    topLeft = [570, 450]
    topRight = [720, 450]
    bottomRight = [color_binary.shape[1] - offset_x_r, color_binary.shape[0]]
    vertices = np.array([[bottomLeft,topLeft,topRight,bottomRight]])
    masked = roi.roi(color_binary,vertices)

    cv2.imwrite(save_dir + 'binary_' + fname + '.jpg', masked)
