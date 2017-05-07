import cv2
import calib
import warp_img
import mult_thresh as mt
import window_search as ws
import os
import numpy as np
import matplotlib.pyplot as plt
#from pylab import *

# TODO: 1. Camera calibration
mtx, dist = calib.calibrate()

# TODO: 2. Distortion correction
test_dir = './test_images/'
save_dir = './drawn_lines_images/'
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
    abs_sob_binary = mt.mag_thresh(undist, sobel_kernel=3, mag_thresh=(80,120))

    #hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    #H = hls[:,:,0]
    #L = hls[:,:,1]
    #S = hls[:,:,2]

    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Yellow threshold
    yellow = cv2.inRange(HSV, (0, 80, 100), (40, 255, 255))

    # White threshold
    sensitivity_1 = 50
    white = cv2.inRange(HSV, (0, 0, 255 - sensitivity_1), (255, 20, 255))

    sensitivity_2 = 35
    HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    white_2 = cv2.inRange(HLS, (0, 255 - sensitivity_2, 0), (255, 255, sensitivity_2))
    white_3 = cv2.inRange(img, (200, 200, 200), (255, 255, 255))

    color_binary = yellow | white | white_2 | white_3

    #cv2.imshow('image',color_binary)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #color_binary = np.zeros_like(dir_grad_r)

    #color_binary[(L >= 60) & (L <= 120)] = 1
    #color_binary[(S >= 160) & (S <= 255)] = 1
    #ret = cv2.imwrite(test_dir + 'h_' + fname, H)
    #ret = cv2.imwrite(test_dir + 'l_' + fname, L)
    #ret = cv2.imwrite(test_dir + 's_' + fname, S)

    combined = np.zeros_like(color_binary)
    #combined[color_binary == 1] = 255
    #combined[(dir_grad_r == 1) | (dir_grad_l == 1)] += 100
    #combined[abs_sob_binary == 1] += 100
    #print(combined.shape)
    combined[((dir_grad_r == 1) | (dir_grad_l == 1)) & (abs_sob_binary == 1)] = 255
    combined = combined | color_binary

    #undist[combined == 1] = 255
    #undist[combined == 0] = 0

    #ret = cv2.imwrite(test_dir + 'dir_grad_color_' + fname, combined)

# TODO: 4. Perspective transform
    warped = warp_img.warp(combined)
    #cv2.imwrite(test_dir + 'warped_test_' + fname, warped)


# TODO: Detect lane lines

    histogram = np.sum(warped[warped.shape[0]/2:,:], axis=0)
    #plt.plot(histogram)
    #plt.title(fname)
    #plt.show()
    out_img = np.dstack((warped, warped, warped))*255
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9

    window_height = np.int(warped.shape[0]/nwindows)
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    margin = 100

    minpix = 50

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):

        win_y_low = warped.shape[0] - (window + 1)*window_height
        win_y_high = warped.shape[0] - window*window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    # The sliding window search seemed to perform worse than the histogram approach
    window_width = 50 # Windows for searching for lane lines in
    window_height = 80 # 9 vertical layers since image height is 720
    margin = 100 # Margin to slide left and right for searching

    window_centroids = ws.find_window_centroids(warped,window_width, window_height, margin)
    out_img = ws.mask(window_centroids,warped)
    cv2.imwrite(save_dir + 'lines_masked_' + fname, out_img)

# TODO: Determine lane curvature
