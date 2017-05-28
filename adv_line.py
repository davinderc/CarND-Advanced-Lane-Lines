import cv2
import calib
import warp_img
import mult_thresh as mt
import window_search as ws
import region_of_interest as roi
import line
import os
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

# TODO: 1. Camera calibration


# TODO: 2. Distortion correction

def process_frame(img):
    global l_line
    global r_line
    global polygon
    global first_poly
    global old_pts

    mtx, dist = calib.calibrate()
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    # TODO: 3. Color/gradient threshold
    # TODO: Test other gradient thresholds to see what is going on

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
    masked = roi.roi(color_binary,vertices) # Mask a region of interest using a function previously created in P1

# TODO: 4. Perspective transform.
    warped = warp_img.warp(masked)

# TODO: Detect lane lines

    histogram = np.sum(warped[7*warped.shape[0]/10:], axis=0)

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
    lane_center = rightx_base - leftx_base # Calculate vehicle position w.r.t. center of lane
    car_dev = (undist.shape[1]/2) - lane_center

    margin = 90

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

    if(leftx.size == 0):
        left_fit = line.l_line.current_fit[0]
        line.l_line.detected = False
    else:
        line.l_line.detected = True
        left_fit = np.polyfit(lefty, leftx, 2)
        line.l_line.current_fit[0] = 0.3*left_fit + 0.7*line.l_line.current_fit[0]
    if(rightx.size == 0):
        right_fit = line.r_line.current_fit[0]
        line.r_line.detected = False
    else:
        line.r_line.detected = True
        right_fit = np.polyfit(righty, rightx, 2)
        line.r_line.current_fit[0] = 0.3*right_fit + 0.7*line.r_line.current_fit[0]

    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    left_fitx_indices = np.copy(left_fitx).astype(np.int)
    right_fitx_indices =  np.copy(right_fitx).astype(np.int)
    y_indices = np.copy(ploty).astype(np.int)

    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    sane_left = (left_fitx_indices >= 0) & (left_fitx_indices < 1280)
    sane_right = (right_fitx_indices >= 0) & (right_fitx_indices < 1280)

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    if(line.first_poly):
        cv2.fillPoly(line.polygon, np.int_([pts]),255)
        old_pts = pts
        line.first_poly = 0
    if(cv2.matchShapes(cv2.fillPoly(warp_zero, np.int_([pts]),255),line.polygon,1,0.0)<0.01):
        cv2.fillPoly(line.polygon, np.int_([pts]),255)
        old_pts = pts



    cv2.fillPoly(color_warp, np.int_([old_pts]), (0,255,0))
    newwarp = warp_img.warp(color_warp, mtx='Minv')
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

# TODO: Determine lane curvature
    ym_per_pix = 30/720 # meters per pixel in y
    xm_per_pix = 3.7/700 # meters per pixel in x
    y_eval = np.max(ploty)

    if(leftx.size == 0):
        left_fit_real = [0,0,0]
    else:
        left_fit_real = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    if(rightx.size == 0):
        right_fit_real = [0,0,0]
    else:
        right_fit_real = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    l_radius = ((1 + (2*left_fit_real[0]*y_eval*ym_per_pix + left_fit_real[1])**2)**1.5)/np.absolute(2*left_fit_real[0])
    r_radius = ((1 + (2*right_fit_real[0]*y_eval*ym_per_pix + right_fit_real[1])**2)**1.5)/np.absolute(2*right_fit_real[0])

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'Radiuses (L, R): ' + str(l_radius) +' m, ' + str(r_radius) + ' m'
    text2 = 'Car offset: ' + str(car_dev*xm_per_pix) + ' m'
    cv2.putText(result, text, (40,40), font, 1, (255,255,200), 1, cv2.LINE_AA)
    cv2.putText (result, text2, (40,80), font, 1, (255, 255, 200), 1, cv2.LINE_AA)

    return result
