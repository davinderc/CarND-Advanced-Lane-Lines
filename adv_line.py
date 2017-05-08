import cv2
import calib
import warp_img
import mult_thresh as mt
import window_search as ws
import os
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip


# TODO: 1. Camera calibration
mtx, dist = calib.calibrate()

# TODO: 2. Distortion correction
#test_dir = './test_images/'
#save_dir = './drawn_lines_images/'

#while(cap.isOpened()):
    #ret, frame = cap.read()
    #if ret == True:
def process_frame(img):
    #for fname in test_fnames:
    #fname = 'test1.jpg'
    #print(fname)
        #img = cv2.imread(test_dir + fname)
    mtx, dist = calib.calibrate()
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    #cv2.imwrite(test_dir + 'undist_' + fname, undist)

    # TODO: 3. Color/gradient threshold
    # TODO: Test other gradient thresholds to see what is going on

    #hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    #H = hls[:,:,0]
    #L = hls[:,:,1]
    #S = hls[:,:,2]

    HSV = cv2.cvtColor(undist, cv2.COLOR_BGR2HSV)

    YUV = cv2.cvtColor(undist, cv2.COLOR_BGR2YUV)
    U = cv2.inRange(YUV, (0, 145, 0), (255, 185, 255))
    V = cv2.inRange(YUV, (0, 0, 120), (255, 255, 140))

    # Yellow threshold
    yellow = (U - V).astype(np.uint8)

    # White threshold
    sensitivity_1 = 50
    white = cv2.inRange(HSV, (0, 0, 255 - sensitivity_1), (255, 20, 255))

    sensitivity_2 = 35
    HLS = cv2.cvtColor(undist, cv2.COLOR_BGR2HLS)
    white_2 = cv2.inRange(HLS, (0, 255 - sensitivity_2, 0), (255, 255, sensitivity_2))
    white_3 = cv2.inRange(undist, (200, 200, 200), (255, 255, 255))

    color_binary = yellow | white | white_2 | white_3

    #cv2.imwrite(test_dir + 'undist_' + fname, undist)

    # Threshold gradients with angles between 30 and 60, and between 120 and 150
    dir_grad_r = mt.dir_threshold(color_binary,sobel_kernel=3, thresh=(np.pi/6,np.pi/3))
    dir_grad_l = mt.dir_threshold(color_binary,sobel_kernel=3, thresh=(2*np.pi/3,5*np.pi/6))

    # Threshold gradients by magnitude
    abs_sob_binary = mt.mag_thresh(color_binary, sobel_kernel=5, mag_thresh=(100,160))

    combined = np.zeros_like(color_binary)

    combined[((dir_grad_r == 1) | (dir_grad_l == 1)) & (abs_sob_binary == 1)] = 255

    #ret = cv2.imwrite(test_dir + 'dir_grad_color_' + fname, combined)

# TODO: 4. Perspective transform.
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


    if(leftx.size == 0):
        left_fit = [0,0,0]
    else:
        left_fit = np.polyfit(lefty, leftx, 2)
    if(rightx.size != 0):
        right_fit = [0,0,0]
    else:
        right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    #print(right_fitx.astype(np.int))
    left_fitx_indices = np.copy(left_fitx).astype(np.int)
    right_fitx_indices =  np.copy(right_fitx).astype(np.int)
    y_indices = np.copy(ploty).astype(np.int)

    template = np.zeros_like(warped, np.uint8) # Add the fitted lines for left and right
    sane_left = (left_fitx_indices >= 0) & (left_fitx_indices < 1280)
    sane_right = (right_fitx_indices >= 0) & (right_fitx_indices < 1280)
    template[y_indices[sane_left],left_fitx_indices[sane_left]] = 255
    template[y_indices[sane_right], right_fitx_indices[sane_right]] = 255
    #template[y_ind_right_flat,right_fitx_flat] = 255
    zero_channel = np.zeros_like(template)
    template = np.array(cv2.merge((zero_channel, template, zero_channel)),np.uint8) # Green line
    #unwarped = warp_img.warp(template, mtx='Minv')


    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    #print(undist.shape, template.shape)
    overlay_in = warp_img.warp(undist)
    out_overlay = cv2.addWeighted(overlay_in, 0.8, template, 1, 0.0) # add overlay to observe fitted lines

    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    # The sliding window search seemed to perform worse than the histogram approach
    #window_width = 50 # Windows for searching for lane lines in
    #window_height = 80 # 9 vertical layers since image height is 720
    #margin = 100 # Margin to slide left and right for searching

    #window_centroids = ws.find_window_centroids(warped,window_width, window_height, margin)
    #out_img = ws.mask(window_centroids,warped)
    #cv2.imwrite(save_dir + 'lines_masked_' + fname, out_img)

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

    #print(type(l_radius))
    #font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(out_overlay, 'Left curvature: ', (10,10), font, 4, (200,180,200), 3, cv2.LINE_AA)
    #print('Radiuses for {} (L, R): {:.3f} m, {:.3f} m'.format(fname,l_radius,r_radius))
    #cv2.imwrite(save_dir + 'lines_overlayed_' + fname, out_overlay)


    video_frame = warp_img.warp(out_overlay, mtx='Minv')

    return video_frame
    #out.write(video_frame)
    #else:
        #break
#cap.release()
#out.release()
