import cv2
import os
import numpy as np
import matplotlib.image as mpimg

def calib():
    dir_list = os.listdir(path='./camera_cal/')

    calib_img = []
    corners = []

    objpoints = []
    imgpoints = []

    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    for fname in dir_list:
        calib_img = cv2.imread('./camera_cal/' + fname)
        gray = cv2.cvtColor(calib_img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        if ret:

            imgpoints.append(corners)
            objpoints.append(objp)

            calib_img = cv2.drawChessboardCorners(calib_img, (9,6), corners, ret)
            #cv2.imwrite('./camera_cal/test' + fname,calib_img)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    '''
    for fname in dir_list:
        calib_img = cv2.imread('./camera_cal/' + fname) # Take calibration images, undistort them, and save them.
        dst = cv2.undistort(calib_img, mtx, dist, None, mtx)
        cv2.imwrite('./camera_cal/' + 'undist' + fname, dst)
    '''
    return mtx, dist
