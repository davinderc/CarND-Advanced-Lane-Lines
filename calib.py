import cv2
import os
import pickle
import numpy as np
import matplotlib.image as mpimg

def calibrate():

    calib_data_file = 'undistortData.pickle'
    if os.path.isfile(calib_data_file): # Check for pickle file and load undistortion matrices.
        with open(calib_data_file, 'rb') as calib_f:
            data = pickle.load(calib_f)
            mtx = data['mtx']
            dist = data['dist']
            del data
        return mtx, dist


    dir_list = os.listdir(path='./camera_cal/')

    calib_img = []
    corners = []

    objpoints = []
    imgpoints = []

    objp = np.zeros((6*9,3), np.float32) # Create grid of standard chessboard pattern
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    for fname in dir_list:
        calib_img = cv2.imread('./camera_cal/' + fname) # Load calibration images and grayscale them
        gray = cv2.cvtColor(calib_img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (9,6), None) # Find corners (54 of them)
        if ret:

            imgpoints.append(corners) # If they are found, the points are mapped between distorted and standard pattern
            objpoints.append(objp)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if not os.path.isfile(calib_data_file): # Save undistortion matrices in pickle file
        try:
            with open(calib_data_file, 'wb') as pfile:
                pickle.dump(
                    {
                        'mtx': mtx,
                        'dist': dist,
                    },
                    pfile, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(e)
            raise

    return mtx, dist
