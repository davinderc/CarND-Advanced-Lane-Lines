import cv2
import os
import numpy as np


def warp(img, mtx='M'):
    #test_dir = './grad_test_images/'
    #test_fnames = os.listdir(path=test_dir)

    #for fname in test_fnames:
    #img = cv2.imread(test_dir + fname)
    img_size = (img.shape[1], img.shape[0])

    offset = 400

    src = np.float32(
    [[612, 440],
     [667, 440],
     [1007, 670],
     [299, 670]])

    dst = np.float32(
    [[offset, 20],
     [img_size[0] - offset, 20],
     [img_size[0] - offset, 700],
     [offset, 700]])

    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)

    if(mtx == 'M'):
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    elif(mtx == 'Minv'):
        warped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)
    #cv2.imwrite(test_dir + 'warped_' + fname,warped)
    return warped
#warp()
