import cv2
import os
import numpy as np


def warp():
    test_dir = './test_images/'
    test_fnames = os.listdir(path=test_dir)

    for fname in test_fnames:
        img = cv2.imread(test_dir + fname)
        img_size = (img.shape[1], img.shape[0])

        src = np.float32(
        [[612, 440],
         [667, 440],
         [1007, 655],
         [299, 655]])

        dst = np.float32(
        [[486, 284],
         [812, 284],
         [812, 655],
         [486, 655]])

        M = cv2.getPerspectiveTransform(src,dst)
        Minv = cv2.getPerspectiveTransform(dst,src)

        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        cv2.imwrite(test_dir + 'warped_' + fname,warped)
        #return warped
warp()
