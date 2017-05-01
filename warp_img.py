import cv2
import numpy as np


def warp():
    img = cv2.imread('./test_images/straight_lines1.jpg')
    img_size = (img.shape[1], img.shape[0])

    src = np.float32(
    [[534, 494],
     [754, 494],
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
    cv2.imwrite('./test_images/warpedImg1.jpg',warped)
warp()
