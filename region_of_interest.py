import numpy as np
import cv2

def roi(img, vertices):
    mask =  np.zeros_like(img)
    if(len(img.shape) > 2):
        channel_count = img.shape[2]
        ignore_mask_color = (255,)*channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
