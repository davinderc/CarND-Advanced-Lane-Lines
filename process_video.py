import cv2
import calib
import warp_img
import mult_thresh as mt
import window_search as ws
import adv_line as al
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
#from pylab import *

vidfile = './project_video.mp4'

output_vid = './test_threshold.mp4'

mtx, dist = calib.calibrate()

clip1 = VideoFileClip(vidfile)

#frames = np.arange(0,50,5)
#frames = [39.0625]

#for i in frames:
    #test_img = clip1.get_frame(39.0625)
    #temp = test_img[:,:,0]
    #test_img[:,:,0] = test_img[:,:,2]
    #test_img[:,:,2] = temp
    #mpimg.imsave('./video_imgs/video_' + i.astype(np.str), test_img)
    #mpimg.imsave('./video_imgs/video_' + '39', test_img)
    #result = al.process_frame(test_img)

test_clip = clip1.fl_image(al.process_frame)

test_clip.write_videofile(output_vid, audio=False)
