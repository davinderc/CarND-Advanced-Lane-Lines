import cv2
import calib
import warp_img
import mult_thresh as mt
import window_search as ws
import adv_line as al
import os
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
#from pylab import *

vidfile = './project_video.mp4'

output_vid = './test.mp4'

mtx, dist = calib.calibrate()

clip1 = VideoFileClip(vidfile)

frames = [551,552,553,554,555]

for i in frames:
    test_img = clip1.get_frame(i)
    result = al.process_frame(test_img)

#test_clip = clip1.fl_image(al.process_frame)

#test_clip.write_videofile(output_vid, audio=False)
