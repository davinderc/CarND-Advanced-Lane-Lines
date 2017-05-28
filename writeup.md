## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undist_calibration2.jpg "Undistorted"
[image2]: ./output_images/undist_straight_lines2.jpg "Road Transformed"
[image3]: ./output_images/binary_video_0.jpg "Binary Example 1"
[image4]: ./output_images/binary_video_45.jpg "Binary Example 2"
[image5]: ./output_images/test_warp_1.jpg "Warp Example Original"
[image6]: ./output_images/warped_test_warp_1.jpg "Warp Example Output"
[image7]: ./output_images/histo_video_0.jpg "Detected pixels 1"
[image8]: ./output_images/histo_video_45.jpg "Detected pixels 2"
[image9]: ./output_images/overlaid_histo_video_0.jpg "Overlaid Detected pixels 1"
[image10]: ./output_images/overlaid_histo_video_45.jpg "Overlaid Detected pixels 2"
[image11]: ./output_images/fit_lines_video_0.jpg "Fit Lines 1"
[image12]: ./output_images/fit_lines_video_45.jpg "Fit Lines 2"
[image13]: ./output_images/detected_video_0.jpg "Detected Lane 1"
[image14]: ./output_images/detected_video_45.jpg "Detected Lane 2"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 'calib.py' file. I decided to separate different functions in this project as much as possible, to organize everything. This way I simply imported the necessary files in a main pipeline.

I start by opening a pickle file to load the data if it's ready, or to save it in for future use if not. Then I prepared the object points, which will be the (x, y, z) coordinates of the chessboard corners in the world. I assume the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. `calib_img` is simply an empty vector for the different calibration images in the given folder. The chessboards should have 54 corners in these images. Three of them failed detection because some corners were outside the image boundaries.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one (this image was corrected for distortion by the camera using the technique described previously):
![alt text][image2]

Distortion correction was performed in line 21 of `adv_line.py` immediately after obtaining the camera calibration data in line 20.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color thresholds to generate a binary image (thresholding steps at lines 31 through 49 in `adv_line.py`).

This was one of the most time-consuming parts of the project, since I ended up testing many combinations of color and gradient thresholds to try to get a robust detection that worked under many conditions. At one point I even attempted to use the openCV Laplacian function, as another student had recommended it over a simple gradient, but I found that for my purposes, it did not improve detection significantly and therefore I didn't use it. In fact, in the end I found the Sobel gradients to be much too noisy and opted only for color thresholding.

I ended up using RGB, HSV, HLS, and YUV colorspaces to detect both white and yellow lane lines. Yellow was best detected using thresholding of a combination of U and V channels from YUV, and thresholding in the HLS colorspace (lines 36-41). The white line was best detected using thresholding in HSV and RGB colorspaces (lines 44-47). In line 49 the binaries were combined.

A trapezoid shaped mask was also applied in lines 51-58 to select a region of interest and remove irrelevant pixels that are not part of the road.

Here's are a couple of examples of my output for this step.

![alt text][image3]

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in line 61 imported from the file `warp_img.py`.  The `warp()` function takes as inputs an image (`img`) and a string (defaulted to `M`) to choose whether to perform warping or unwarping, using `M` or `Minv`.  I chose to hardcode the source and destination points in the following manner:

```python
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
```

This could have been done more generally to provide source and destination points for any size of images, but I found it more important to spend my time in finding the proper color thresholding first and left this as it is. This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 612, 440      | 400, 20       |
| 667, 440      | 880, 20       |
| 1007, 670     | 880, 700      |
| 299, 670      | 400, 700      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. Although the lane lines in the test image are clearly parallel after warping the perspective, it seems the test points become nearly invisible due to the warping. In the original image, the black points are the source points and the red points are the destination points.

![alt text][image5]

![alt text][image6]

One odd thing is how the lines don't seem perfectly straight after warping, which is something that could be investigated to improve the algorithm.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In lines 65 to 139 I detected the lane line pixels and fit them to a second order polynomial. I used a histogram method to discover the base of the lines (where they started), using the bottom 15% of the image, since using a full half of the image ended up providing incorrect starting points when the lines curved.

Using this base and a window with set margin width, I divided up the height into 9 portions and found pixels in each portion moving upwards in the image, always recentering the window when I found enough pixels in the previous portion. I adjusted the margins and minimum number of pixels necessary to recenter the windows. Lines 91 to 110 detect pixels in each portion.

The detected pixels are shown in the following examples. In some cases it was surprisingly difficult to detect all the visible lane line pixels without introducing significant amounts of noise in other parts of the video.

Detected pixels:

![alt text][image7]

![alt text][image8]

Overlaid detected pixels:

![alt text][image9]

![alt text][image10]

The fit lane lines were calculated in lines 119 to 139, checking whether pixels were detected, and if not using the previous frames fitted lines. A 2nd order polynomial was used and is shown as follows:

![alt text][image11]

![alt text][image12]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did calculated the radius of curvature in lines 165 through 179 in my code in `adv_line.py`. I used a conversion factor to obtain the radius in meters. I calculated the position of the vehicle with respect to center in lines 80 and 81. In lines 181 to 185, these were overlaid on the images so that they would appear in the video.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 147 through 162 in my code in `adv_line.py`. The left and right line curvatures are fairly similar, although not the same, probably due to noise in pixel detection. I used openCV's matchShapes function to check that I wasn't getting ridiculously different detected lanes and if I did, to use the previous polygon to overlay on the lane, thus attenuating the noise a bit. Here is an example of my result on a test image:

![alt text][image13]

![alt text][image14]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/bUb7UW1oNiE)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Initially, using gradients to detect lane lines failed spectacularly, although it really looked great at the start. The gradients tended to magnify any noise created from shadows or pavement color variations. I spent a lot of time looking for different colorspace combinations to use to detect lane lines properly, and once I gave up on gradients I started to see significant improvements in detection. However, it still took me a while before I realized that I needed to use various colorspaces to detect the lane lines in various conditions.

In addition, not knowing all the different libraries that openCV and Numpy have meant it took longer before I discovered an easier way of doing things. There are probably a lot of different libraries and techniques that I haven't even heard of before that could help in improving my algorithm.

I think that my algorithm would fail spectacularly if it were shown tight curves, faded lane lines, poorly drawn lane lines, rough road conditions/colors, and significant variations due to shadows. I think I would need to learn a lot more about colorspaces to find ones that could make my algorithm more robust.

In addition, I attempted to use a weighted average to smooth out the detection, but did not take full advantage of all of the data I could use from previous frames' detections, and this could definitely help improve my lane detection.

Lastly, some students mentioned that they used Deep Learning to detect lane lines with great success. I think I will eventually try this technique out as well, since this method does not need me to pick out the perfect parameters, and can end up being more robust than someone inexperienced choosing the color parameters.
