# Advanced Lane Line Finding

This project is about building a lane line detector that is robust to changes in lighting conditions. 
It is part of the Udacity self-driving car Nanodegree. Please see the links below for details and the project requirements

* [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
* [rubric](https://review.udacity.com/#!/rubrics/571/view)

# Introduction
The steps of this project are as follows:  

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply the distortion correction to the raw image.  
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view"). 
* Detect lane pixels and fit to find lane boundary.
* Determine curvature of the lane and vehicle position with respect to center.
* Warping the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
* Run the entire pipeline on a sample video recorded on a sunny day on the I-280. 

---
[//]: # (Image References)

[image1]: ./output_images/test_calibration.jpg "Undistorted"
[image2]: ./output_images/undistort_image.jpg "Undistorted"
[image3]: ./output_images/binary.jpg "Binary Example"
[image4]: ./output_images/roi.jpg "Region of interest"
[image5]: ./output_images/project_lines.jpg "Projected lines"


In the following I will consider all steps individually and describe how I addressed each point in the implementation. 
The images for camera calibration are stored in the folder called `camera_cal/video`.  Images in `test_images` are for testing the pipeline on single frames.  Results are in the  `ouput_images` folder and subfolder `calibration`.

# Camera calibration 

For extracting lane lines that bend it is crucial to work with images that are distortion corrected. Non image-degrading abberations such as pincussion/barrel distortion can easily be corrected using test targets. Samples of chessboard patterns recorded with the same camera that was also used for recording the video are provided in the `camera_cal` folder. 
The code for distortion correction is contained in IPython notebook located in "./camera_calebration.ipynb" .  

We start by preparing "object points", which are (x, y, z) coordinates of the chessboard corners in the world (assuming coordinates such that z=0).  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

`objpoints` and `imgpoints` are then used to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
![Undistort][image1]

# Test Image Pipeline

## Example of a distortion corrected image
Applying the undistortion transformation to a test image yields the following result (left distorted, right corrected)
![Undistort][image2]
## Binary lane line image using gradient and color transforms

For color thresholding I worked in HLS space. Only the L and S channel were used. I used the s channel for a gradient filter along x and saturation threshold, as well as the l channel for a luminosity threshold filter. A combination of these filters
is used in the function `binarize` in the file `edge_detection.ipynb`. The binarized version of the image above looks as follows
![Binarized image][image3]


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. See the following image
![alt text][image4]

## Identifying lane line pixels using sliding windows
The function `find_peaks(img,thresh)` in cell of `edge_detection.ipynb` takes the bottom half of a binarized and warped lane image to compute a histogram of detected pixel values. The result is smoothened using a gaussia filter and peaks are subsequently detected using. The function returns the x values of the peaks larger than `thresh` as well as the smoothened curve. 

Then I wrote a function `get_next_window(img,center_point,width)` which takes an binary (3 channel) image `img` and computes the average x value `center` of all detected pixels in a window centered at `center_point` of width `width`. It returns a masked copy of img a well as `center`.

The function `lane_from_window(binary,center_point,width)` slices a binary image horizontally in 6 zones and applies `get_next_window`  to each of the zones. The `center_point` of each zone is chosen to be the `center` value of the previous zone. Thereby subsequent windows follow the lane line pixels if the road bends. The function returns a masked image of a single lane line seeded at `center_point`. 
Given a binary image `left_binary` of a lane line candidate all properties of the line are determined within an instance of a `Line` class. The class is defined in cell 11
``` 
    left=Line(n)
    detected_l,n_buffered_left = left.update(left_binary)
```
The `Line.update(img)` method takes a binary input image `img` of a lane line candidate, fits a second order polynomial to the provided data and computes other metrics. Sanity checks are performed and successful detections are pushed into a FIFO que of max length `n`. Each time a new line is detected all metrics are updated. If no line is detected the oldest result is dropped until the queue is empty and peaks need to be searched for from scratch. 

A fit to the current lane candidate is saved in the `Line.current_fit_xvals` attribute, together with the corresponding coefficients. The result of a fit for two lines is shown below.

![line fit][image5]

---

# Video Processing Pipeline

Code for Video Processing Ptpeline us in `edge_detection.ipynb`. The processed project video can be found here:
[link to my video result](https://youtu.be/y0f0Yk1EXZo)

---

# Discussion

## Conclusion

 Model doesn't perform very well and not to be accurate on bad wether conditions for if road contains shade. Below are few enhancements I think will improve the model tremondously.
 * Cut image
 * Use resion of interest
 * Tweak parameter of edege detection.
 * Better techneique to remove noise.
 * Build seprate lane line detector for yellow and white line togater with additional logic, will decide which line to choose