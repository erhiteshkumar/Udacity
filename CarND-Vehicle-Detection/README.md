
# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


[![Vehicle detection and tracking]](https://youtu.be/IyFqdHMh8Nk)

This is my solution to project 5 of Udacity self-driving car nanodegree. The goal is to detect and track cars on video stream.

Code:
- All commented code can be found at VehicleDetection.ipynb jupyter notebook.
- test_images/ contains images of road to test and finetune pipeline
- test_video.mp4, project_video.mp4 - videos for pipeline testing
- data/ - folder for datasets unpacking

# Overview

Project code consist of following steps:

1. Load datasets
2. Extract features from datasets images
3. Train classifier to detect cars. (I use simple default SVM with rbf kernel)
4. Scan video frame with sliding windows and detect hot boxes
5. Use hot boxes to estimate cars positions and sizes
6. Use hot boxes from previous steps to remove false positives hot boxes and make detection more robust

# Datasets

In this project I use two datasets. First is project dataset. It is splitted into [cars images](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-car images](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip). Here is examples of dataset images:


Steps for vehicle detection are as follow:

    1. Feature extraction and data split. Following are the steps:
       a. Car and non-car image data is downloaded from GTI vehicle image database and KITTI vision benchmark suite
       b. Extract features and combine them
       c. Split data in training and test
    2. Build classifier to predict images based on features, using LinearSVM. 
    3. Build pipeline to detect cars. Pipeline has following steps:
       a. Find all rectangles which has cars in image
       b. Draw heatmap with identified boxes
       c. Remove false positives by applying thresolds
       d. Label the heatmap image to identify number of cars in frame
       e. Draw identified cars areas on original image
    4. Run pipeline on video
       Run the above pipleine for every frame of video.

You can see all details in VehicleDetection.ipynb(https://github.com/erhiteshkumar/Udacity/blob/master/CarND-Vehicle-Detection/VehicleDetection.ipynb) <br/>

Images are shown after each step as required.

# Video Implementation
The youtube video of the final implementation can be accessed by clicking the following link (https://youtu.be/IyFqdHMh8Nk)

# Conclusion

Probems:
	1. Identifying parameters for extracting HOG features. Did some trail and error using multiple 
	   color_spaces and orient.
	2. Figuring out a way to filter false positives. HOG sub-sampling was better solution but I tried
	   to figure out other solutions
	3. Pipeline will fail in cases where images doesn't resemble training set like more-lighting or
	   different weather conditions like snow or rain.
Future improvements:
	Due to limited bandwidth for this project I haven't explored to append all three features(HOG, binned color and color histogram). I believe that HSV(all channels) or H(trading speed with accuracy) with all there features(HOG,binned_color,color_histogram) should give better results than my current approach. This will improve classifer accuracy and will give more smoother object identification. Using CNN to instead of sliding window search and CNN as a feature extractor.
