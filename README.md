# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[hog1]: ./images_for_report/HOG_1.png "HOG"
[hog2]: ./images_for_report/HOG_2.png "HOG"
[car]: ./images_for_report/car.png "Car"
[non_car]: ./images_for_report/non_car.png "Non car"
[bounding_boxes_all]: ./images_for_report/bounding_boxes_all.png "All bounding boxes"
[vehicle_bounding_boxes]: ./images_for_report/vehicle_bounding_boxes.png "Vehicle bounding bounding_boxes_all"
[heatmap]: ./images_for_report/heatmap.png "Heatmap"
[final_bounding_boxes]: ./images_for_report/vehicle_final_bounding_boxes.png "Final bounding boxes"

## Code

The implementation of this project can be found in the iPython notebook `Vehicle_Detection.ipynb`.

## Classifier

A linear SVM classifier was used to classify vehicles. The feature set used for the classifier were:

* HOG features on the YCrCb colour space
* Colour histogram consisting of 32 bins of each dimension of the YCrCb colour space
* 32 x 32 spatial bins in the YCrCb colour space

The parameters for the HOG features were

* pixels per cell - 8 x 8
* cells per block - 2
* orientation - 9

Different colour spaces were trialed and the HOG features for those colour spaces were visualised. In combination with the results of training the classifier with each colour space, it was found that YCrCb provided the best results in visualising and classifying vehicles and so was used for the model. 

Different parameters for the HOG features were also trialled and the parameter set above was chosen as they produced the lowest error in classification. 

A visualisation of the HOG features can be seen below:

![alt text][hog1]

![alt text][hog2]

The classifier was trained in the second code cell block. GTI and KITTI vehicle and non-vehicle data were used for training and testing. An example of each is shown below:

![alt text][car]

![alt text][non_car]

HOG features, spatial bin and colour histogram features on the YCrCb colour space were extracted from each image. The feature set was normalised using `StandardScaler` from the `sklearn` Python package. The data was dividded using an 80/20 split into training and test data. A linear SVM was trained with a test accuracy of 99.16%. 


## Sliding Window Search

A sliding window was used to find vehicles in an image. The function slide_window() generates windows of given size and overlap across a specified region. To improve efficiency of the HOG feature calculation, the function find_cars() is used which calculates HOG features for the entire image and then subsamples it. The parameter `scale` in the function represents the scale at which to downsample the image. 

Due to perspective, vehicles further away appear smaller in the image. Since the vehicle images in the training set are tight around the vehicle or smaller, scales were chosen to produce bounding box images of vehicles similar to those that the model was trained on. 

The following scales were used:

* scale of 1 between y = 400 and y = 500 with 2 cells per step
* scale of 1.5 between y = 400 and y = 500 with 2 cells per step
* scale of 1.75 between y = 450 and y = 600 with 1 cell per step
* scale of 2 between y = 450 and y = 600 with 1 cell per step

The search was restricted to y = 400 to y = 600 since cars are not observed above the horizon or too close to our vehicle. More than one scale was used for each region to try to capture vehicles at different distances and thus of different sizes in the image. A moderate level of overlap was used so that for each vehicle in the image, there would be at least two bounding boxes that captured a sizeable portion of the vehicle. (Heatmap threshold of two was used to prevent false positives)

A visualisation of all bounding boxes and classifications are shown below.

![alt text][bounding_boxes_all]

![alt text][vehicle_bounding_boxes]

Initially many false positives were observed. I hypothesised that a more powerful model was necessary and so more features (including spatial binning) was added to the SVM feature vector. 

Due to overlapping bounding boxes, we tend to get multiple classifications for each vehicle in the image. Sometimes false positives are also observed. To produce more reliable car detections and one detection for each car, a heat map is used. A threshold of two was applied to the heatmap to reduce false positives and `scipy.ndimage.measurements.label()` was used to identify individual blobs (assumed to be individual vehicles) in the heatmap. Bounding boxes were constructed around these blobs to produce the final vehicle detections. 

![alt text][heatmap]

![alt text][final_bounding_boxes]

## Discussion

Some problems experienced in my implementation include wobbly bounding boxes and the occasional false positive. To account for this, a more powerful model could be used. Perhaps a convolutional neural network such as YOLO or ensemble methods such as random forests may produce better results. 

To stabilise bounding boxes, smoothing could be performed across frames. A heatmap approach similar to the one used for one frame could be used across frames. However, a potential issue with doing this may be a lag in detections. 

A hypothetical case that may cause the pipeline to fail could be vehicles not specifically trained for. Larger vehicles such as trucks were not observed in the image and so may not be correctly classified. 