# Udacity-CarND-Self-Driving-Car
Udacity Self-Driving Car Engineer Nanodegree projects.<br>
*Image source: Udacity*

## Overview

<p align="center">
  <img src="https://readrise.s3.amazonaws.com/m/content/140/thumb_1406762.png" />
</p>
<p align="center">Self Driving Car System<p align="center">

The self-driving car system comes down with 3 main parts as Mapping (**Computer vision**, **Sensor fusion** and **Localization**), **Path planning** and **Control**. 
  - The mapping part helps the self-driving car figure out its environment and where it is in the world by using vairous sensor devices such as Camera, Lidar , Radar, Gps and even the previously constructed map. 
    - **Computer vision** is how to use camera to detect lane lines, traffic signs, and track objects such as vehicles and pedestrians. 
    - **Sensor vision** is how to intergate data from other sensor as Lidar and Radar together with Camera data to re-build self-driving car's 3D surrounding environment. Lidar is used to detect objects by clustering its point cloud while using Radar to guess those objects' velocity. This part needs the calibration between those sensors to label each 3D object from **Computer vision** and map all of them to the local map. Note that Lidar often fail to detect objects in case of rainy day, or multiple objects stayed closely, etc.
    - **Localization** is how to localize the car in the real world. The car could use GPS or Lidar with previously constructed map to detect its localization. Highly accurate RTK-GPS method could be accurate within 20 cm but it easily fails in cities with high buildings. Lidar with previously constructed map method is suited with local map but not global map. The bigger Lidar's map the higger error accumulated when constructing map. 
  - The **path planning** part is to build a trajectory to run the car to its destination. This part considers tracked objects from **sensor vision** and road rules like traffic sign detection from **Computer vision** to decide the safe, comfortable and compliance action (maneuver).
  - The **control** part is to make the car turn the steering wheel, hit the throttle or the brake, in other to follow the trajectory from **path planning** part as closely as possible.
  
### Projects

#### TERM 1: Computer Vision and Deep Learning

<!DOCTYPE html>
<html>
<head>
  <!-- meta charset="UTF-8" -->
  <!-- meta name="viewport" content="width=device-width, initial-scale=1.0" -->
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
</head>
<body>
<table style="width:100%">
  <tr>
    <th>
      <p align="center">
           <a href="./Udacity-CarND-P01-LaneLines"><img src="https://s3.amazonaws.com/video.udacity-data.com/topher/2018/February/5a8222a9_1-2-project-finding-lane-lines2x/1-2-project-finding-lane-lines2x.jpg" alt="Overview" width="60%" height="60%"></a>
           <br style="font-size:10vw;">P01: Finding Lane Lines </br> 
      </p>
    </th>
    <th><p align="center">
           <a href="./Udacity-CarND-P02-Traffic-Sign-Classifier"><img src="https://s3.amazonaws.com/video.udacity-data.com/topher/2018/February/5a822410_1-9-project-traffic-sign-classifier2x/1-9-project-traffic-sign-classifier2x.jpg" alt="Overview" width="60%" height="60%"></a>
           <br style="font-size:10vw;">P02: Traffic Sign Classifier </br>
        </p>
    </th>
    <th><p align="center">
           <a href="./Udacity-CarND-P03-Behavioral-Cloning"><img src="https://s3.amazonaws.com/video.udacity-data.com/topher/2018/February/5a82244a_1-12-project-behavioral-cloning2x/1-12-project-behavioral-cloning2x.jpg" alt="Overview" width="60%" height="60%"></a>
           <br style="font-size:10vw;">P03: Behavioral Cloning </br>
        </p>
   </th>
  </tr>
  <tr>
   <th><p align="center">
           <a href="./Udacity-CarND-P04-Advanced-Lane-Lines"><img src="https://s3.amazonaws.com/video.udacity-data.com/topher/2018/February/5a822474_1-13-project-advanced-lane-finding2x/1-13-project-advanced-lane-finding2x.jpg"                         alt="Overview" width="60%" height="60%"></a>
           <br style="font-size:10vw;">P04: Ad. Lane Finding </br>
        </p>
    </th> 
    <th><p align="center" style="font-size:10vw;">
           <a href="./Udacity-CarND-P05-Vehicle-Detection"><img src="https://s3.amazonaws.com/video.udacity-data.com/topher/2018/February/5a8224c3_1-17-project-vechicle-detection-and-tracking2x/1-17-project-vechicle-detection-and-tracking2x.jpg"                         alt="Overview" width="60%" height="60%"></a>
           <br style="font-size:10vw;">P05: Vehicle Detection </br>
        </p>
    </th>
  </tr>
</table>

</body>
</html>

#### TERM 2: Sensor Fusion, Localization, and Control

<!DOCTYPE html>
<html>
<head>
  <!-- meta charset="UTF-8" -->
  <!-- meta name="viewport" content="width=device-width, initial-scale=1.0" -->
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
</head>
<body>
<table style="width:100%">
  <tr>
    <th>
      <p align="center">
           <a href="./Udacity-CarND-P01-LaneLines"><img src="https://s3.amazonaws.com/video.udacity-data.com/topher/2018/February/5a8227c5_2-6-project-extended-kalman-filter-project2x/2-6-project-extended-kalman-filter-project2x.jpg" alt="Overview" width="60%" height="60%"></a>
           <br style="font-size:10vw;">P06: Extended Kalman Filters </br> 
      </p>
    </th>
    <th><p align="center">
           <a href="./Udacity-CarND-P02-Traffic-Sign-Classifier"><img src="https://s3.amazonaws.com/video.udacity-data.com/topher/2018/February/5a8227e8_2-8-unscented-kalman-filter-project2x/2-8-unscented-kalman-filter-project2x.jpg" alt="Overview" width="60%" height="60%"></a>
           <br style="font-size:10vw;">P07: Unscented Kalman Filters </br>
        </p>
    </th>
    <th><p align="center">
           <a href="./Udacity-CarND-P03-Behavioral-Cloning"><img src="https://s3.amazonaws.com/video.udacity-data.com/topher/2018/February/5a8228b1_2-15-project-kidnapped-vehicle-project2x/2-15-project-kidnapped-vehicle-project2x.jpg" alt="Overview" width="60%" height="60%"></a>
           <br style="font-size:10vw;">P08: Kidnapped Vehicle </br>
        </p>
   </th>
  </tr>
  <tr>
   <th><p align="center">
           <a href="./Udacity-CarND-P04-Advanced-Lane-Lines"><img src="https://s3.amazonaws.com/video.udacity-data.com/topher/2018/February/5a8228d4_2-17-project-pid-controller-project2x/2-17-project-pid-controller-project2x.jpg"                         alt="Overview" width="60%" height="60%"></a>
           <br style="font-size:10vw;">P09: PID Controller </br>
        </p>
    </th> 
    <th><p align="center" style="font-size:10vw;">
           <a href="./Udacity-CarND-P05-Vehicle-Detection"><img src="https://s3.amazonaws.com/video.udacity-data.com/topher/2018/February/5a822900_2-20-project-model-predictive-control-project2x/2-20-project-model-predictive-control-project2x.jpg"                         alt="Overview" width="60%" height="60%"></a>
           <br style="font-size:10vw;">P10: Model Predictive Control </br>
        </p>
    </th>
  </tr>
</table>

</body>
</html>

## Table of Contents

#### [P01 - Finding Lane Lines on the Road](https://github.com/LUUTHIENXUAN/Udacity-CarND-Self-Driving-Car/tree/master/Udacity-CarND-P01-LaneLines)
 - **Summary:** Detected highway lane lines on a video stream. Used OpencV image analysis techniques to identify lines, including Hough Transforms and Canny edge detection.
 - **Keywords:** Computer Vision, Python
 - **Comment from Udacity reviewer:** [LINK](https://review.udacity.com/#!/reviews/851338/shared) 
 
#### [P02 - Traffic Sign Classifier](https://github.com/LUUTHIENXUAN/Udacity-CarND-Self-Driving-Car/tree/master/Udacity-CarND-P02-Traffic-Sign-Classifier)
 - **Summary:** Built and trained a deep neural network to classify traffic signs, using TensorFlow. Experimented with different network architectures. Performed image pre-processing and validation to guard against overfitting.
 - **Keywords:** Computer Vision, Deep Learning, TensorFlow, Python
 - **Comment from Udacity reviewer:** [LINK](https://review.udacity.com/#!/reviews/911777/shared)
 
#### [P03 - Behavioral Cloning](https://github.com/LUUTHIENXUAN/Udacity-CarND-Self-Driving-Car/tree/master/Udacity-CarND-P03-Behavioral-Cloning)
 - **Summary:** Built and trained a convolutional neural network for end-to-end driving in a simulator, using TensorFlow and Keras. Used optimization techniques such as regularization and dropout to generalize the network for driving on multiple tracks.
 - **Keywords:** Deep Learning, Convolutional Neural Networks, Keras, Python
 - **Comment from Udacity reviewer:** [LINK](https://review.udacity.com/#!/reviews/969487/shared)
 
#### [P04 - Advanced Lane Finding](https://github.com/LUUTHIENXUAN/Udacity-CarND-Self-Driving-Car/tree/master/Udacity-CarND-P04-Advanced-Lane-Lines)
 - **Summary:** Built an advanced lane-finding algorithm using distortion correction, image rectification, color transforms, and gradient thresholding. Identified lane curvature and vehicle displacement. Overcame environmental challenges such as shadows and pavement changes.
 - **Keywords:** Computer Vision, OpenCV, Python
 - **Comment from Udacity reviewer:** [LINK](https://review.udacity.com/#!/reviews/1011619/shared)

#### [P05 - Vehicle Detection and Tracking](https://github.com/LUUTHIENXUAN/Udacity-CarND-Self-Driving-Car/tree/master/Udacity-CarND-P05-Vehicle-Detection)
 - **Summary:** Created a vehicle detection and tracking pipeline with OpenCV, histogram of oriented gradients (HOG), and support vector machines (SVM). Optimized and evaluated the model on video data from a automotive camera taken during highway driving.
 - **Keywords:** Computer Vision, Histogram of Oriented Gradients, Machine Learning, OpenCV, Support Vector Machines, Python
 - **Comment from Udacity reviewer:** [LINK](https://review.udacity.com/#!/reviews/1009726/shared)
 
 #### [P06 - Vehicle Detection and Tracking](https://github.com/LUUTHIENXUAN/Udacity-CarND-Self-Driving-Car/tree/master/Udacity-CarND-P05-Vehicle-Detection)
 - **Summary:** Implemented an Extended Kalman Filter algorithm in C++ capable of tracking a pedestrian's motion in two dimensions.
 - **Keywords:** C++
 - **Comment from Udacity reviewer:** [LINK](https://review.udacity.com/#!/reviews/1095502/shared)
