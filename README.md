# Project 7: Unscented Kalman Filter Project

## 1. Introduction

This project utilizes a kalman filter to estimate the state of a moving object of interest with noisy lidar and radar measurements. With Unscented Kalman Filters, non-linear motion would be able to accurately tracked!

## 2. Results & Discussion

Below is the sensor fusion general flow by using Unsented Kalman Filter (UKF)
<p align="center">
  <img src="./pics/EKF_outline.png" />
</p>
<p align="center">Sensor Fusion general flow<p align="center">

### Paramaters Initialization 
Initial State and x and Covariance Matrix P were setup as following:

<!---
$$x =\begin{pmatrix}
{0.001}\\
{0.001}\\
{0.001}\\
{0.001}
\end{pmatrix}$$ >
-->

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space;x=\begin{pmatrix}{0.001}\\{0.001}\\{0.001}\\{0.001}\end{pmatrix}" />
</p>

<!---
$$P =\begin{pmatrix}
0.0&0.0&0.0&0.0&0.0\\
0.0&0.0&0.0&0.0&0.0\\
0.0&0.0&0.5&0.0&0.0\\
0.0&0.0&0.0&0.5&0.0\\
0.0&0.0&0.0&0.0&0.5
\end{pmatrix}$$
-->

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space;P=\begin{pmatrix}0.0&0.0&0.0&0.0&0.0\\0.0&0.0&0.0&0.0&0.0\\0.0&0.0&0.5&0.0&0.0\\0.0&0.0&0.0&0.5&0.0\\0.0&0.0&0.0&0.0&0.5\end{pmatrix}" />
</p>

Longitudinal acceleration and yaw acceleration noise measurement were set up as follow:  
- `std_a_`: 1.8 *(process noise standard deviation longitudinal acceleration in m/s^2)*. 
- `std_yawd_`: 0.7 *(process noise standard deviation yaw acceleration in rad/s^2)*. 

### Result

The following images show the final RMSE (Root Mean Squared Error ) values of both datasets by combining Lidar and Radar sensors. Error is the difference between actual measurement and its estimated value. The smaller RMSE the higher accuracy archived.

Dataset 1             |  Dataset 2 
:-------------------------:|:-------------------------:
<img src="./pics/UKF_dataset1.png" width="500">  |   <img src="./pics/UKF_dataset2.png" width="500"> 

When turning off one of those sensors, it affects the accuracy so bad.

Without Lidar sensor

Dataset 1             |  Dataset 2 
:-------------------------:|:-------------------------
<img src="./pics/UKF_dataset1_withoutLIDAR.png" width="500">  |   <img src="./pics/UKF_dataset2_withoutRADAR.png" width="500"> 

Without Radar sensor

Dataset 1             |  Dataset 2 
:-------------------------:|:-------------------------:
<img src="./pics/UKF_dataset1_withoutRADAR.png" width="500">  |   <img src="./pics/UKF_dataset2_withoutRADAR.png" width="500"> 

Overall, the UKF does a good job for predicting location of tracking objects. Lidar (Laser) measures object location precisely which help improve the location prediction while Radar gives better object's velocity estimation which therefore improve velocity prediction. 

### Normalized Innovation Squared (NIS)
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space;\varepsilon=(z_{k+1}-z_{k+1|k})^{T}.S_{k+1|k}^{-1}.(z_{k+1}-z_{k+1|k})" />
</p>



## 3. Set up environment
This project involves the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases)

This repository includes two files that can be used to set up and install [uWebSocketIO](https://github.com/uWebSockets/uWebSockets) for either Linux or Mac systems. 
For windows you can use either Docker, VMware, or even [Windows 10 Bash on Ubuntu](https://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/) to install uWebSocketIO. 
Please see [this concept in the classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77) for the required version and installation scripts.

Once the install for uWebSocketIO is complete, the main program can be built and run by doing the following from the project top directory.
```
1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./UnscentedKF
```

Tips for setting up your environment can be found [here](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/23d376c7-0195-4276-bdf0-e02f1f3c665d)

Note that the programs that need to be written to accomplish the project are src/FusionEKF.cpp, src/FusionEKF.h, kalman_filter.cpp, kalman_filter.h, tools.cpp, and tools.h

The program main.cpp has already been filled out, but feel free to modify it.

Here is the main protcol that main.cpp uses for uWebSocketIO in communicating with the simulator.

INPUT: values provided by the simulator to the c++ program
```C++
//the measurement that the simulator observed (either lidar or radar)
["sensor_measurement"] 
```
OUTPUT: values provided by the c++ program to the simulator
```C++
["estimate_x"] //kalman filter estimated position x
["estimate_y"] // kalman filter estimated position y
["rmse_x"]
["rmse_y"]
["rmse_vx"]
["rmse_vy"]
```

## 4. Other Important Dependencies

* cmake >= 3.5
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## 5. Basic Build Instructions

	1. Clone this repo.
	2. Make a build directory: `mkdir build && cd build`
	3. Compile: `cmake .. && make` 
	   * On windows, you may need to run: `cmake .. -G "Unix Makefiles" && make`
	4. Run it: `./UnscentedKF `

## 6. Editor Settings

We've purposefully kept editor configuration files out of this repo in order to
keep it as simple and environment agnostic as possible. However, we recommend
using the following settings:

* indent using spaces
* set tab width to 2 spaces (keeps the matrices in source code aligned)

## 7. Code Style

Please (do your best to) stick to [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html).

## 8. Generating Additional Data

This is optional!

If you'd like to generate your own radar and lidar data, see the
[utilities repo](https://github.com/udacity/CarND-Mercedes-SF-Utilities) for
Matlab scripts that can generate additional data.
