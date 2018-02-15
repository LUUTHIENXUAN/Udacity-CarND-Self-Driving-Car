## Writeup Template

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. Apply a distortion correction to raw images.
3. Use color transforms, gradients, etc., to create a thresholded binary image.
4. Apply a perspective transform to rectify binary image ("birds-eye view").
5. Detect lane pixels and fit to find the lane boundary.
6. Determine the curvature of the lane and vehicle position with respect to center.
7. Warp the detected lane boundaries back onto the original image.
8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## [Rubric Points](https://review.udacity.com/#!/rubrics/571/view) 

### Camera Calibration

 1. #### Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.
	 The code for this step is contained in the code cell No.2 of the IPython notebook. 
	 ```python
	 class Undistortion:
	    def __init__(self, image_direction, 
	                 chessboard_nx, chessboard_ny):
	       ...
	    def camera_calibration(self):
	       ...
	    def save_data(self):
	       ...
	    def get_data(self):
	       ...
	    def undistorted_image(self, img):
	       ... 
	 ```   
	I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

	I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
	![Before distortion](https://github.com/LUUTHIENXUAN/Udacity-CarND-Advanced-Lane-Lines-P4/blob/master/camera_cal/calibration1.jpg)
![After distortion](https://github.com/LUUTHIENXUAN/Udacity-CarND-Advanced-Lane-Lines-P4/blob/master/After_distortion.png)


### Pipeline (single images)

 1. #### Provide an example of a distortion-corrected image.

	To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
	![enter image description here](https://github.com/LUUTHIENXUAN/Udacity-CarND-Advanced-Lane-Lines-P4/blob/master/distortion_correction_test_image.png)

 2. #### Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

	I *did not use a combination of color and gradient thresholds* to generate a binary image because color thresholds were sensitive to environment. Even though I tried various color space like: `HSL`, `HSV`, `LUV`, `LAB`, `YUV` in code cell No.13, their thresholds adjustment is hard to adapt globally and easy to fail for example in harder challenge video.  
	```python
	def color_gradient(img, combine = False):
	    gradient = gradient_threshold(img)
	    color = color_threshold(img)
	    if combine == True:
	        ...
	    else:
	        ...
	```
	Instead, I *found out a simple way using `Sobel` threshold only*, which could extract exactly line pixels in various environment conditions. The idea is that : 
	> Black-to-White transition is taken as Positive slope (it has a
	> positive value) while White-to-Black transition is taken as a Negative slope (It has negative value)	
	
	![Detect edges](https://docs.opencv.org/3.0-beta/_images/double_edge.jpg)
	
	Using negative slope (White-to-Black) only can help reduce noises such as shadows lines on roads.
	```python
	def abs_sobel_thresh(img, orient='x',sobel_kernel=3, abs_thresh=(0,255)):
    
    # and take the absolute value
	    if orient == 'x':
	        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	        sobelx[sobelx > 0] = 0
	        abs_sobel = np.absolute(sobelx)
    return binary_output
	```
	To detect white lines, I used combination of `LUV`, `YUV`. For yellow lines, `Lab` was chosen as following:
	```python
		img = cv2.medianBlur(img,5)
        l_channel = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)[:,:,0]
        y_channel = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)[:,:,0]
        white  = cv2.addWeighted(l_channel, 0.5, y_channel, 0.5, 0)
        yellow = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:,:,2]
        
        white_threshold  = sobel_thres(white  ,thres =(30,255), more_gradient = False)
        yellow_threshold = sobel_thres(yellow  ,thres =(30,255), more_gradient = False)
        
        combined_binary = np.zeros_like(yellow)
        combined_binary[(yellow_threshold ==1)|(white_threshold ==1)] = 1
    ``` 	
	Here's an example of my output for this step. 
	![Gradient threshold](https://github.com/LUUTHIENXUAN/Udacity-CarND-Advanced-Lane-Lines-P4/blob/master/gradient_thres.png)
	![Even more gradient threshold](https://github.com/LUUTHIENXUAN/Udacity-CarND-Advanced-Lane-Lines-P4/blob/master/evenmore_gradient_thres.png)

 3. #### Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

	The code for my perspective transform includes a function called `perspective_transform(self, img)`, which appears in the 6th code cell of the IPython notebook).  
	```python
	class Transform():
	    def __init__(self):
	        ...
	    def save_data(self, img, coords):
	        ...
	    def get_data(self):
	        ...
	    def perspective_transform(self, img):
	        ...
	```
	
	The `perspective_transform(self, img)` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:
	```python
	 self.src = np.float32([bottom_left, top_left,
	                        top_right,bottom_right])
     offset   = 80 # offset for dst points
     self.dst = np.float32([[offset, img_size[1]-offset],
                            [offset,offset],
                            [img_size[0]-offset,offset],
                            [img_size[0]-offset,
                             img_size[1]-offset]])
	```

	I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image as following:
	![Perspective transform](https://github.com/LUUTHIENXUAN/Udacity-CarND-Advanced-Lane-Lines-P4/blob/master/perspective_transform.png)

 4. #### Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
	 Take a histogram along all the columns in the lower part of the image, so the two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines. 
	 Use those indicators as starting points for where to search for the lines. 
	 From starting points, use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame. 
	 Windows which have more than `minpix` pixels would be considered as part of lines and stored  those pixels into left and right indices respectively.
	  Extract left and right line pixel positions and fit a second order polynomial to each 
	```python
	# peaks in a Histogram
	def histogram_sldwindows(binary_warped, viz = False):
		...
		# "expected non-empty vector for x" error
	    if (np.sum(leftx) !=0) & (np.sum(rightx) !=0):
	        non_empty = True
	        # Fit a second order polynomial to each
	        left_fit = np.polyfit(lefty, leftx, 2)
	        right_fit = np.polyfit(righty, rightx, 2)
	```
	From the next frame, search in a margin around the previous line 's starting points and do the same way as descrived above.
	```python
	def quickfind_lines(binary_warped,left_fit, right_fit):
    
	    return leftx, lefty, rightx, righty,\
	           left_fit, right_fit 
	``` 
	Here is some test images:
	![identify lane-line pixels and fit their positions](https://github.com/LUUTHIENXUAN/Udacity-CarND-Advanced-Lane-Lines-P4/blob/master/fit_line.png)
 6. #### Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
	I calculated the radius of curvature of the lane and the position of the vehicle with respect to center in code cell No.27 and No.28:
	```python
	# Road properties: compute the radius of curvature
	def curvature_measure(img,leftx,lefty,rightx,righty,
	                      ym_per_pix = 30/720,
	                      xm_per_pix = 3.7/700 ):
		...
	    return left_curverad,right_curverad
	```
	
	```python
	def offset_measure(img, left_fit, right_fit,
	                   xm_per_pix = 3.7/1000):
	    ...
	    return left_distance, right_distance
	```
	

 7. #### Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

	I implemented this step in code cell No.31.
	```python
	def pipline_image(frame):
		...
	    return result #write_offset
    ``` 
	Here is an example of my result on a test image:
	
	![Pipline image](https://github.com/LUUTHIENXUAN/Udacity-CarND-Advanced-Lane-Lines-P4/blob/master/pipline_image_test.png)

---

### Pipeline (video)

 1. #### Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

	Here's a [link to my harder challenge video ](https://www.youtube.com/watch?v=8du21USFQ00)
	Here's a [link to my challenge video](https://www.youtube.com/watch?v=i4-9PxyFGq4)
	Here's a [link to my project video](https://www.youtube.com/watch?v=tjtGp2oVaUE)

---

### Discussion

 1. #### Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
	 **Line Pixels Detection**
	 It is hard to adapt one method that could detect line pixels in various environments detections. I experienced a lot of color spaces such as `HSL`, `HSV`, `LUV`, `LAB`, `YUV`, also their combinations. Final combinations were  `LUV`, `YUV`for white lines, `Lab`  for yellow lines. Their thesholds were fixed in `(30,255)` which was good enough even in `harder_challenge_video`
	 ```python
	 white_threshold  = sobel_thres(white  ,thres =(30,255), more_gradient = False)
     yellow_threshold = sobel_thres(yellow ,thres =(30,255), more_gradient = False)
	```
	 **Line Pixels Extraction**
	 I made some modifications in ***Line Finding Method: Peaks in a Histogram*** to make sure you can find the lines as robustly as possible.
	 First, the road may contains a lot of noises such as shadows pixels from trees so that taking a histogram of the **lower half** of the image would be failed in harder challenge video. In face, the lower part of the image, the more robust starting points are chosen. The way of choosing starting points as following:
	 ```python
	 """STARTING POINT FINDING OPTIMIZATION """
      # Assuming we created a warped binary image called "binary_warped"
      # Take a histogram of the bottom part of the image, its size changed as below.
      # Find the peak of the left and right halves of the histogram would be possible starting point
      # Note: The smaller size chosen, the more exact starting point could be chosen for solid lines.
      # On the other hand, it would be noise for dash lines. 
      
    left_pos, right_pos = [],[] #starting point position 
    for i in reversed(range(2,6)):
        histogram = np.sum(binary_warped[binary_warped.shape[0]- binary_warped.shape[0]//i:,:], axis=0)
        midpoint = np.int(histogram.shape[0]/2)
        left_pos.append(np.argmax(histogram[:midpoint]))
        right_pos.append(np.argmax(histogram[midpoint:])+ midpoint)
    ...
    """STARTING POINT FINDING OPTIMIZATION END"""
    ```
    Secondly, in one frame **recentering next window on their mean position at current window** would be failed in heavy curves so dynamic adjustment should be included. Dynamic adjustment's calculation is as following:
    ```python
	   if len(good_left_inds) > minpix:
            if left_recenter == True:
                dst1 = np.int(np.mean(nonzerox[good_left_inds])) - leftx_current
                leftx_current = np.int(np.mean(nonzerox[good_left_inds])) + dst1
            else: leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            if right_recenter == True:
                dst2 = np.int(np.mean(nonzerox[good_right_inds])) - rightx_current
                rightx_current = np.int(np.mean(nonzerox[good_right_inds])) + dst2
            else: rightx_current = np.int(np.mean(nonzerox[good_right_inds]))  
      ```
            
	**Sanity Check**
	I made some checking function to make sure detected lane lines are real:
	Checking that they have similar curvature:
	```python
	#Checking that new lines have similar curvature
    diffs_curvature = np.absolute(new_curvature_left -\
                                  new_curvature_right)/(new_curvature_left +\
                                                        new_curvature_right)
	```
	Checking that they are separated by approximately the right distance horizontally
	```python
	 separation = [x - y for x, y in zip(right_line.line_pos(binary_warped),\
                                        left_line.line_pos(binary_warped))]
	```
	Checking that they are roughly parallel with similar slope.
	```python
	slope = (np.absolute(left_line.current_fit[0]), np.absolute(right_line.current_fit[0]))
    diffs_slope = np.absolute((slope[0] - slope[1])/((slope[0] + slope[1])))
	```
