
# **Project 1: Finding Lane Lines on the Road** 

**Finding Lane Lines on the Road**

This project aims to find lane markings on the road  by using computer vision techniques such as Canny Edge detection and Hough Transform line detection.
This project was done as following steps::
1. Color selection/Gray_scaling.
1. Region of interest selection.
1. Gaussian smoothing.
1. Canny Edge detection.
1. Hough Transform line detection.
1. Average/extrapolate line segments and draw them onto the image(frame).
1. Apply to video.

### Image Result
![helper_functions_used](https://github.com/LUUTHIENXUAN/Udacity-CarND-LaneLines-P1/blob/master/helper_functions_used.png)

![draw_line_improved](https://github.com/LUUTHIENXUAN/Udacity-CarND-LaneLines-P1/blob/master/draw_line_improved.png)

### Video Result
1. [Video: solidYellowLeft](https://youtu.be/VCLtHHBUPZA)
1. [Video: solidWhiteRight Video](https://youtu.be/tFdcKgX3r5c)
1. [Videochallenge Video](https://youtu.be/RUNxUG4AucU)

## Reflection

**`draw_line`** function was modified and replaced with **`slopebased_draw_lines`** as followed:<br>
Divide line segments into 2 groups, one with positive slope and another one with negative slope. 
```
if slope > 0:
   positive_slope_lines += [((x1+x2)*0.5,(y1+y2)*0.5, slope)]
else:
   negative_slope_lines += [((x1+x2)*0.5,(y1+y2)*0.5, slope)]
```
 Then take average `slope` and `x,y` value at each group and extrapolate the line segments from the bottom to the top of the region of interest by calculating top and bottom points's `x` value.<br>
```
average = [float(sum(l))/len(l) for l in zip(*slope_lines)]
```
```
top_x    = (left_apex[1] - average[1])/average[2]   + average[0]
bottom_x = (left_bottom[1] - average[1])/average[2] + average[0]
```
 Note: In case of video, the slope and position of left and right line were took average in 20 frames.
```
aver_negative_slope = average_list(20, global_aver_negative_slope, aver_negative_slope )
aver_positive_slope = average_list(20, global_aver_positive_slope, aver_positive_slope )
```
Simple sanity check also be applied to reduce unexpected values at current frames, which exceed 20% values comparing to averaged value.
```
if (abs(average_l[0]-element[0][0])/average_l[0]) > 0.2 or \
   (abs(average_l[1]-element[0][1])/average_l[1]) > 0.2 :
       l += [(average_l[0],average_l[1])]
       l = l[-l_length:]
```
Function                    |Explaination
--------                   | ---
**`slopebased_draw_lines`**| separate line segments based on its slope value into left and right group, then use `extrapolate` to each group. To make pipeline more stable, `average_list` as below was adapted.
**`extrapolate`**          |take average value at each group then extrapolate to top and bottom the region of interest. 
**`average_list`**         |take average value over frames (default 20 frames) to make the pipeline more stable. Also, do simple sanity check to filter out any values at current frame exceeded over 20% of averaged values over 20 frames.
                


## Discussion
**SHORTCOMINGS**
>   <font color='black'> The first shortcoming of this method (Hough Transform) would be relying on the environment conditions. Unclear weather, bad road surface, cars surrounding, other road markings,..etc  may result so much noise at edge detection phase. 
>   -  In **`color selection`** and **`region of interest selection`** section, parameters tuning works well at some specific cases, but it is a challenge to find a common ground of them that could work well in global working environment(for example, lines under shadow/ glaring light or shaped curve ones)

>   <font color='black'> The another shortcoming  would be straight lane line detection and noises. 
>   -  **`extrapolate`** function which is used for extending averaged line segments from bottom to the top of region of interest would fail in shaped curve. It also easy to fail with noises as mentioned above. 

**IMPROVEMENTS**
>   -  Instead of using **`grayscale`** function to take gray_scale image, using **`color_selection`** (RGB color space )with red and green threshold picked up yellow line better. 

>   -  The purpose of **`color selection`** section is to filter out irrelevant pixels of lane lines based on the color. Using different color space rather than RGB may pick up white and yellow pixel more precisely.

>  -  OpenCV has **`cv2.polyfit`** function could draw curved lines. Hough transform's  output (`y value`)may be used as input of  **`cv2.polyfit`**.

>  - **Noise** 's solution  would be taking average less few frames or reduce the height of region of interest. Furthermore, I did simple sanity check, which filter out any values exceeded over 20% of averaged values in 20 frames. 
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTczMjExNzgyOV19
-->