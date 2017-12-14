# **Project 2: Build a Traffic Sign Recognition Classifier** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

 1. Load the data set (see below for links to the project data set)
 2. Explore, summarize and visualize the data set
 3.  Design, train and test a model architecture
 4.  Use the model to make predictions on new images
 5.  Analyze the softmax probabilities of the new images
 6.  Summarize the results with a written report

---
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. 

Here is the statistics of the traffic signs data set:

| Statistics of the traffic signs data set |     Size    |
|:----------------------------------------:|:-----------:|
| Number of training examples              | 34799       |
| Number of validation examples            | 4410        |
| Number of testing examples               | 12630       |
| Image data shape                         | (32, 32, 3) |
| Number of classes                        | 43          |
#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributed at a glance
![Dataset](https://github.com/LUUTHIENXUAN/Udacity-CarND-Traffic-Sign-Classifier-P2/blob/master/Distribution_Labels_before.png)

And some random images
![Random Image](https://github.com/LUUTHIENXUAN/Udacity-CarND-Traffic-Sign-Classifier-P2/blob/master/random_image.png)
### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

 1. Generate fake data/ add more data.
 Some classes have less than 300 images as above, which may cause poor classification than other images. So image augmentation is the way to go to makes those classes have larger numbers of images. The original data set would be made by adding its transformed versions.  
 More diverse training set deformations were also investigated in such as brightness, contrast, shear and blur perturbations to address the numerous real-world deformations.
  > `def transform(image)`
  > Use OpenCv to change brightness, contrast and blur the image  
  > Use OpenCV's transformation functions `cv2.warpAffine` to randomly generate datas
  >     1. position ([-2,2] pixels), AND
  >     2. scale ([.9,1.1] ratio), AND
  >     3. rotation([-15,+15] degrees)These versions are randomly
     
 	|  Training Set | Training Set after Argument |
	|:-------------:|:---------------------------:|
	| 34799 samples |               51690 samples |
	
	Here is an exploratory visualization of the data set after argument. It is a bar chart showing how the data distributed at a glance
![Dataset after argument](https://github.com/LUUTHIENXUAN/Udacity-CarND-Traffic-Sign-Classifier-P2/blob/master/Distribution_Labels_after.png)
 2. Pre-process the Data Set
As a first step, I decided to convert the images to `YUV` and use channel `Y` as the grayscale
because it showed image clearly than converting image to gray at some tricky conditions.
Then I use `cv2.equalizeHist` `cv2.createCLAHE` to emphasize edges.

 3. Normalize the Data Set `def normaize(data)`
  I normalized the image data so that the data has mean zero and equal variance

Here is an example of an original image and an preprocessed image:
![Dataset after argument](https://github.com/LUUTHIENXUAN/Udacity-CarND-Traffic-Sign-Classifier-P2/blob/master/transform_iamge.png)


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:--------------------- |:--------------------------------------------- | 
| <i>Input</i>         	| 32x32x1 Y channel image   					| 
| <i>Convolution 5x5</i>| 1x1 stride, same padding, outputs 32x32x64 	|
| <i>RELU</i>			|												|
| <i>Max pooling</i>	| 2x2 stride,  outputs 16x16x64 				|
| <i>Convolution 5x5</i>| 1x1 stride, same padding, outputs 10x10x16   	|
| <i>RELU</i>			|												|
| <i>Max pooling</i>	| 2x2 stride,  outputs 5x5x16.  				|
| <i>Flatten</i>	    | 												|
| <i>Fully connected</i>| Input = 400. Output = 120.					|
| <i>RELU</i>			|												|
| <i>Dropout</i>		|												|
| <i>Fully connected</i>| Input = 120. Output = 84.	     				|
| <i>RELU</i>			|												|
| <i>Dropout</i>		|												|
| <i>Fully connected</i>| Input = 84. Output = 43.      				|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an parameters as below:
| Parameter         	|     Value	        					|
|:--------------------- |-------------------------------------: |
| EPOCHS                |  40                               	|  
| BATCH_SIZE            |  128                               	| 
| rate                  |  0.0009                               |
| mu                    |  0		                            |
| sigma                 |  0.1                                  |

  
#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
| Data set          	|     Accuracy        					|
|:--------------------- |-------------------------------------: |
| training set          |  0.939                               	|  
| validation set        |  0.968                               	| 
| test set              |  0.939	                            |

**Note**
> 1. Pre-process in step 2 alone, `Validation Accuracy` can reach up to 0.93.
> 2. Data augmentation was implemented to reduce overfitting on models. `Validation Accuracy` can reach at 0.94
> 3. `BATCH_SIZE`'s increment from 128 to 256, 512 results in reduction of Validation Accuracy. Keep BATCH_SIZE= 128 as default.
> 4. Reduce learning `rate`, for example from 0.001 to 0.0001 makes `Validation Accuracy` worsen, while trying to run more EPOCH is not
> going to help. Feels like 20~40 EPOCH is good enough.    `rate` =
> 0.0009, and `EPOCH` = 30, `Validation Accuracy` = 0.946.
> 5. Try different regulation as dropout. Apply dropout at fully connected layer can help improve the accuracy of neural network.
> Validation Accuracy could reach up to 0.968~0.971 . 
> **Note**: Add  dropout after  max_pool decrease Validation Accuracy.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 10 German traffic signs that I found on the web:
![Images from Internet](https://github.com/LUUTHIENXUAN/Udacity-CarND-Traffic-Sign-Classifier-P2/blob/master/Internet_images.png)
The 8th image might be difficult to classify because it slightly different with images in training set.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:
![Top 5 prediction](https://github.com/LUUTHIENXUAN/Udacity-CarND-Traffic-Sign-Classifier-P2/blob/master/top5guess.png)

The model was able to correctly guess 8 of the 10 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 0.939

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

![Top 5 softmax probabilities as bar charts](https://github.com/LUUTHIENXUAN/Udacity-CarND-Traffic-Sign-Classifier-P2/blob/master/top5guess_bar_new.png) 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


