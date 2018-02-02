## Project 5: Behavioral Cloning

**Vehicle Detection Project**

The goals / steps of this project are the following:

1.  Perform a `HOG` (Histogram of Oriented Gradients) feature extraction on a labeled training set of images and train a classifier `Linear SVM` classifier. 
	+ Set up `C`, `loss` parameters and conduct Grid Search To Find Parameters Producing Highest Score.
	+ Train a new classifier using the best parameters found by the grid search.
	+ Check the score of this classifier
	**Note**: Other classifiers as `LRC(Logistic Regression Classifier)`, `MLPC (Multi-Layer Perceptron Classifier)`, `GBC (Gradient Boosting Classifier)`, `BC (Bagging Classifier)`,  `GauNB (Gaussian Naive Bayes)`, `SGDC`, `NuSVC` were also trained to adapt the classifier with best performance.  
2. Apply a color transform and append binned color features, but not histograms of color, to HOG feature vector. 
	```python
	spatial_feat   = True   # Spatial features on or off
	hist_feat      = False  # Histogram features on or off
	hog_feat       = True   # HOG features on or off
	```

3. Normalize your features and randomize a selection for training and testing.
4. Implement a sliding-window technique and use trained classifier to search for vehicles in images.
5. Run pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
6. Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric Points](https://review.udacity.com/#!/rubrics/513/view) 
### Writeup / README

### Histogram of Oriented Gradients (HOG)

1. ####  Explain how (and identify where in your code) you extracted HOG features from the training images.
	I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:
	```python
	cars    = glob.glob('Dataset/vehicles/vehicles/*/*.png')
	notcars = glob.glob('Dataset/non-vehicles/non-vehicles/*/*.png')
	```
	![alt text][image1]
   I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.
Here is an example using the in gray and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:
![alt text][image2]

2. #### Explain how you settled on your final choice of HOG parameters.
	I tried various combinations of parameters such as `color_space` , `orient`,  `spatial_feat`, `hist_feat` and found out that `Lab` or `YUV` of color_space performed well in term of reducing false-positive. 
	Increasing `orient` value dramatically reduces false-positive but badly affect the pipeline performance. `hog_channel` in `ALL` mode is better under various weather conditions. Stop using `hist_feat` contributes in term of performance but still acceptable in accuracy. Here is the final choice of HOG parameters. 

	```python
	color_space    = 'Lab'  # Can be RGB, HSV, LUV, HLS, *YUV, *YCrCb, *Lab
	orient         = 11     # HOG orientations
	pix_per_cell   = 16     # HOG pixels per cell
	cell_per_block = 2      # HOG cells per block
	hog_channel    = "ALL"  # Can be 0, 1, 2, or "ALL"
	spatial_size   = (32,32)# Spatial binning dimensions
	hist_bins      = 32     # Number of histogram bins
	spatial_feat   = True   # Spatial features on or off
	hist_feat      = False  # Histogram features on or off
	hog_feat       = True   # HOG features on or off
	```

3. #### Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).
	I trained various classifiers using 11 orientations, 16 pixels per cell and 2 cells per block. Data set was split up by 20% into randomized training and test sets. Feature vector length was 4260.
	```python 
	# Split up data into randomized training and test sets
	rand_state = np.random.randint(0, 100)
	X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)
    ```
    ```python
    # Train classifer function
    def training_classifier(classifier = "SVC"):
	    if classifier == "SVC":
	        # Support Vector Classifier
	        train_class = LinearSVC()
	    elif classifier == "LRC":
	        # Logistic Regression Classifier
	        train_class = LogisticRegression()
	    elif classifier == "MLPC":
	        # Multi-Layer Perceptron Classifier
	        train_class = MLPClassifier(solver='sgd',\
	                                    learning_rate_init=0.01,\
	                                    max_iter=10000)
	    elif classifier == "GBC":
	        # Gradient Boosting Classifier
	        train_class = ensemble.GradientBoostingClassifier(random_state= 42)
	    elif classifier == "BC":
	        # Bagging Classifier
	        train_class = ensemble.BaggingClassifier(random_state= 42)
	    elif classifier == "GauNB":
	        # Gaussian Naive Bayes
	        train_class = naive_bayes.GaussianNB()
	    elif classifier == "SGDC":
	        # SGD Classifier
	        train_class = linear_model.SGDClassifier(random_state= 42)
	    elif classifier == "NuSVC":
	        # NuSVC
	        train_class = svm.NuSVC(random_state= 42)    
    ...
    ```
 


### Sliding Window Search

1. #### Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
	I decided to search random window positions at 2 scales as follow:
	
	| Parameters   | Window 1    | Window 2    |
	|--------------|-------------|-------------|
	| xy_window    | (96,96)     | (160,160)   |
	| y_start_stop | [350,550]   | [450,700]   |
	| xy_overlap   | (0.75,0.75) | (0.55,0.55) |
	
	![alt text][image3]

2. #### Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?
	Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color, which provided a nice result.  Here are some example images:
	![alt text][image4]

---

### Video Implementation

1. #### Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
	Here's a [link to my video result](./project_video.mp4)


2. #### Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.
	I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  
	Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

1. #### Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
	To improve the accuracy for `Linear SVM` I conducted Grid Search To find parameters producing highest score. Test Accuracy was improved from  **0.9885** to **0.9907**. 
	I also tried other classifier as described from the top of this markdown. Some of them perform faster by less accuracy then `Linear SVM`. `MLPC` seems to best choice for accuracy. Here is the best 3 when considering trade-off between speed and accuracy.
	1. SVC

	> 49.44394397735596 Seconds to train this classifier... 		
	> Train Accuracy of this classifier =  1.0 		
	> Test  Accuracy of this classifier =   0.9885
	> 	0.21602153778076172 Seconds to predict

		
	> Best Parameters：{'loss': 'squared_hinge', 'C': 0.001}
	> Train Accuracy of this classifier =  0.9997
	>  Test  Accuracy of this classifier =  0.9907
	2. LRC
	
	> 30.371036767959595 Seconds to train this classifier... 
	> Train Accuracy of this classifier =  1.0 
	> Test  Accuracy of this classifier =  0.9907
	> 0.05200505256652832 Seconds to predict

	3. MLPC 
	
	> 39.426942348480225 Seconds to train this classifier... 
	> Train Accuracy of this classifier =  1.0 
	> Test  Accuracy of this classifier =  0.9961
	> 0.05400538444519043 Seconds to predict

	To improve the performance of pipline. OpenCV Hog was adapted instead of Scikit. Here is the function to switch between OpenCV of Scikit:
	```python
	# To get hog features, use scikit's hog or OpenCV's HOGDescriptor
	# Define a function to return HOG features and visualization
	def get_hog_features(img, orient, pix_per_cell, cell_per_block,\
                         vis=False, feature_vec=True, method = "opencv"):
    
	    if method == "scikit":
	        # Call with two outputs if vis==True
	        if vis == True:
	            features, hog_image = hog(img,orientations=orient, 
                                          pixels_per_cell=(pix_per_cell, 
                                                           pix_per_cell),
                                          cells_per_block=(cell_per_block,
                                                           cell_per_block), 
                                          transform_sqrt=True, 
                                          visualise=vis,
                                          feature_vector=feature_vec)
	            return features, hog_image
	        # Otherwise call with one output
	        else:
	            features = hog(img, orientations=orient, 
                               pixels_per_cell=(pix_per_cell,
                                                pix_per_cell),
                               cells_per_block=(cell_per_block,
                                                cell_per_block), 
                               transform_sqrt=True, 
                               visualise=vis, feature_vector=feature_vec)
	            return features
    
	    if method == "opencv":
	        cell_size    = (pix_per_cell, pix_per_cell)
	        block_size   = (cell_per_block, cell_per_block) 
	        nbins        = orient
	        #https://discussions.udacity.com/t/ways-to-improve-processing-time/237941/13
	        hogCV = cv2.HOGDescriptor(_winSize     =(img.shape[1] // cell_size[1] * cell_size[1],img.shape[0] // cell_size[0] * cell_size[0]),
                                      _blockSize   =(block_size[1] * cell_size[1],block_size[0] * cell_size[0]),
                                      _blockStride =(cell_size[1], cell_size[0]),
                                      _cellSize    =(cell_size[1], cell_size[0]),
                                      _nbins       = nbins)

	        return np.ravel(hogCV.compute(img))
    ```
