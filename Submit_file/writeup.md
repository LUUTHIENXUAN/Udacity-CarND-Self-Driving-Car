# **Project 3: Behavioral Cloning** 


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

| File         | Details                                             |
|--------------|-----------------------------------------------------|
| `model.py`   | containing the script to create and train the model |
| `drive.py`   | for driving the car in autonomous mode              |
| `model.h5`   | containing a trained convolution neural network     |
| `writeup.md` | summarizing the results                             |

|File            |Details  |
|--              |--       |
|`model.py`      |containing the script to create and train the model  |
|`drive.py`      |for driving the car in autonomous model  |
|`model.h5`      |containing a trained convolution neural network  |
|`writeup.md`    |summarizing the results  |

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
My model based on Nvidia model with some modifications such as adding `dropout`, `activation('relu')` after `Dense` layer and the data was cropped, resized and finally normalized in the model using a Keras lambda layer. 
```sh
        # crop the images inside Keras and get one sample image
	model.add(Cropping2D(cropping=((70, 0), (0, 0)), input_shape=(160, 320, 3)))
	cropped_img_sample = model.predict(img_sample, batch_size=128)
	
	# resize the images inside Keras and get one sample image
	#  Keras 1.2 has some issues when using the syntax lambda x: function_name(x).
	#  model.add(Lambda(lambda x: resize_function(x)))
	model.add(Lambda(resize_img))
	resized_img_sample = model.predict(img_sample, batch_size=128)
	
	# normolize the features inside Keras and get one sample image
	model.add(Lambda(lambda x: x /255.0 -0.5))
	normalized_img_sample = model.predict(img_sample, batch_size=128)
```


#### 2. Attempts to reduce over-fitting in the model

The model(Nvidia model) contains dropout layers in order to reduce over-fitting. 
```sh
                model.add(Dense(100))
		model.add(Dropout(.5))
		model.add(Activation('relu'))
		
		model.add(Dense(50))
		model.add(Dropout(.5))
		model.add(Activation('relu'))
		
		model.add(Dense(10))
		model.add(Activation('relu'))
		
		model.add(Dense(1))
```
The model was trained and validated on different data sets to ensure that the model was not over_fitting by splitting 20% from data set. 
```sh
	history_object = model.fit(X_train, y_train,\
	                           validation_split= 0.2, shuffle=True,\
	                           nb_epoch = 10, batch_size = 128,\
	                           callbacks = [checkpoint, callback])
```
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, with a learning rate = 0.0001.
```sh
optimizer = Adam(lr = 0.0001)
model.compile(optimizer = optimizer, loss='mse')
```

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a data set  alone which is provided by Udacity, including `center`, `left` and `right` images captured by camera and also the the `steering` angles.  `throttle`, `brake`, and `speed` were not applicable.   

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use Lenet model. 
```sh
     if model_chosen == 'Lenet':
		# Lenet model
		model.add(Convolution2D(6,5,5,activation ='relu'))
		model.add(MaxPooling2D())
		model.add(Convolution2D(16,5,5,activation ='relu'))
		model.add(MaxPooling2D())
		model.add(Flatten())
		model.add(Dense(120))
		model.add(Dense(84))
		model.add(Dense(1))
```
I thought this model might be appropriate because it was used in Traffic sign classification. But It did not end well. So I switched to Nvidia model which was trained for real car to drive autonomously. 
```sh
elif model_chosen == 'Nvidia':	
		# Nvidia model
		model.add(Convolution2D(24,5,5,subsample = (2,2), activation ='relu'))
		model.add(Convolution2D(36,5,5,subsample = (2,2), activation ='relu'))
		model.add(Convolution2D(48,5,5,subsample = (2,2), activation ='relu'))
		model.add(Convolution2D(64,3,3,activation ='relu'))
		model.add(Convolution2D(64,3,3,activation ='relu'))
		model.add(Flatten())
		
		model.add(Dense(100))
		model.add(Dropout(.5))
		model.add(Activation('relu'))
		
		model.add(Dense(50))
		model.add(Dropout(.5))
		model.add(Activation('relu'))
		
		model.add(Dense(10))
		model.add(Activation('relu'))
		
		model.add(Dense(1))
		# Print a summary of a Keras model details 
		# such as the number of layers and size of each layer
		model.summary()
```
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The validation loss was so small that indicate the model overfitting but I added 2 dropout layers just in case.
Result: 
- Car drived safely on track 1 at first lap only with Dropout after `model.add(Dense(100))`
- Car drived safely on track 1 with more Dropout after `model.add(Dense(50))`
- Car failed with more Dropout after `model.add(Dense(10))`

Here is visulization of validation loss and training loss
![Loss Visulization](https://github.com/LUUTHIENXUAN/Udacity-CarND-Behavioral-Cloning-P3-/blob/master/loss_visulization.png)

Then I use `ModelCheckpoint` to save the model from the best epoch, and  `EarlyStopping` to stop the training after the model stops improving over a specified delta as below:

```sh
    # save the model from the best epoch
	checkpoint = ModelCheckpoint(filepath = 'model.h5', verbose = 1,\
	                             save_best_only=True, monitor='val_loss')
	# stop the training after the model stops improving over a specified delta
	callback   = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
```

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle may be fallen off the track at some heavy curves. To improve the driving behavior in these cases, I guess more augmented data should be generated, for example: shifting the images including adjusting steering angles.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture details such as the number of layers and size of each layer was summarized as following:
```sh
model.summary()
```
| Layer (type)                    | Output Shape                    | Param # | Connected to             |
|---------------------------------|---------------------------------|---------|--------------------------|
| cropping2d_1 (Cropping2D)       | (None, 90, 320, 3)              | 0       | cropping2d_input_1[0][0] |
| lambda_1 (Lambda)               | (None, 64, 128, 3)              | 0       |                          |
| lambda_2 (Lambda)               | (None, 64, 128, 3)              | 0       |                          |
| convolution2d_1 (Convolution2D) | convolution2d_1 (Convolution2D) | 1824    |                          |
| convolution2d_2 (Convolution2D) | (None, 13, 29, 36)              | 21636   |                          |
| convolution2d_3 (Convolution2D) | (None, 5, 13, 48)               | 43248   |                          |
| convolution2d_4 (Convolution2D) | (None, 3, 11, 64)               | 27712   |                          |
| convolution2d_5 (Convolution2D) | (None, 1, 9, 64)                | 36928   |                          |
| flatten_1 (Flatten)             | (None, 576)                     | 0       |                          |
| dense_1 (Dense)                 | (None, 100)                     | 57700   |                          |
| dropout_1 (Dropout)             | (None, 100)                     | 0       |                          |
| activation_1 (Activation)       | (None, 100)                     | 0       |                          |
| dense_2 (Dense)                 | (None, 50)                      | 5050    |                          |
| dropout_2 (Dropout)             | (None, 50)                      | 0       |                          |
| activation_2 (Activation)       | (None, 50)                      | 0       |                          |
| dense_3 (Dense)                 | (None, 10)                      | 510     |                          |
| activation_3 (Activation)       | (None, 10)                      | 0       |                          |
| dense_4 (Dense)                 | (None, 1)                       | 11      |                          |

Here is a visualization of the architecture printed by:
```sh
plot(model, to_file="model.png"):
```
![model at a glance](https://github.com/LUUTHIENXUAN/Udacity-CarND-Behavioral-Cloning-P3-/blob/master/model.png)

#### 3. Creation of the Training Set & Training Process

I used data set provide by Udacity and did not record any data at all.

To augment the data sat, I also flipped images and angles thinking that this would make the car drive smoothly.
After the collection process, I had 48216 number of data points. I then preprocessed this data by cropping, resizing and finally normalizing the whole data points.
Here is an example of pre-processed image:
![enter image description here](https://github.com/LUUTHIENXUAN/Udacity-CarND-Behavioral-Cloning-P3-/blob/master/Preprocessing_sample_image.png)

My Keras raised weird error when I tried to cut off the bottom of the image so that I kept the bottom cut as 0.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. I should use fit_generator instead of generator because in many cases neural network needs a large dataset for training and it might not be possible to read all the dataset into memory. Due to time constraints, I will come back later to adapt this method. 
