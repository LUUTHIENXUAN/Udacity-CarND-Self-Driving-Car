"""
 IMPORTING USEFUL PACKAGES
"""
import csv
import cv2
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation, ELU
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.backend import tf as ktf
from keras.optimizers import Adam
from keras.models import Model
from keras.utils.visualize_util import plot
from keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt

"""
 LOAD THE DATA
 GENERATA MORE DATA
 1. Use the left-right images to make car recovery(get back to central line)
    left-images  : steer a little bit to the right with correction =  0.25
    right-images : steer a little bit to the left  with correction = -0.25
"""
# read log file at each row and store them in lines
lines = []
with open('./example_training_data/data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line) 

# split "lines" data and store them seperately in images, measurements
images, measurements  = [], []
for line in lines:
	
	# skip the first line(header)
	if line[0] == "center" : continue
	
	# store center, left and right images to "images"
	# center: line[0], left: line[1], right: line[2]
	for i in range(3):
		# get filename from csv 
		filename = line[i].split('/')[-1]
		
		# debug failed loading image
		# Note: cv2 read image in BGR format
		# https://www.pyimagesearch.com/2014/11/03/display-matplotlib-rgb-image/
		image = cv2.imread('./example_training_data/data/IMG/'+ filename)
		if image is None:
			print("Incorrect path", direction_path)
		images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	
	# store steering angles
	correction = 0.25
	steering   = float(line[3])
	measurements.append(steering)
	measurements.append(steering + correction)
	measurements.append(steering - correction)
	
print (len(images), len(measurements))

"""
 GENERATA MORE DATA
 1. Flip the images
"""
# Augment data_set before feeding to karas
augmented_images        = []
augmented_measurements  = []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	
	flipped_image        = cv2.flip(image, 1)
	flipped_measurement  = measurement*-1.0
	augmented_images.append(flipped_image)
	augmented_measurements.append(flipped_measurement)

# convert images to array before feeding to karas	
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

print ('[Expected X_train: (n, 160, 320, 3)/ y_train:(n,)]: ', X_train.shape, y_train.shape)
 
"""
 BUILD & TRAIN THE MODEL
 Note: better training performance with these steps below in the GPUs 
 rather than in the CPU
 
 1. Crop out the images (help the model run faster)
    70 pixels of the top & 0 pixels of the bottom
    0 pixels of left and right images
    Resize the image (help the model run faster)
 2. Normolize the features
 2. Use Lenet/Nvidia model
    Note: 
    1. Original model's loss after epoch 1 was small enough to indicate not overfitting.
    2. However, Dropout layer was added to reduce overfitting.
       Result: 
       2.1 Car drived safely on track 1 at first circle only with Dropout after model.add(Dense(100))
       2.2 Car drived safely on track 1 with more Dropout after model.add(Dense(50))
       2.3 Car drving failed with more Dropout after model.add(Dense(10))
       
 3. Complie, Fit and Save the model
"""

def resize_img(input):
    from keras.backend import tf as ktf
    return ktf.image.resize_images(input, 64, 128)

def train_model(model_chosen = 'Nvidia'):
	
	model      = Sequential()
	img_sample = X_train[0:1,:,:,:]
	
	"""
	# normolize the features inside Keras
	model.add(Lambda(lambda x: x /255.0 -0.5, input_shape=(160,320,3)))
	normalized_img_sample = model.predict(img_sample, batch_size=128)
	# crop the images inside Keras
	model.add(Cropping2D(cropping=((70, 0), (0, 0))))
	cropped_img_sample = model.predict(img_sample, batch_size=128)
	# resize the images inside Keras
	#  Keras 1.2 has some issues when using the syntax lambda x: function_name(x).
	#  model.add(Lambda(lambda x: resize_function(x)))
	model.add(Lambda(resize_img))
	resized_img_sample = model.predict(img_sample, batch_size=128)
	"""
	
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
	
	f, ax = plt.subplots(4, sharex=True, figsize=(7, 10))
	ax[0].imshow(img_sample[0])
	ax[0].set_title('Sample Image')
	ax[1].imshow(cropped_img_sample[0].astype(np.uint8))
	ax[1].set_title('Cropped Image')
	ax[2].imshow(resized_img_sample[0].astype(np.uint8))
	ax[2].set_title('Resized Image')
	ax[3].imshow(normalized_img_sample[0,:,:,0], cmap = 'gray')
	ax[3].set_title('Normalized image')
	plt.savefig('Preprocessing_sample_image.png')
	plt.show()
	
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
	
	# Initialize the optimizer with a learning rate = 0.0001 and complie it
	optimizer = Adam(lr = 0.0001)
	model.compile(optimizer = optimizer, loss='mse')
	
	# save the model from the best epoch
	checkpoint = ModelCheckpoint(filepath = 'model.h5', verbose = 1, save_best_only=True, monitor='val_loss')
	# stop the training after the model stops improving over a specified delta
	callback   = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
	
	# Train the model
	history_object = model.fit(X_train, y_train,\
	                           validation_split= 0.2, shuffle=True,\
	                           nb_epoch = 10, batch_size = 128,\
	                           callbacks = [checkpoint, callback])
	
	"""
	Visulize the model, plot model into a graph
	  Install extra packages below: 
	  pip install graphviz
	  pip install pydot
	  pip install pydot-ng (In case :module 'pydot' has no attribute 'find_graphviz')
	"""
	plot(model, to_file="model.png")
	
	
	"""
	Visulize loss
	"""
	### print the keys contained in the history object
	print(history_object.history.keys())
	
	### plot the training and validation loss for each epoch
	plt.plot(history_object.history['loss'])
	plt.plot(history_object.history['val_loss'])
	plt.title('model mean squared error loss')
	plt.ylabel('mean squared error loss')
	plt.xlabel('epoch')
	plt.legend(['training set', 'validation set'], loc='upper right')
	plt.savefig('loss_visulization.png')
	plt.show()
	
	return model

train_model(model_chosen = 'Nvidia')

"""
 VISUALIZING LOSS

history_object = model.fit_generator(train_generator,\
                                     samples_per_epoch = len(train_samples),\
                                     validation_data   = validation_generator,\
                                     nb_val_samples    = len(validation_samples),\
                                     nb_epoch          = 5,\
                                     verbose           = 1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
"""
