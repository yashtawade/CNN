#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 08:42:32 2017

@author: yashtawade
"""

#
#Importing necessary packages and libraries
from keras.models import Sequential #Initializes the NN
from keras.layers import Convolution2D #Convulution Step - Add convolutional layers
from keras.layers import MaxPooling2D #Pooling Step - Add pooling layers
from keras.layers import Flatten #Flattening Step - Pooled feature maps to large feature vectors
from keras.layers import Dense 

#Initializing the CNN
classifier = Sequential() #Creating an object 'classifier' of the Sequential class

#Convolution layer with feature detector set as (32,3,3) and input_shape=(64, 64, 3) with the rectifier activation function
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

#Pooling done using a pool size of 2X2
classifier.add(MaxPooling2D(pool_size=(2,2)))

#2nd convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#3rd convolutional layer with 64 feature detectors
classifier.add(Convolution2D(64, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


#Flattening done using the Flatten method which 'flattens' the pooled feature map into a one dimensional vector
classifier.add(Flatten())

#Adding the Hidden Layer i.e Fully connected layer 
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid')) #Sigmoid as we have binary outcome - Cat or Dog

#Compiling the CNN choosing the loss function as binary cross entropy 
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'path of the file',#path in the directory
        target_size=(64, 64), #resizing the images
        batch_size=32, #batches of 32 images
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'Path of the file',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000, #Images in the training set
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)  #Images in the test set

        
#Making a single prediction
import numpy as np   
from keras.preprocessing import image #Imported in order to load the test image -> my_test_image
my_test_image = image.load_img('Path of the file',target_size=(64, 64))
my_test_image = image.img_to_array(my_test_image) #Converts 64X64 image into 64X64X3 Array
my_test_image = np.exapnd_dims(my_test_image,axis = 0)#
classifier.predict(my_test_image)
training_set.class_indices
