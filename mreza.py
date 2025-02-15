# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 19:28:01 2019

@author: Nikola Mandic
model neuronske mreze za racunanje
"""
#%% 
import numpy as np
import cv2
import os
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
from keras.models import save_model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D,Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K


import tensorflow as tf

#%%
def cnn_model(x_train,x_test,y_train,y_test):
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print('Number of images in x_train', x_train.shape[0])
    print('Number of images in x_test', x_test.shape[0])
    
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))
    
    model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
    model.fit(x=x_train,y=y_train, epochs=10)
    model.evaluate(x_test, y_test)
    
    model.save('modelConvo.h5')
    
    return model

#%%
def loadModel():
    model = None
    try:
        model = load_model('modelConvo.h5')
        if model is None:
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            model = cnn_model(x_train,x_test,y_train,y_test)
            return model
    except NameError:
        print('Cant find model')
    
    return model  


