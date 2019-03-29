# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 19:38:56 2019

@author: Nikola Mandic
SVM classifier
"""
#%%
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.svm import SVC # SVM klasifikatore
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import numpy as np
import cv2
import os

#%%
def get_hog():
    # Racunanje HOG deskriptora za slike iz MNIST skupa podataka
    img_size = (28, 28)
    nbins = 9
    cell_size = (8, 8)
    block_size = (1, 1)
    hog = cv2.HOGDescriptor(_winSize=(img_size[1] // cell_size[1] * cell_size[1],
                                      img_size[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
    return hog
#%%
def reshape_data(input_data):
    # transformisemo u oblik pogodan za scikit-learn
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))

def invert(img):
    return 255 - img

def dilate(image):
    kernel = np.ones((2,2)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)

def erode(image):
    kernel = np.ones((2,2)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

def scale(x):
    return x/255.0

#%%
#%%
def train_classifier(hog_descriptor):
    # Treniranje klasifikatora na MNIST skupu podataka
    print("Loading MNIST dataset...")
    dataset = datasets.fetch_mldata("MNIST original")
    features = np.array(dataset.data)
    labels = np.array(dataset.target, 'int')
    
    print("Prepare data...")
    
    print(features[0])
    plt.imshow(features[0].reshape(28,28),'gray')
    
    x = []
    for feature in features:
        x.append(hog_descriptor.compute(feature.reshape(28, 28)))
    x = np.array(x, 'float32')
    x = reshape_data(x)
    
    
    x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42) 
    print('Train shape: ', x_train.shape, y_train.shape)
    print('Test shape: ', x_test.shape, y_test.shape)
    
    clf_svm = SVC(kernel='linear', probability=True) 
    clf_svm.fit(x_train, y_train)
    y_train_pred = clf_svm.predict(x_train)
    y_test_pred = clf_svm.predict(x_test)
    print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
    print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))
      
    return clf_svm 


def get_SVM():
    hog = get_hog()
    svm = train_classifier(hog)
    return svm
    
