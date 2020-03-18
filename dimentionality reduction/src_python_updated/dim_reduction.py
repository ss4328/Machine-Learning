#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 22:46:05 2020

@author: shivanshsuhane
"""
import os
from matplotlib.image import imread
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def projectPoints(d, v1, v2):

    eigenData = [];
    for observation in d:
        projectedData = [ np.dot(observation, v1), np.dot(observation, v2) ]
        eigenData.append(projectedData)

    return eigenData

print("Dimentionality reduction: ")
path = os.getcwd() + '/yalefaces'
print("Current path: " + os.getcwd())

imgDirs = []
images = []
imData = []


# Step 1: Read in the list of files
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        # if '.txt' in file:
        imgDirs.append(os.path.join(r, file))


# Read in all the images to images array
i=0
for imgDir in imgDirs:
    im = Image.open(imgDir)     #read image
    im = im.resize((40,40))     #downsize to 40x40
    im = im.resize((1600,1))    #flatten to 1x1600
    images.append(im)
    imageData = np.array(im)
    imData.append(imageData)
    i+=1
    
# Now data has 154 cols of 1x1600 images    
data = np.concatenate(imData, axis=0)

# Standardize data
data = (data - np.mean(data,axis=0)) / np.std(data,axis=0)  

# Calculate covariance matrix
covMat = np.cov(data)
eigen_vals, eigen_vecs = np.linalg.eig(covMat)

# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

#finding W matrix
matrix_w = np.hstack((eigen_pairs[0][1].reshape(154,1), eigen_pairs[1][1].reshape(154,1)))
print('Matrix W:\n', matrix_w)

#finding transformed matrix; projection calculation
transformed = matrix_w.T.dot(data)

#plotting the 2-d data
plt.scatter(transformed[0], transformed[1])
plt.gca().grid(True, linewidth=0.7, linestyle=':')
plt.axis([-10, 10, -10, 10])
plt.show();