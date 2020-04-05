#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 22:57:03 2020

@author: shivanshsuhane
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#part 2
x_arr = []
maxIters = 1000
yMat = []
x1,x2,x =0,0,0
er = 0.1
deltaMin = 2**(-23)

#loop do do linear descend
for i in range(maxIters):
    grad= 2 * (x + x - 2)
    x_old = x
    x_old = x
    x = x_old -(er * grad)
    x_arr.append(x)
    yMat.append((x + x - 2)**2)
    delta = abs(x_old-x)        #if our delta achieved is smaller than the given delta, we break the loop 
    if delta <deltaMin:
        break

#plots code
plt.plot(yMat)
plt.title('Iter vs j')
plt.xlabel('Iter')
plt.ylabel('j')

fig2 = plt.figure()
plt.plot(x_arr)
plt.show()

plt.title('Iter vs x1')
plt.plot(x1)
plt.xlabel('Iter')
plt.ylabel('x1')
plt.show()

#part 3

#read in data
df = pd.read_csv("x06simple.csv", index_col = "Index")

df = df[['Temp of Water','Length of Fish','Age']]
#randomize the data
data = df.to_numpy()
np.random.shuffle(data)
cols = list(df)
cols[0], cols[2] = cols[2], cols[0]
print(df[:5])



#split to test and train datasets
#train=data.sample(frac=0.66,random_state=0) #random state is a seed value
trainCount = int(len(data)*2/3)

trainMat, testMat = data[:trainCount, :],data[trainCount:,:]    #https://stackoverflow.com/questions/3674409/how-to-split-partition-a-dataset-into-training-and-test-datasets-for-e-g-cros

#testMat = (testMat - np.mean(testMat,axis=0)) / np.std(testMat,axis=0)      #standardize test data
#trainMat = (trainMat - np.mean(trainMat,axis=0)) / np.std(trainMat,axis=0)  #standardize train data
yTrain = trainMat[:,-1]     #get values for train y
yTest = testMat[:,-1]     #get values for test y

testMat = np.delete(testMat,-1,1)   #delete y values from test X
trainMat = np.delete(trainMat,-1,1) #delete y values from train X

#yTrain1 = (yTrain - np.mean(yTrain,axis=0)) / np.std(yTrain, axis = 0)

print(len(df))
print(len(testMat))
print(len(trainMat))

#fn to compute the closed-form linear regression
#fn basically tries to deduce underlying parameter weights in a equation
#the weights which give the best RMSE are correct
    #add bias feature
    #now we want to minimize the suqared error over all observations


trainMat = np.append(arr = np.ones([trainMat.shape[0],1]).astype(int), values = trainMat,axis=1)  #concatenate the ones to train matrix
Xt = trainMat.transpose()
XtX = np.dot(Xt,trainMat)
theta = np.dot(np.dot(np.linalg.inv(XtX), Xt), yTrain)

print(theta)
print(trainMat)

testMat = np.append(arr = np.ones([testMat.shape[0],1]).astype(int), values = testMat,axis=1)  #concatenate the ones to train matrix
pred_outputs = np.matmul(testMat, theta)

MSE = np.mean(np.square(yTest-pred_outputs),axis=0)
print(MSE)
RMSE = np.sqrt(MSE)

print(RMSE)
#ans=linReg(trainMat, yTrainMat)
#Apply the soln to testing samples

#calculate the RMSE for the observation


