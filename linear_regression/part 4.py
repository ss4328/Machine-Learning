# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 22:57:03 2020

@author: shivanshsuhane
"""

import numpy as np
import pandas as pd

#abstract Lin reg, and RMSE functionalities and create another fn for S-fold validation

def RMSE(pred_y, actual_y):
    RMSE = np.sqrt((np.mean(np.square(actual_y-pred_y),axis=0)))
    return RMSE

#accepts X, y and returns the weights for computing a linear regression
def linReg(trainMat, yTrain):
    trainMat = np.append(arr = np.ones([trainMat.shape[0],1]).astype(int), values = trainMat,axis=1)  #concatenate the ones to train matrix
    Xt = trainMat.transpose()
    XtX = np.dot(Xt,trainMat)
    theta = np.dot(np.dot(np.linalg.inv(XtX), Xt), yTrain)
    return theta

#accepts a test matrix and weights from linReg, returns an numpy array of predicted values 
def getPredictedValues(weights, X):
    X = np.append(arr = np.ones([X.shape[0],1]).astype(int), values = X,axis=1)  #concatenate the ones to train matrix
    pred_outputs = np.matmul(X, weights)
    return pred_outputs

def squaredErrors(pred_y, actual_y):
    sqErr = np.square(actual_y-pred_y)
    return sqErr

if __name__ == "__main__":
    #read in data
    df = pd.read_csv("x06simple.csv", index_col = "Index")
    df = df[['Temp of Water','Length of Fish','Age']]
    
    #randomize the data
    data = df.to_numpy()
    
    
    s=5
    RMSEArr = np.empty([20])
    
    FoldErrs = np.empty([s])
    
    
    for ErrIndex in range(0,20):
        np.random.shuffle(data)
        folds = np.empty([s])
        
        trainCount = int(len(data)*(s-1)/s)
        foldErrs = np.empty([s])
        for sIndex in range(0,s):
            #print(sIndex)
            
            trainMat, testMat = data[:trainCount, :],data[trainCount:,:]
            
            yTrain = trainMat[:,-1]     #get values for train y
            yTest = testMat[:,-1]     #get values for test y
            
            testMat = np.delete(testMat,-1,1)   #delete y values from test X
            trainMat = np.delete(trainMat,-1,1) #delete y values from train X
            
            weights = linReg(trainMat, yTrain)
            pred_outputs = getPredictedValues(weights, testMat)
            
            sampleSqErrors = squaredErrors(pred_outputs,yTest)     #collection of sq. Difference of sample
            FoldErrs[sIndex] = np.mean(sampleSqErrors, axis=0)   #s error values
            
        
        
        RMSEIter = np.sqrt(np.mean(foldErrs,axis=0)) #RMSE for iter
        RMSEArr[ErrIndex] = RMSEIter #20 values
        
    print(RMSEArr)    
    print(np.mean(RMSEArr))
    print(np.std(RMSEArr))
        
        
            