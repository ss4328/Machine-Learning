#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 01:21:28 2020

@author: shivanshsuhane
"""

import numpy as np
import pandas as pd
import math


def log_likelihood(features, target, weights):
    features = np.append(arr = np.ones([features.shape[0],1]).astype(int), values = features,axis=1)  #concatenate the ones to train matrix
    scores = np.dot(features, weights)
    var1 = target*scores - np.log(1 + np.exp(scores))
    ll = np.sum( var1 )
    return ll


def printRunStatistics(preds, labels):
    print("Run Statistics: ")
    
    TP=0
    FP=0
    FN=0
    TN=0
    accu=0
    
    
    for i in range (len(labels)):
        label = labels[i]
        pred = preds[i]
        if label==1 and label==pred:
            TP+=1
            accu+=1
        elif label==0 and label==pred:
            TN+=1
            accu+=1
        elif label==1 and label!=pred:
            FN+=1
        elif label==0 and label!=pred:
            FP+=1
    
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    FVal = 2*precision*recall/(precision+recall)
    accu=accu/np.size(preds,0)
    print("Accuracy :"+ str(accu))
    print("Recall: "+ str(recall))
    print("Precision: "+str(precision))
    print("f-measure: "+str(FVal))
    
    
def logReg(trainMat, yTrain, stepLimit, learning_rate):
    trainMat = np.append(arr = np.ones([trainMat.shape[0],1]).astype(int), values = trainMat,axis=1)  #concatenate the ones to train matrix
    dim = trainMat.shape[1]
    theta = np.zeros((dim,1))
    size = np.size(trainMat,0)
        
    i=0
    while i< stepLimit:
        scores = np.dot(trainMat,theta)
        predictions = sigmoid(scores)
        
        
        #gradient ascent
        output = (np.asarray(yTrain).T-np.asarray(predictions).T).T
        
        gradient = (1/size)*np.dot(trainMat.T,output)
        theta+= learning_rate*gradient
        i+=1
        
    return theta
    
#accepts X, y and returns the weights for computing a linear regression
def linReg(trainMat, yTrain):
    trainMat = np.append(arr = np.ones([trainMat.shape[0],1]).astype(int), values = trainMat,axis=1)  #concatenate the ones to train matrix
    Xt = trainMat.transpose()
    XtX = np.dot(Xt,trainMat)
    theta = np.dot(np.dot(np.linalg.inv(XtX), Xt), yTrain)
    return theta

def getPredictedValues(weights, X):
    X = np.append(arr = np.ones([X.shape[0],1]).astype(int), values = X,axis=1)  #concatenate the ones to train matrix
    pred_outputs = np.matmul(X, weights)
    return pred_outputs

#calculate hypothesis using sigmoid function
def calcHypothesis(weights,x):  
    h=sigmoid(np.dot(x,weights))
    return h

def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    
    #1) read in data
    df = pd.read_csv('spambase.data', header=None)
    #df.rename(columns={57:'is_spam'}, inplace=True)
    
    #2) randomize the data
    data = df.to_numpy()
    np.random.shuffle(data)
    print(df[:5])
    
    
    #3) split to test and train datasets
    trainCount = int(len(data)*2/3)
    trainMat, testMat = data[:trainCount, :],data[trainCount:,:]    #https://stackoverflow.com/questions/3674409/how-to-split-partition-a-dataset-into-training-and-test-datasets-for-e-g-cros
    
    yTrain = np.asarray(trainMat[:,-1])    #get values for train y
    yTest = testMat[:,-1]     #get values for test y
    
    #delete y values from original matrices
    testMat = np.delete(testMat,-1,1)   
    trainMat = np.delete(trainMat,-1,1) 
    
    #4) standardize the matrices
    trainMean = np.mean(trainMat,axis=0)
    trainDev = np.std(trainMat,axis=0)
    trainMat = (trainMat - trainMean) / trainDev  
    #standardize testmat with mean and dev from train?
    testMat = (testMat - trainMean) / trainDev    
    
    #add back the y values for sample split
    trainMat = np.column_stack((trainMat, yTrain))

    #5) Divides the training data into two groups: trainSpamSamples, trainNonSpamSamples
    trainSpamSamples = trainMat[trainMat[:,-1]==1]
    ySpam = trainSpamSamples[:,-1]
    trainSpamSamples=np.delete(trainSpamSamples,-1,1)  
    
    trainNonSpamSamples = trainMat[trainMat[:,-1]==0]
    yNonSpam= trainNonSpamSamples[:,-1]
    trainNonSpamSamples=np.delete(trainNonSpamSamples,-1,1)
    
    trainMat = np.delete(trainMat,-1,1) 
    #Weights = linReg(trainMat, yTrain)
    
    
    stepLimit = 30000
    learning_rate = 0.5
    #10*math.exp(-7)
    
    
    Weights = logReg(trainMat, yTrain, stepLimit, learning_rate)
    
    yPredictions = getPredictedValues(Weights,testMat)
    yPredictionsFinal = log_likelihood(testMat,yTest,Weights)
    yPredictions = yPredictions > 0.5

        
    #8) now we have the predictions array, we compare it with original to get our model statistics    
    printRunStatistics(yPredictions, yTest)