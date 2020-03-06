#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 20:30:50 2020

@author: shivanshsuhane
"""

import numpy as np
import pandas as pd
import math


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
        
    #i=0
   # for pred in preds:
    #    label=labels[i]
    #    if pred==1:
     #       if pred==label:
      #          TP+=1
      #          accu+=1
    #        else:
     #           FP+=1
     #   else:
      #      if pred==label:
      #          TN+=1
      #          accu+=1
       #     else:
        #        FN+=1
    #
    
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    FVal = 2*precision*recall/(precision+recall)
    accu=accu/np.size(preds,0)
    print("Accuracy :"+ str(accu))
    print("Recall: "+ str(recall))
    print("Precision: "+str(precision))
    print("f-measure: "+str(FVal))

def getMeanAndDeviation(X):
    columnMean = []
    columnDev =[]
    
    for column in X.T[:-1]:
        columnMean.append(np.mean(column))
        columnDev.append(np.std(column, ddof=1))
        print(columnMean)
        print(columnDev)

    return np.asarray(columnMean), np.asarray(columnDev)    #convert to np arrays before returning


def getGaussian(featureVal, mean, dev):
    #if dev =0, feature is useless so just return 0
    if(dev==0):
        return 0
    e = math.e
    denom=1         #do we need denominator? not really since we're just comparing values
    exponent = math.exp(-((featureVal-mean)**2 / (2 * dev**2 )))
    return (1 / (math.sqrt(2 * math.pi) * dev)) * exponent
    

def getBayesianProbability(obs, mean, dev, usualP):
    obs = np.transpose(obs)
    p=usualP
    
    #calc gaussian value for each feature
    i=0
    for featureVal in obs:
        p*=getGaussian(featureVal, mean[i], dev[i])
        
    return p

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
    
    yTrain = trainMat[:,-1]     #get values for train y
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
    trainSpamSamples=np.delete(trainSpamSamples,-1,1)  
    
    trainNonSpamSamples = trainMat[trainMat[:,-1]==0]
    trainNonSpamSamples=np.delete(trainNonSpamSamples,-1,1)  
    
    
    #calc mean, stddev of spam and not spam data
    spamMean, spamDev = getMeanAndDeviation(trainSpamSamples)
    nonSpamMean, nonSpamDev = getMeanAndDeviation(trainNonSpamSamples)
    overallSpamP, overallNonSpamP = np.size(trainSpamSamples,0)/np.size(trainMat,0), np.size(trainNonSpamSamples,0)/np.size(trainMat,0)
    
    #calc bayesian probability for being spam r not spam based on mean and dev
    #whichever prob is higher will decide spam or not spam
    yPredictions=[]
    
    #6) Create gaussian model for each feature and use it to classify
    testMatTransposed= np.transpose(testMat)
    for obs in testMat:
        PSpam = getBayesianProbability(obs,spamMean, spamDev, overallSpamP)
        print(PSpam)
        PNotSpam = getBayesianProbability(obs,nonSpamMean, nonSpamDev, overallNonSpamP)
        print(PNotSpam)
        #7) Classify each testing sample using these models and choose the class label based on which class probability is higher.
        if(PSpam >= PNotSpam):
            yPredictions.append(1)
        else:
            yPredictions.append(0)
        
    #8) now we have the predictions array, we compare it with original to get our model statistics    
    printRunStatistics(yPredictions, yTest)
        