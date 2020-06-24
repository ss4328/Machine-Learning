#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 20:53:01 2020

@author: shivanshsuhane
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[1,2,-1])], remainder='passthrough')  #pClass, and embarked
sc  = StandardScaler()

def loadData():
    dataset = pd.read_csv("train.csv")
    dataset= preProcessFill(dataset)
    df=pd.DataFrame(dataset)
#    df=preProceessNormalizeData(df)
    
    Y = df[df.columns[1]].to_numpy()
    Ydim0=np.shape(Y)[0]
    print(Ydim0)
    Y = Y.reshape((Ydim0,1))

    usecols=[0,2,4,5,6,7,8,9,11]       #doesn't take in cabin or survived or name
    X = df[df.columns[usecols]].to_numpy()
    
    return X,Y
    
def preProcessFill(dataset):
    dataset['Age'].fillna((dataset['Age'].mean()),inplace=True)
    dataset['Embarked'].fillna('C',inplace=True)
    
    return dataset


def preProcessCategoricalData(X):
    X = np.array(ct.fit_transform(X))
    return X
    
def preProceessNormalizeData(df):
    df[['Age', 'Fare']] = sc.fit_transform(df[['Age', 'Fare']])
    return df
    

if __name__ == "__main__":
    X, Y = loadData()
    
    X = preProcessCategoricalData(X)