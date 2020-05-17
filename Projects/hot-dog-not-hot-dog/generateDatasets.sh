#!/bin/sh

#1/4. Generate Training set -> hot dogs
python h5Converter.py /Users/shivanshsuhane/Desktop/code/ml_gitRepo/projects/hot-dog-not-hot-dog/datasets/combinedData/test testData
 
#2/4. Generate Training set -> NOT hot dogs
python h5Converter.py /Users/shivanshsuhane/Desktop/code/ml_gitRepo/projects/hot-dog-not-hot-dog/datasets/combinedData/train trainData

mv /Users/shivanshsuhane/Desktop/code/ml_gitRepo/projects/hot-dog-not-hot-dog/testData.h5 /Users/shivanshsuhane/Desktop/code/ml_gitRepo/projects/hot-dog-not-hot-dog/datasets/testSet.h5
mv /Users/shivanshsuhane/Desktop/code/ml_gitRepo/projects/hot-dog-not-hot-dog/trainData.h5 /Users/shivanshsuhane/Desktop/code/ml_gitRepo/projects/hot-dog-not-hot-dog/datasets/trainSet.h5

echo "All files translate. H4s generated @root dir location"