# Assignment: Logistic regression (for NN)

## About:
Hot dog/Not-hot-dog classifier using Logistic Regression and Neural Network.

## Working:
### Image processing: 
- Loop over the images and resize each image to 100 x 100 px.
- Sort the images into train (HD + NHD) and test (HD + NDH) sets.
- Flatten each image to 100x100x1. 
- Randomly mix the images in train and test in themselves.
- Slice the ground-truth values into a separate (Y) arrays.

## Algorithm:
- Compute loss on each turn, minimize the loss using gradient descent
Algorithm used: 
[![log-reg-algo.png](https://i.postimg.cc/qqWdcQ6q/log-reg-algo.png)](https://postimg.cc/Lgk72tTF)

## Results:
- 56% accuracy in test, 93% in train while using the custom made Logistic regression classifier
- 53% accuracy in test while using Scikit-learn classifier
- Precision and recall hover between 53-68%

## Improvements:
- Using Neural Networks to make the classification
- Splitting the data for a better ratio (right now its 50-50) than what's provided from kaggle

## Notes:
I treaked learning rate to 0.02 as it yielded better result than the default 0.01. If you need the rationale, see the graph drawn for the learning_rate comparison. 
- You can see that 0.02 results for a lower cost. 
- However, LR or 0.02 spikes cost up for 0-2k and 2-6k iterations by a huge number. Was this a good decision? Does it have repurcususions? 
- Learning rate is an hyperparameter. This questions (related to hyperparameter tuning) will be addressed in depth in course 2: Hyperparameter tuning. Consider this a trailer. 

Q) Did I have fun doing this?
A) HELL YES

## Changes:
- Learning rate tweaked to 0.02 instead of suggested 0.01
- Removed some code related to deprecated library scipy.misc. Added PILLOW library code that does the same. 

(5/22/20) In some point, it'd be amazing to create a jupyter notebook for this code. I'll try to do that later sometime when I have time/energy.