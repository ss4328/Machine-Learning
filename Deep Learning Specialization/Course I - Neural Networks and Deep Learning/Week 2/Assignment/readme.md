# Assignment: Logistic regression (for NN)

## About:
Cat/Non-cat classifier using Logistic Regression

## Notes:
I treaked learning rate to 0.02 as it yielded better result than the default 0.01 by coursera. If you need the rationale, see the graph drawn for the learning_rate comparison. 

- You can see that 0.02 results for a lower cost. 
- However, LR or 0.02 spikes cost up for 0-2k and 2-6k iterations by a huge number. Was this a good decision? Does it have repurcususions? 
- Learning rate is an hyperparameter. This questions (related to hyperparameter tuning) will be addressed in depth in course 2: Hyperparameter tuning. Consider this a trailer. 



## Changes:
- Learning rate tweaked to 0.02 instead of suggested 0.01 (I'm a rebel)
- Removed some code related to deprecated library scipy.misc. Added PILLOW library code that does the same. 

(4/22/20) In some point, it'd be amazing to create a jupyter notebook for this code. I'll try to do that later sometime when I have time/energy.