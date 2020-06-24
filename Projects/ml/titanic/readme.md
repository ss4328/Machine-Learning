# Titanic: Machine Learning from Disaster

## About
This project is my attempt to solve Kaggle's famous Titanic Passenger survival prediction challenge. The competition is simple: use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.

https://www.kaggle.com/c/titanic/


## The Challenge
The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc)

In the following readme file, I'll document my method, and in-depth breakdown and logs of how I worked through the problem.

## Method

### Overview: 

Following are the steps I'll take in the challenge:
1. Data Exploration: Before embarking on this challenge, I'll explore & visualize the data to find+fix missing data, create a mental map of some patterns linking data to prediction results. The core question this part will address is: What are the existing nominal links that exist b/w data and predictions? 
2. Preprocessing: I'll preprocess the given datasets to create train and test datasets worthy of being feeded to my model
3. Model Training: Selecting the algorithm, actually training it
4. Optimization: Making the model faster, and more accurate, tuning hyperparameters if needed.
5. Result analysis: A v. basic analysis of the complexity of the model, checking if predictions make sense, finding the patterns where the model is underperforming.

Afterwords, I'll repeat steps #3,#4,#5 until satisfied with the result.

### Data Exploration:

 My first instinct is to make sense of the data to make out how to make sense of the data; this is an important step as it guides to pick the optimal model(s) available for usage. This could be the most time-consuming part of the problem. 

#### Data Exploration:
In the training data, I have the entries of 891 passengers and each has 12 features:
- PassengerID,
- Survived: Survival - 0 = No, 1 = Yes,
- pClass: Ticket class - 1 = 1st, 2 = 2nd, 3 = 3rd, 
- Name, 
- Sex, 
- Age: Age in years,
- SubSp: # of siblings / spouses aboard the Titanic
- Parch: # of parents / children aboard the Titanic
- Ticket: Ticket number,
- Fare: Passenger fare,
- Cabin: Cabin number,
- Embarked: Port of Embarkation - C = Cherbourg, Q = Queenstown, S = Southampton

Data Breakdown: 
- Categorical Data - Represents charactersitcs; no mathematical meaning
	1. Nominal Data - Data has no order: Sex, Embarked
	2. Ordinal Data - Data's ordering matters: pClass
- Numerical Data - Has mathematical meaning
	1. Discrete Data - sibSp, Parch, 
	2. Continuous Data - Age, Fare
- Useless Data - Noise we can't use (might get updated with new insights) - Name, Ticket, Cabin
- Results column - What we're predicting: Survived

Initial observation (Mayb bias model):
I think the most deciding factory may be Age, Fare, (subSp,Parch), Sex, pClass. Since these could dictate mostly who will survive; Emarged port could have little meaning, Name, Ticket ID, Cabin could have little to no impact whatsoever. 

There are basic data visualizations on Kaggle. 


### Preprocessing

Data Cleaning:
- Several entries are empty: Age, Cabin, Embarked 
- For missing ages, We'll impute data with strategy = 'mean'; this will fill missing ages with age's mean. 
- We don't need to fill in Cabins, there's so much data missing to just drop the usage of cabin values altogether. 
- For embarked, only two values are missing so we'll just randomly fill in embarked port. 
- We also drop Names from our training dataset, I think they don't add any value for computation

Transforming Catagorical Data:
- We have two fratures - Embarked, and Passenger class that would serve us better to get OneHotEncoded. Data is hardly usable here as a string
- Similarly, Sex is also OneHotEncoded.

Standardizing data:
- Fare, and Age are features that'll serve best after standardization.
- Data standardizing is seldom hard and often provides useful value. It ensures no particular feature overpowers our prediction
- I'll go for standardization as opposed to normalization as the data we have isn't normally distributed. 



## Credits
Since this is my First ML project, I'll probably learn the most in this project. I'm thankful to:
- Several medium authors for valuable insights on several parts of the problem. Here are the blog posts I made use of: 
1. https://towardsdatascience.com/data-types-in-statistics-347e152e8bee
2. https://towardsdatascience.com/data-preprocessing-for-non-techies-feature-exploration-and-engineering-f1081438a5de
