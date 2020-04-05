# Dimentionality Reduction

In this project, my aim is to create my own PCA algorithm from scratch by using matlab. 

## Execution

Run the .m script in root directory via matlab


## Theory
Dimentionality reduction is done to ensure that the data we receive is efficiently tailored for usage. In statistics, machine learning, and information theory, dimensionality reduction or dimension reduction is the process of reducing the number of random variables under consideration by obtaining a set of principal variables. Approaches can be divided into feature selection and feature extraction.

We can do this by a number of ways: choosing only a certain number of features (makes you wonder which ones thought?), or deleting/adding features greedily (again which ones though?) or come up with a method to create new features from the data itself. We can see the first two approaches run into some very awkward quesions which directly or indirectly lead into us refer the data itself and make a data-centered solution. However the third approach: Creating new features from the data itself, is quite generic which is why we want to choose it for most times. This approach is called Principal Component Analysis (PCA) and is a very easy Machine Learning Algorithm.


The main linear technique for dimensionality reduction, principal component analysis, performs a linear mapping of the data to a lower-dimensional space in such a way that the variance of the data in the low-dimensional representation is maximized. In practice, the covariance (and sometimes the correlation) matrix of the data is constructed and the eigenvectors on this matrix are computed. The eigenvectors that correspond to the largest eigenvalues (the principal components) can now be used to reconstruct a large fraction of the variance of the original data. Moreover, the first few eigenvectors can often be interpreted in terms of the large-scale physical behavior of the system, because they often contribute the vast majority of the system's energy, especially in low-dimensional systems. Still, this must be proven on a case-by-case basis as not all systems exhibit this behavior. The original space (with dimension of the number of points) has been reduced (with data loss, but hopefully retaining the most important variance) to the space spanned by a few eigenvectors.




## Explain me like I'm Five? 
Okay here we go. Imagine I give you the option of choosing ANY amusement park in the world and I promise you I'll take you there. Now if you're a modern prodigy like Carl Gauss (my favorite mathematician btw) and you have all the modern tools at your disposal i.e. the internet, you'll quickly google 'Best amusement parks in the world!' or something similar in your rudimentary 5y/o vocabolary. 4 hours later, you have an excel list of world's 2000 best amusement parks and you want to find the best one. Now I know 'best' is subjective and being a 5 y/o with limited understanding of the world you try to not be biased about any rides, you want the wholistic package deal.

You are an industrious 5 y/o and have curated 15 data points per park like: Number of ferris wheels, roller coasters height, location weather, number of candies, ice cream flavors served, games, central theme(s) count, cartoon characters count, etc etc. You quickly discover that most of these are same across all entertainment parks! (Ofc disneyland is the best, but it does have similar roller coasters than say wonderla?) Your work is now getting harder! All of them look the same!

Now you bother your mom for some icecream and think calmly what to do... 

You quickly realize you can just look at the things(features) that differ the most and then make a decision! What an ingenious idea??! You quickly realize location weather is so far the most diverse feature in your dataset, followed by the cartoon characters count, then number of ferris wheels. Great! You now ignore all the other features and just focus on what's differing the most and find your perfect amusement park.

(Pss, you might wanna standardize your excel data first or you might land somewhere weird)

## Data + Project
What we have here is the yalefaces dataset and my aim is to find the principal components, generate eigenfaces, and recreate the facial images using matlab. 

Process:

1. Read data and standardize data to a matrix.
2. Find most significant columns by using covarience matrix
3. Find eigenfaces
4. Recreate images by multiplying stored dataset with most significant eigenvectors
5. Make a video to see change in quality per eigenface addition

