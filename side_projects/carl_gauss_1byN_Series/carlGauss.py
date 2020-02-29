#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:14:46 2020

@author: shivanshsuhane
"""

#calc the number of terms when they converge to  more than one in the series 1/1, 1/2, 1/3, 1/4 ..... 1/n

import matplotlib.pyplot as plt

#slow as shit fn    Bottleneck fn using fuunctional approach
def sumOfFractions(fractionSet):            #calculates the sum of a set of fractions
    res =0
    for frac in fractionSet:
        res+=frac
    return res



limit = 19       #terms total for the plot
sum =5

n=1
plotPoints = []


condition = True

thisFractionNumbersAdded = 0


thisOneSum =0;

#fractionSet=[]
while len(plotPoints)<limit:
    thisFraction = 1/n
    #fractionSet.append(thisFraction)
    thisFractionNumbersAdded +=1
    thisOneSum+=thisFraction
    if thisOneSum>=1:   #sum of fractions <1 add frac to list)
        plotPoints.append(thisFractionNumbersAdded)
        thisFractionNumbersAdded=0
        fractionSet = []
        thisOneSum=0
    n+=1
            
        
plt.plot(plotPoints)
plt.show()


#how many terms are needed to reach x?
