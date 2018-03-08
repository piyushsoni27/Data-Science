# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 00:00:06 2018

@author: Piyush
"""

import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

def model_fit(alg, xtrain, ytrain, cv_ratio):
    # create training and testing vars
    X_train, X_test, Y_train, Y_test = train_test_split(xtrain, ytrain, test_size=cv_ratio)
    
    alg.fit(X_train, Y_train)
    
    train_prediction = alg.predict(X_train)
    test_prediction = alg.predict(X_test)
    
    ## Mean Squared Error train set
    print("Mean Squared Error Train: %.2f",  mean_squared_error(Y_train, train_prediction))
    
    ## Mean Squared Error test set
    print("Mean Squared Error Test: %.2f",  mean_squared_error(Y_test, test_prediction))
    
    ## Variance Error train:
    print("Variance Error Train: %.2f",  r2_score(Y_train, train_prediction))
    
    ## Variance Error test:
    print("Variance Error Test: %.2f",  r2_score(Y_test, test_prediction))
    
    return


# fit a model Linear Regression
lm = linear_model.LinearRegression()

model = lm.fit(X_train, Y_train)
predictions = lm.predict(X_test)
print(model.score)

## Ridge model
predictors = [x for x in X_train.columns if x not in [target]+IDcol]
alg2 = Ridge(alpha=0.05,normalize=True)
modelfit(alg2, train, test, predictors, target, IDcol, 'alg2.csv')
coef2 = pd.Series(alg2.coef_, predictors).sort_values()
coef2.plot(kind='bar', title='Model Coefficients')