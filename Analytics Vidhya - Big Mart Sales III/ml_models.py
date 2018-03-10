# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 00:00:06 2018

@author: Piyush
"""

import numpy as np
import pandas as pd
from math import sqrt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold

from matplotlib import pyplot as plt

test_original = pd.read_csv("D:\\Timepass\Big mart\Data\\test.csv")

def regression_model(alg, data, predictors, target, n_fold=5, save=False, test_df=None, file_path=None):
    
    alg.fit(data[predictors], data[target])
    
    predictions = alg.predict(data[predictors])
    
    error = sqrt(mean_squared_error(data[target], predictions))
    print("Train Error : %s" % "{0:}".format(error))
    
    kf = KFold(data.shape[0], n_folds=n_fold)
    fold_error = []
    
    for train, test in kf:
        train_predictors = data[predictors].iloc[train, :]
        train_target = data[target].iloc[train]
        
        alg.fit(train_predictors, train_target)
        
        fold_error.append(sqrt(mean_squared_error(data[target].iloc[test], alg.predict(data[predictors].iloc[test, :]))))
        
    print("Cross-Validation Score : %s" % "{0:}".format(np.mean(fold_error)))
    
    alg.fit(data[predictors], data[target])
    
    if(save):
        test_predictions = alg.predict(test_df[predictors])
        
        base2 = test_original[['Item_Identifier', 'Outlet_Identifier']]
        base2["Item_Outlet_Sales"] = test_predictions
    
        base2.to_csv(file_path, index=False)
    
    
    
train = pd.read_csv("D:\\Timepass\Big mart\Data\\train_final.csv")
test = pd.read_csv("D:\\Timepass\Big mart\Data\\test_final.csv")
test_original = pd.read_csv("D:\\Timepass\Big mart\Data\\test.csv")

test.drop("Unnamed: 0", inplace=True, axis=1)

predictors = ['Item_Fat_Content', 'Item_MRP', 'Item_Type',
       'Item_Visibility', 'Item_Weight', 'Outlet_Identifier',
       'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type',
       'Outlet_Establishment_Year_from_2018', 'Item_Code', 'Item_Category']

target = 'Item_Outlet_Sales'

## Baseline (LeaderBoard : 1773)
mean_sales = train.Item_Outlet_Sales.mean()

base1 = test_original[['Item_Identifier', 'Outlet_Identifier']]
base1["Item_Outlet_Sales"] = mean_sales

base1.to_csv("D:\Timepass\Big mart\Submissions\mean.csv", index=False)

## Linear Regression (LeaderBoard : 1232)
print("\nLinear :\n")
lm = LinearRegression()
regression_model(lm, train, predictors, target)


## Ridge model (LeaderBoard : 1251)
print("\nRidge :\n")
alg = Ridge(alpha=0.05,normalize=True)
regression_model(alg, train, predictors, target, save=False, test_df=test, file_path="D:\Timepass\Big mart\Submissions\RidgeRegression.csv")
coef2 = pd.Series(alg.coef_, predictors).sort_values()
coef2.plot(kind='bar', title='Model Coefficients')

## Decision Trees  (LeaderBoard : 1191)
print("\nDecision Trees :\n")
print("alg1:")
alg = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
regression_model(alg, train, predictors, target, save=False, test_df=test, file_path="D:\Timepass\Big mart\Submissions\DecisionTree.csv")

print("\nalg2:") ## (LeaderBoard : 1184)
alg = DecisionTreeRegressor(max_depth=8, min_samples_leaf=150)
regression_model(alg, train, predictors, target, save=False, test_df=test, file_path="D:\Timepass\Big mart\Submissions\DecisionTree2.csv")

## Random Forest (LeaderBoard : 1186)
print("\nRandom Forest :\n")
print("alg1:")
alg = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
regression_model(alg, train, predictors, target, save=False, test_df=test, file_path="D:\Timepass\Big mart\Submissions\RandomForest.csv")

print("\nalg2:") ## (LeaderBoard : 1184.79)
alg = RandomForestRegressor(n_estimators=400,max_depth=6, min_samples_leaf=100,n_jobs=4)
regression_model(alg, train, predictors, target, save=True, test_df=test, file_path="D:\Timepass\Big mart\Submissions\RandomForest1.csv")



