# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 00:00:06 2018

@author: Piyush
"""

import numpy as np
import pandas as pd
from math import sqrt
from scipy.stats import boxcox
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold

from matplotlib import pyplot as plt

save = True


test_original = pd.read_csv("/media/piyush/New Volume/Data Science/Analytics Vidhya - Big Mart Sales III/Data/test.csv")


def inv_boxcox(y_box, lambda_):
    pred_y = np.power((y_box * lambda_) + 1, 1 / lambda_) - 1
    return pred_y

def regression_model(alg, data, predictors, target, n_fold=5, save=False, lambda_=None, test_df=None, file_path=None):
    
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
        test_predictions = inv_boxcox(alg.predict(test_df[predictors]), lambda_)
        
        base2 = test_original[['Item_Identifier', 'Outlet_Identifier']]
        base2["Item_Outlet_Sales"] = test_predictions
    
        base2.to_csv(file_path, index=False)
    
    
    
train = pd.read_csv("/media/piyush/New Volume/Data Science/Analytics Vidhya - Big Mart Sales III/Data/train_final.csv")
test = pd.read_csv("/media/piyush/New Volume/Data Science/Analytics Vidhya - Big Mart Sales III/Data/test_final.csv")

train.drop("Unnamed: 0", inplace=True, axis=1)
test.drop("Unnamed: 0", inplace=True, axis=1)


## Checking skewness of Sales Column
train.Item_Outlet_Sales.skew()
## Treating skewness lop1p transform was not enough to treat skewness thus boxcox transform is taken
#sns.distplot(np.log1p(total.Item_Outlet_Sales[total.isTrain == 1]))
sns.distplot(boxcox(train.Item_Outlet_Sales)[0])

train.Item_Outlet_Sales, lambda_ = boxcox(train.Item_Outlet_Sales + 1)


predictors = test.columns

target = 'Item_Outlet_Sales'

## Baseline (LeaderBoard : 1773, without_skew : 1826, one_hot : 1826)
mean_sales = train.Item_Outlet_Sales.mean()

base1 = test_original[['Item_Identifier', 'Outlet_Identifier']]
base1["Item_Outlet_Sales"] = inv_boxcox(mean_sales,lambda_)

if(save):
    base1.to_csv("/media/piyush/New Volume/Data Science/Analytics Vidhya - Big Mart Sales III/Submissions/mean_without_skew.csv", index=False)

## Linear Regression (LeaderBoard : 1232, without_skew : 1206, one hot encodeing : 1210)
print("\nLinear :\n")
lm = LinearRegression()
regression_model(lm, train, predictors, target, save=save, test_df=test, lambda_=lambda_, file_path="/media/piyush/New Volume/Data Science/Analytics Vidhya - Big Mart Sales III/Submissions/LinearRegression_without_skew.csv")


## Ridge model (LeaderBoard : 1251, without_skew : 1245, one hot encodeing : 1224)
print("\nRidge :\n")
alg = Ridge(alpha=0.05,normalize=True)
regression_model(alg, train, predictors, target, save=save, test_df=test, lambda_=lambda_, file_path="/media/piyush/New Volume/Data Science/Analytics Vidhya - Big Mart Sales III/Submissions/RidgeRegression_without_skew.csv")
coef2 = pd.Series(alg.coef_, predictors).sort_values()
coef2.plot(kind='bar', title='Model Coefficients')

## Decision Trees  (LeaderBoard : 1191, without_skew : 1204 ,one hot encoding : 1203)
print("\nDecision Trees :\n")
print("alg1:")
alg = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
regression_model(alg, train, predictors, target, save=save, test_df=test, lambda_=lambda_, file_path="/media/piyush/New Volume/Data Science/Analytics Vidhya - Big Mart Sales III/Submissions/DecisionTree_without_skew.csv")

print("\nalg2:") ## (LeaderBoard : 1184, without_skew : 1199, one hot encoding : 1199)
alg = DecisionTreeRegressor(max_depth=8, min_samples_leaf=150)
regression_model(alg, train, predictors, target, save=save, test_df=test, lambda_=lambda_, file_path="/media/piyush/New Volume/Data Science/Analytics Vidhya - Big Mart Sales III/Submissions/DecisionTree1_without_skew.csv")

## Random Forest (LeaderBoard : 1186, without_skew : 1198, one hot encoding : 1198)
print("\nRandom Forest :\n")
print("alg1:")
alg = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
regression_model(alg, train, predictors, target, save=save, test_df=test, lambda_=lambda_, file_path="/media/piyush/New Volume/Data Science/Analytics Vidhya - Big Mart Sales III/Submissions/RandomForest_without_skew.csv")

print("\nalg2:") ## (LeaderBoard : 1184.79, without_skew : 1197, one hot encoding : 1198)
alg = RandomForestRegressor(n_estimators=400,max_depth=6, min_samples_leaf=100,n_jobs=4)
regression_model(alg, train, predictors, target, save=save, test_df=test, lambda_=lambda_, file_path="/media/piyush/New Volume/Data Science/Analytics Vidhya - Big Mart Sales III/Submissions/RandomForest1_without_skew.csv")



