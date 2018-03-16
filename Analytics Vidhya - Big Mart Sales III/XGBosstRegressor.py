#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 23:38:33 2018

@author: piyush
"""
import numpy as np
import pandas as pd
from math import sqrt

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import boxcox

from xgboost.sklearn import XGBRegressor

save = True


test_original = pd.read_csv("/media/piyush/New Volume/Data Science/Analytics Vidhya - Big Mart Sales III/Data/test.csv")

def inv_boxcox(y_box, lambda_):
    pred_y = np.power((y_box * lambda_) + 1, 1 / lambda_) - 1
    return pred_y

train = pd.read_csv("/media/piyush/New Volume/Data Science/Analytics Vidhya - Big Mart Sales III/Data/train_final.csv")
test = pd.read_csv("/media/piyush/New Volume/Data Science/Analytics Vidhya - Big Mart Sales III/Data/test_final.csv")

train.drop("Unnamed: 0", inplace=True, axis=1)
test.drop("Unnamed: 0", inplace=True, axis=1)

train.Item_Outlet_Sales, lambda_ = boxcox(train.Item_Outlet_Sales + 1)

predictors = test.columns

target = 'Item_Outlet_Sales'

xgb = XGBRegressor()

"""
Initial Parameters:
    'base_score': 0.5,
 'colsample_bylevel': 1,
 'colsample_bytree': 1,
 'gamma': 0,
 'learning_rate': 0.1,
 'max_delta_step': 0,
 'max_depth': 3,
 'min_child_weight': 1,
 'missing': None,
 'n_estimators': 100,
 'nthread': -1,
 'objective': 'reg:linear',
 'reg_alpha': 0,
 'reg_lambda': 1,
 'scale_pos_weight': 1,
 'seed': 0,
 'silent': True,
 'subsample': 1
 """
 
xgb.fit(train[predictors], train[target])
 
if(save):  # (leaderboard : 1202)
    file_path="/media/piyush/New Volume/Data Science/Analytics Vidhya - Big Mart Sales III/Submissions/xgb.csv"

    test_predictions = inv_boxcox(xgb.predict(test[predictors]), lambda_)
        
    base2 = test_original[['Item_Identifier', 'Outlet_Identifier']]
    base2["Item_Outlet_Sales"] = test_predictions
    
    base2.to_csv(file_path, index=False)

params = {
  "learning_rate" : np.arange(0.01, 0.2, 0.02),
 "min_child_weight" : np.arange(1, 50, 5),
 "max_depth" : np.arange(3,10),
 "subsample" : np.arange(0.1, 1, 0.1)
 }

xgb_random = RandomizedSearchCV(estimator = xgb, param_distributions = params, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

xgb_random.fit(train[predictors], train[target])

if(save):  # (leaderboard : 1202)
    file_path="/media/piyush/New Volume/Data Science/Analytics Vidhya - Big Mart Sales III/Submissions/xgb_random.csv"

    test_predictions = inv_boxcox(xgb_random.predict(test[predictors]), lambda_)
        
    base2 = test_original[['Item_Identifier', 'Outlet_Identifier']]
    base2["Item_Outlet_Sales"] = test_predictions
    
    base2.to_csv(file_path, index=False)

"""
{'learning_rate': 0.069999999999999993,
 'max_depth': 3,
 'min_child_weight': 11,
 'subsample': 0.90000000000000002}
"""



