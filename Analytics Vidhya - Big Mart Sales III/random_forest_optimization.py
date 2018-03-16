#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 19:06:06 2018

@author: piyush
"""
import numpy as np
import pandas as pd
from math import sqrt
import seaborn as sns

from matplotlib import pyplot as plt
from scipy.stats import boxcox

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from pprint import pprint
from sklearn.ensemble import RandomForestRegressor


from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold


save = True
file_path="/media/piyush/New Volume/Data Science/Analytics Vidhya - Big Mart Sales III/Submissions/RandomForest_RandomizedSearch_without_skew.csv"


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

rf = RandomForestRegressor()

#pprint(rf.get_params())


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(random_grid)

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

#rf_random.fit(train[predictors], train[target])
"""
Best Params from RandomizedSearchCV:
{'bootstrap': True,
 'max_depth': 10,
 'max_features': 'auto',
 'min_samples_leaf': 4,
 'min_samples_split': 5,
 'n_estimators': 200}
"""
"""
if(save):       #(LeaderBoard : 1199, one hot : 1205)
    file_path="/media/piyush/New Volume/Data Science/Analytics Vidhya - Big Mart Sales III/Submissions/RandomForest_RandomizedSearch_without_skew.csv"

    test_predictions = inv_boxcox(rf_random.predict(test[predictors]), lambda_)
    
    base2 = test_original[['Item_Identifier', 'Outlet_Identifier']]
    base2["Item_Outlet_Sales"] = test_predictions

    base2.to_csv(file_path, index=False)
"""
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [5, 10, 15, 20],
    'max_features': [12],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [3, 5, 7],
    'n_estimators': [100, 150, 250, 300, 350]
}

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
#grid_search.fit(train[predictors], train[target])

"""
{'bootstrap': True,
 'max_depth': 5,
 'max_features': 12,
 'min_samples_leaf': 3,
 'min_samples_split': 3,
 'n_estimators': 300}
"""

# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': np.arange(1,10,1),
    'max_features': [12],
    'min_samples_leaf': [1,2,3],
    'min_samples_split': [2,3,4],
    'n_estimators': [260, 280, 300, 320, 340]
}

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
#grid_search.fit(train[predictors], train[target])
"""
{'bootstrap': True,
 'max_depth': 5,
 'max_features': 12,
 'min_samples_leaf': 3,
 'min_samples_split': 4,
 'n_estimators': 340}
"""

if(save):       #(LeaderBoard : 1199)
    file_path="/media/piyush/New Volume/Data Science/Analytics Vidhya - Big Mart Sales III/Submissions/RandomForest_GridSearchCVSearch_without_skew.csv"

    test_predictions = inv_boxcox(grid_search.predict(test[predictors]), lambda_)
    
    base2 = test_original[['Item_Identifier', 'Outlet_Identifier']]
    base2["Item_Outlet_Sales"] = test_predictions

    base2.to_csv(file_path, index=False)