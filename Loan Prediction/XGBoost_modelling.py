# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:17:18 2018

@author: p.soni
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
## sklearn wrapper for xgboost--> helps to use sklearn grid search with parallel processing.
from xgboost.sklearn import XGBClassifier

from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4

def modelfit(alg,data, predictors, outcome, useTrainCV = True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(data[predictors].values, label=data[outcome].values)
        
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round = alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
        
    alg.fit(data[predictors], data[outcome], eval_metric = 'auc')
    
    predictions = alg.predict(data[predictors])
    data_predprob = alg.predict_proba(data[predictors])[:,1]
    
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(data[outcome].values, predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(data[outcome], data_predprob))
    

train = pd.read_csv("D:\\Timepass\Loan Predictor\Data\\train_final.csv")
test = pd.read_csv("D:\\Timepass\Loan Predictor\Data\\test_final.csv")

train.drop(["Unnamed: 0","index"], axis=1, inplace=True)
test.drop(["Unnamed: 0","index"], axis=1, inplace=True)

cat_var = ["Education", "Gender", "Married", "Property_Area", "Self_Employed", "Dependents"]

le = LabelEncoder()

for i in cat_var:
    train[i] = le.fit_transform(train[i])
    

outcome_var = "Loan_Status"

"""
Approach for XGBoost parameter tunning

1. Choose a relatively high learning rate. Generally a learning rate of 0.1 works 
    but somewhere between 0.05 to 0.3 should work for different problems. 
2. Determine the optimum number of trees for this learning rate. XGBoost has a very useful 
    function called as “cv” which performs cross-validation at each boosting iteration and
    thus returns the optimum number of trees required.
4. Tune tree-specific parameters ( max_depth, min_child_weight, gamma, subsample, colsample_bytree)
    for decided learning rate and number of trees. Note that we can choose different parameters to 
    define a tree and I’ll take up an example here.
5. Tune regularization parameters (lambda, alpha) for xgboost which can help reduce model complexity 
    and enhance performance.
6. Lower the learning rate and decide the optimal parameters .
"""

### Fix learning rate and number of estimators for tuning tree-based parameters

print("\nXGBoost classifier Fix learning rate")
predictor_var = ['Credit_History', 'Dependents', 'Education', 'Gender', 'LoanAmount',
       'Loan_Amount_Term', 'Married', 'Property_Area',
       'Self_Employed', 'total_income']

xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

modelfit(xgb1, train, predictor_var, outcome_var)

###  Tune max_depth and min_child_weight( optimum around 5 for both)
## It takes very much time
"""
param_test1 = {
 'max_depth': [4,5,6],
 'min_child_weight':[4,5,6]
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(train[predictor_var],train[outcome_var])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
"""

## Tune Gamma
print("Tunning Gamma:")
param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch3.fit(train[predictor_var],train[outcome_var])
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
