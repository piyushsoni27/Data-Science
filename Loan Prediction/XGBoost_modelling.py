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

from sklearn import cross_validation, metrics   #Additional sklearn functions
from sklearn.model_selection import GridSearchCV   #Performing grid search

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
    

train = pd.read_csv("/media/piyush/New Volume/Data Science/Loan Prediction/data/train_final.csv")
test = pd.read_csv("/media/piyush/New Volume/Data Science/Loan Prediction/data/test_final.csv")

train.drop(["Unnamed: 0","index"], axis=1, inplace=True)
test.drop(["Unnamed: 0","index"], axis=1, inplace=True)

cat_var = ["Education", "Gender", "Married", "Property_Area", "Self_Employed", "Dependents"]

le = LabelEncoder()

for i in cat_var:
    train[i] = le.fit_transform(train[i])
    
for i in cat_var:
    test[i] = le.fit_transform(test[i])
    
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

print("\nXGBoost classifier without tunning(xgb1)")
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
"""
Model Report
Accuracy : 0.842
AUC Score (Train): 0.883966
"""

###  Tune max_depth and min_child_weight( optimum around 5 for both)
## It takes very much time

param_test1 = {
 #'max_depth': np.arange(3,10,2),        #Optimum value comes out as "7"
 'max_depth': [6, 7, 8],             #Checking for smaller interval around optimum value
 #'min_child_weight': np.arange(1,6,2)   #Optimum value comes out as "3"
 'min_child_weight': [2, 3, 4]      #Checking for smaller interval around optimum value
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=-1,iid=False, cv=5)
#gsearch1.fit(train[predictor_var],train[outcome_var])
#print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)

"""
Optimum values: 
    'max_depth' --> 7 
    'min_child_weight' --> 3 
"""
## Tune Gamma
print("Tunning Gamma:")
param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=7,
 min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test3, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
#gsearch3.fit(train[predictor_var],train[outcome_var])
#print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)

"""
Gamma --> 0.0
"""

print("\nXGBoost classifier tuned values(xgb2)")

xgb2 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=7,
 min_child_weight=3,
 gamma=0.0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

modelfit(xgb1, train, predictor_var, outcome_var)


## Tune subsample and colsample_bytree
print("\nTunning subsample and colsample_bytree")

param_test4 = {
 #'subsample':[i/10.0 for i in range(1,5)],         ## 0.2 optimum
 'subsample':[i/100.0 for i in range(10,30,5)],      ## checking in interval 0.05
 #'colsample_bytree':[i/10.0 for i in range(1,5)]   ## 0.2 optimum
 'colsample_bytree':[i/100.0 for i in range(10,30,5)]   ## checking in interval 0.05
}
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=7,
 min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test4, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
#gsearch4.fit(train[predictor_var],train[outcome_var])
#print(gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_)

"""
subsample --> 0.2
colsample_bytree --> 0.2
"""

# Tune Regularization parameter
print("\nTunning Regularization Parameters:")
param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=7,
 min_child_weight=3, gamma=0.0, subsample=0.2, colsample_bytree=0.2,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test6, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
#gsearch6.fit(train[predictor_var],train[outcome_var])
#print(gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_)

## Since CV score of optimum reg_alpha is less thus we try values closer to previous reg_alpha

param_test7 = {
 'reg_alpha':[0, 0.05, 0.1, 0.15, 0.2]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=7,
 min_child_weight=3, gamma=0.0, subsample=0.2, colsample_bytree=0.2,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test6, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
#gsearch6.fit(train[predictor_var],train[outcome_var])
#print(gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_)

"""
reg_alpha --> 0.1
"""
print("\nXGBoost classifier tuned values(xgb3)")
xgb3 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=7,
 min_child_weight=3,
 gamma=0,
 subsample=0.2,
 colsample_bytree=0.2,
 reg_alpha=0.1,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb3, train, predictor_var, outcome_var)
"""
Result(xgb3):
    
Model Report
Accuracy : 0.8306
AUC Score (Train): 0.822139
"""

print("\nXGBoost classifier reduced learning rate(xgb4)")
xgb4 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=1000,
 max_depth=7,
 min_child_weight=3,
 gamma=0,
 subsample=0.2,
 colsample_bytree=0.2,
 reg_alpha=0.1,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
#modelfit(xgb3, train, predictor_var, outcome_var)

print("\nAll parameters tunning:\n")
param_test1 = {
 'max_depth': np.arange(3,10,2),        #Optimum value comes out as "7"
 #'max_depth': [6, 7, 8],             #Checking for smaller interval around optimum value
 'min_child_weight': np.arange(1,6,2),   #Optimum value comes out as "3"
 #'min_child_weight': [2, 3, 4]      #Checking for smaller interval around optimum value
 'subsample':[i/10.0 for i in range(1,5)],
 'colsample_bytree':[i/10.0 for i in range(1,5)],
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch1 = GridSearchCV(estimator = XGBClassifier(objective= 'binary:logistic', nthread=4, seed=27), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
gsearch1.fit(train[predictor_var],train[outcome_var])
#print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
#print(gsearch1.cv_results_)
"""
Optimum parameters:
    'subsample': 0.2
    'colsample_bytree': 0.2, 
    'gamma': 0.1, 
    'max_depth': 5, 
    'min_child_weight': 3, 
    'reg_alpha': 1, 
"""

print("\nXGBoost classifier simultaneous tunning(xgb5)")
xgb5 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=3,
 gamma=0.1,
 subsample=0.2,
 colsample_bytree=0.2,
 reg_alpha=1,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb3, train, predictor_var, outcome_var)
"""
XGBoost classifier simultaneous tunning(xgb5)

Model Report
Accuracy : 0.8306
AUC Score (Train): 0.822139
"""

sample = pd.read_csv("/media/piyush/New Volume/Data Science/Loan Prediction/data/sample.csv")
sample.Loan_ID = pd.read_csv("/media/piyush/New Volume/Data Science/Loan Prediction/data/test.csv").Loan_ID
s = gsearch1.predict(test)
sample.Loan_Status = s

sample.Loan_Status[sample.Loan_Status==1] = 'Y'
sample.Loan_Status[sample.Loan_Status==0] = 'N'

sample.to_csv("/media/piyush/New Volume/Data Science/Loan Prediction/data/submission1.csv", index=False, header=True)
