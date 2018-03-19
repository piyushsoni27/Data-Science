# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 23:36:25 2017

@author: Piyush
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency

from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report, precision_recall_curve, confusion_matrix

#split into train and test
from sklearn.model_selection import train_test_split

# Feature Scaling
from sklearn.preprocessing import StandardScaler

## Reading Data
train = pd.read_csv("/media/piyush/New Volume/Data Science/Predict the criminals Hackerearth/Data/criminal_train.csv")
test_orig = pd.read_csv("/media/piyush/New Volume/Data Science/Predict the criminals Hackerearth/Data/criminal_test.csv")

train_len = train.shape[0]

total = pd.concat([train, test_orig])
total.drop("PERID", axis=1, inplace=True)
ct = pd.crosstab(train.AIIND102[train.AIIND102 != -1], train.Criminal)

"""
Correlation B/W features:

1. B/W numerical features:
--> Pearson's Correlation:
        DataFrame.corr(method='pearson', min_periods=1    

2. B/W Categorical features:
--> Chi-Square Test:
        Calculating correlation b/w categorical features.
        I/P : Crosstab of 2 cat. vars.  [i.e. pd.crosstab()]

        scipy.stats.chi2_contingency(ct)

        p-value is the measure of correlation

3. B/W Numerical and Categorical feature
--> One-way ANOVA test
        scipy.stats.f_oneway(*args)
        
        p-value is the measure of correlation
"""
chi2_contingency(ct)

train=total[:train_len]
test=total[train_len:]

test.drop("Criminal", axis=1, inplace=True)

predictors = test.columns
target = "Criminal"

x_train, x_test, y_train, y_test = train_test_split(train[predictors], train[target],test_size = .33, random_state = 1)

"""
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

test = sc.transform(test)
"""
logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)
logreg_accy = round(accuracy_score(y_pred,y_test), 3)
print (logreg_accy)

test_pred = logreg.predict(test)
file_path="/media/piyush/New Volume/Data Science/Predict the criminals Hackerearth/Submissions/baseline.csv"

    
base2 = test_orig[['PERID']]
base2["Criminal"] = pd.Series(test_pred, dtype=int)

base2.to_csv(file_path, index=False)