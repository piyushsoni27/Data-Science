# -*- coding: utf-8 -*-
"""
Created on Mon May 14 16:18:34 2018

@author: p.soni
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train = pd.read_csv("Data/train_new.csv", index_col="Datetime")
test = pd.read_csv("Data/test_new.csv", index_col="Datetime")

train.index = pd.to_datetime(train.index, format="%Y-%m-%d")
test.index = pd.to_datetime(test.index, format = "%Y-%m-%d")

train_set = train.loc['2012-08-25' : '2014-06-24']
valid_set = train.loc['2014-06-25' : '2014-09-25']

train_set.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='train')
valid_set.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='valid')
plt.xlabel("Datetime")
plt.ylabel("Passenger count")
plt.legend(loc='best')
plt.show()


"""
Baseline model (Naive):
    In this we assume next expected output is equal to last observed point
"""

y_hat = valid_set.copy()

#last observed value:
y_hat["Naive"] = train_set.iloc[-1].Count

plt.figure(figsize=(12,8))
plt.plot(train_set.index, train_set['Count'], label='Train')
plt.plot(valid_set.index,valid_set['Count'], label='Valid')
plt.plot(y_hat.index,y_hat['Naive'], label='Naive Forecast')
plt.legend(loc='best')
plt.title("Naive Forecast")
plt.show()
