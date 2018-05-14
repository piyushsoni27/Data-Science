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
