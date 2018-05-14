# -*- coding: utf-8 -*-
"""
Created on Mon May 14 10:17:13 2018

@author: p.soni
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("Data//train.csv")
test = pd.read_csv("Data//test.csv")

train_original = train.copy()
test_original = test.copy()

train.Datetime = pd.to_datetime(train.Datetime, format="%d-%m-%Y %H:%M")
test.Datetime = pd.to_datetime(test.Datetime, format = "%d-%m-%Y %H:%M")
test_original.Datetime = pd.to_datetime(test_original.Datetime, format = "%d-%m-%Y %H:%M")


for i in (train, test, test_original):
    i["year"] = i.Datetime.dt.year
    i["month"] = i.Datetime.dt.month
    i["day"] = i.Datetime.dt.day
    i["hour"] = i.Datetime.dt.hour
    
train["weekdays"] = train.Datetime.dt.dayofweek

train["workingday"] = train.weekdays.apply(lambda s: 1 if(s==5 or s==6) else 0)

train.index = train.Datetime

train.drop(["ID", "Datetime"], axis=1, inplace=True)

#plt.figure(figsize=(18,6))
#plt.plot(train.Count)

#Plot by year
#train.groupby("year")['Count'].mean().plot.bar()

train.groupby("hour")["Count"].mean().plot.bar()

train.groupby("workingday")["Count"].mean().plot.bar()

## passenger count is more in working days
train.groupby("weekdays")["Count"].mean().plot.bar()

hourly = train.resample('H').mean()
weekly = train.resample('W').mean()
monthly = train.resample('M').mean()
daily = train.resample('D').mean()

fig, axs= plt.subplots(4,1)

hourly.Count.plot(figsize=(15,8), ax=axs[0])
daily.Count.plot(figsize=(15,8), ax=axs[1])
weekly.Count.plot(figsize=(15,8), ax=axs[2])
monthly.Count.plot(figsize=(15,8), ax=axs[3])

plt.show()

test.index = test.Datetime
test.drop(["ID", "Datetime"], axis=1, inplace=True)


## Since daily TS is more stable than Houly thus converting to daily TS
test = test.resample("D").mean()
train = daily.copy()


