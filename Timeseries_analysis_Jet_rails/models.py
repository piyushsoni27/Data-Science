# -*- coding: utf-8 -*-
"""
Created on Mon May 14 16:18:34 2018

@author: p.soni
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm


train = pd.read_csv("Data/train_new.csv", index_col="Datetime")
test = pd.read_csv("Data/test_new.csv", index_col="Datetime")
submission = pd.read_csv("submission.csv")

train.index = pd.to_datetime(train.index, format="%Y-%m-%d")
test.index = pd.to_datetime(test.index, format = "%Y-%m-%d")

train_set = train.loc['2012-08-25' : '2014-06-24']
valid_set = train.loc['2014-06-25' : '2014-09-25']

"""
train_set.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='train')
valid_set.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='valid')
plt.xlabel("Datetime")
plt.ylabel("Passenger count")
plt.legend(loc='best')
plt.show()
"""

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

rms_naive = sqrt(mean_squared_error(valid_set.Count, y_hat.Naive))

print("RMS of Naive : %f" % rms_naive)

"""
Moving Average (Rolling Mean):
    For last 10, 20 and 50 observations
"""

y_hat_avg = valid_set.copy()
y_hat_avg["moving_avg"] = train_set.Count.rolling(10).mean().iloc[-10]
plt.figure(figsize=(15,5)) 
plt.plot(train_set['Count'], label='Train')
plt.plot(valid_set['Count'], label='Valid')
plt.plot(y_hat_avg['moving_avg'], label='Moving Average Forecast using 10 observations')
plt.legend(loc='best')
plt.show()

rms_moving_avg_10 = sqrt(mean_squared_error(valid_set.Count, y_hat_avg.moving_avg))

y_hat_avg = valid_set.copy()
y_hat_avg["moving_avg"] = train_set.Count.rolling(20).mean().iloc[-10]
plt.figure(figsize=(15,5)) 
plt.plot(train_set['Count'], label='Train')
plt.plot(valid_set['Count'], label='Valid')
plt.plot(y_hat_avg['moving_avg'], label='Moving Average Forecast using 10 observations')
plt.legend(loc='best')
plt.show()

rms_moving_avg_20 = sqrt(mean_squared_error(valid_set.Count, y_hat_avg.moving_avg))

y_hat_avg = valid_set.copy()
y_hat_avg["moving_avg"] = train_set.Count.rolling(50).mean().iloc[-10]
plt.figure(figsize=(15,5)) 
plt.plot(train_set['Count'], label='Train')
plt.plot(valid_set['Count'], label='Valid')
plt.plot(y_hat_avg['moving_avg'], label='Moving Average Forecast using 10 observations')
plt.legend(loc='best')
plt.show()

rms_moving_avg_50 = sqrt(mean_squared_error(valid_set.Count, y_hat_avg.moving_avg))

print("RMS for Moving avg for last 10 itr : %f" % rms_moving_avg_10)
print("RMS for Moving avg for last 20 itr : %f" % rms_moving_avg_20)
print("RMS for Moving avg for last 50 itr : %f" % rms_moving_avg_50)


"""
Exponential Smoothing
1. In this technique, we assign larger weights to more recent observations than to
    observations from the distant past.
2. The weights decrease exponentially as observations come from further in the past, 
    the smallest weights are associated with the oldest observations.
"""
y_hat = valid_set.copy()
model = SimpleExpSmoothing(np.asarray(train_set.Count)).fit(smoothing_level = 0.6, optimized=False)
y_hat['SES'] = model.forecast(len(valid_set))
plt.figure(figsize=(16,8))
plt.plot(train_set['Count'], label='Train')
plt.plot(valid_set['Count'], label='Valid')
plt.plot(y_hat['SES'], label='SES')
plt.legend(loc='best')
plt.show()

rms_SES = sqrt(mean_squared_error(valid_set.Count, y_hat.SES))

print("RMS For SES : %f" % rms_SES)

"""
Holt Linear trend Model
1. It is an extension of exponential smoothing, differene is that it remembers trend of the time series.

-> Trend: which shows the trend in the time series, i.e., increasing or decreasing behaviour of the time series.
-> Seasonal: which tells us about the seasonality in the time series.
-> Residual: which is obtained by removing any trend or seasonality in the time series.
"""

sm.tsa.seasonal_decompose(train_set.Count).plot()

y_hat_avg = valid_set.copy()
fit1 = Holt(np.asarray(train_set['Count'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1)
y_hat_avg['Holt_linear'] = fit1.forecast(len(valid_set))

plt.figure(figsize=(13,8))
plt.plot(train_set['Count'], label='Train')
plt.plot(valid_set['Count'], label='Valid')
plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear')
plt.legend(loc='best')
plt.show()

## Daily predictions: must be converted to hourly predictions
test.prediction = fit1.forecast(len(test))
rms_holt = sqrt(mean_squared_error(valid_set.Count, y_hat_avg.Holt_linear))

print("RMS For holt : %f" % rms_holt)
