# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 09:56:01 2018

@author: p.soni
"""

import numpy as np
import pandas as pd

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

from datetime import datetime

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    """
    Check stationarity:
    Plotting Rolling Statistics: We can plot the moving average or moving variance 
    and see if it varies with time. By moving average/variance I mean that at any 
    instant ‘t’, we’ll take the average/variance of the last year, i.e. last 12 months. 
    But again this is more of a visual technique.
    
    Dickey-Fuller Test: This is one of the statistical tests for checking stationarity. 
    Here the null hypothesis is that the TS is non-stationary. The test results comprise 
    of a Test Statistic and some Critical Values for difference confidence levels. 
    If the ‘Test Statistic’ is less than the ‘Critical Value’, we can reject the null 
    hypothesis and say that the series is stationary. Refer this article for details.
    """
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)
    
    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

rcParams['figure.figsize'] = 15, 6

data = pd.read_csv('AirPassengers.csv')

"""
parse_dates: This specifies the column which contains the date-time information.
             The column name is ‘Month’.
index_col: A key idea behind using Pandas for TS data is that the index has to
             be the variable depicting date-time information. So this argument 
             tells pandas to use the ‘Month’ column as index.
date_parser: This specifies a function which converts an input string into datetime 
             variable. Be default Pandas reads data in format ‘YYYY-MM-DD HH:MM:SS’.  
             If the data is not in this format, the format has to be manually defined. 
             Something similar to the dataparse function defined here can be used for 
             this purpose.
"""
print(data.head())
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
data = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month',date_parser=dateparse)

ts = data['#Passengers']
          
#ts[datetime(1949,1,1)]

plt.plot(ts)



test_stationarity(ts)

"""
Results of Dickey-Fuller Test:
Test Statistic                   0.815369
p-value                          0.991880
#Lags Used                      13.000000
Number of Observations Used    130.000000
Critical Value (1%)             -3.481682
Critical Value (5%)             -2.884042
Critical Value (10%)            -2.578770

Though the variation in standard deviation is small, mean is clearly increasing
 with time and this is not a stationary series. Also, the test statistic is way
 more than the critical values. Note that the signed values should be compared 
 and not the absolute values.
"""
"""
There are 2 major reasons behind non-stationaruty of a TS:
1. Trend – varying mean over time. For eg, in this case we saw that on average,
 the number of passengers was growing over time.
2. Seasonality – variations at specific time-frames. eg people might have a 
tendency to buy cars in a particular month because of pay increment or festivals.
"""

ts_log = np.log(ts)
plt.plot(ts_log)

moving_avg = pd.rolling_mean(ts_log,12)
plt.plot(ts_log)
plt.plot(moving_avg, color='red')

ts_log_moving_avg_diff = ts_log - moving_avg
#ts_log_moving_avg_diff.head(12)

ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)

expwighted_avg = pd.ewma(ts_log, halflife=12)
plt.plot(ts_log)
plt.plot(expwighted_avg, color='red')

ts_log_ewma_diff = ts_log - expwighted_avg
test_stationarity(ts_log_ewma_diff)

"""
Eliminating Trend and Seasonality
The simple trend reduction techniques discussed before don’t work in all cases, 
particularly the ones with high seasonality. Lets discuss two ways of removing 
trend and seasonality:

Differencing – taking the differece with a particular time lag
Decomposition – modeling both trend and seasonality and removing them from the model.
"""

## Differencing

ts_log_diff = ts_log - ts_log.shift()  ## first order difference
plt.plot(ts_log_diff)

ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)

## Decomposing 

decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

# Drawback of decompose is that it is not intuitive to convert the residual to 
# original future values
ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)
"""
Results of Dickey-Fuller Test:
Test Statistic                -6.332387e+00
p-value                        2.885059e-08
#Lags Used                     9.000000e+00
Number of Observations Used    1.220000e+02
Critical Value (1%)           -3.485122e+00
Critical Value (5%)           -2.885538e+00
Critical Value (10%)          -2.579569e+00

Test statistics is significantly lower than 1%.
"""
print(ts_log_diff.index)

ts_log_diff.to_csv("ts_log.csv")


