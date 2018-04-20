# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 20:31:17 2018

@author: p.soni
"""
import pandas as pd
#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf

ts_log_diff = pd.read_csv("ts_log_diff.csv", header=None, names = ["Month", '#Passengers' ], index_col=0, squeeze=True)
                                                                   
ts_log_diff.index=pd.to_datetime(ts_log_diff.index, format='%Y-%m')
