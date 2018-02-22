#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 22:24:31 2018

@author: piyush
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv("/media/piyush/New Volume/Data Science/Loan Prediction/data//train.csv")
test = pd.read_csv("/media/piyush/New Volume/Data Science/Loan Prediction/data//test.csv")

train_len = train.shape[0]

train.Loan_Status[train.Loan_Status == 'Y'] = 1
train.Loan_Status[train.Loan_Status == 'N'] = 0

train.Credit_History.fillna(train.Loan_Status, inplace = True)


total = pd.concat([train,test])
## Check no. of missing values
train.isnull().sum()
test.isnull().sum()

## Distribution analysis
total.ApplicantIncome.hist(bins = 50)
total.boxplot(column="ApplicantIncome")
total.boxplot(column="ApplicantIncome", by="Education")

total.LoanAmount.hist(bins = 50)

##treating skewness
total.LoanAmount = np.log1p(train.LoanAmount)

## Categorical variables
train.Credit_History.value_counts()
train.pivot_table(values="Loan_Status", index="Credit_History", aggfunc=lambda x:x.map({'Y':1, 'N':0}).mean())

temp = pd.crosstab(train['Credit_History'], train['Loan_Status'])
temp.plot(kind="bar", stacked = True, grid = False)

temp1 = pd.crosstab([train['Credit_History'],train.Gender], train['Loan_Status'])
temp1.plot(kind="bar", stacked = True, grid = False)

## Filling missing Values
total.Self_Employed.value_counts()
total.Self_Employed.fillna('No',inplace = True)

table = total.pivot_table(values="LoanAmount", index="Self_Employed", columns="Education", aggfunc=np.median)

def fill_amount(x):
    return table.loc[x["Self_Employed"], x["Education"]]

total.reset_index(inplace = True)
total.LoanAmount.fillna(total[total.LoanAmount.isnull()].apply(fill_amount,axis=1), inplace = True)

total["total_income"] = total.ApplicantIncome + total.CoapplicantIncome
total.total_income.hist(bins = 50)

## Remove skewness
total.total_income = np.log1p(total.total_income)

total.drop(["Loan_ID", "ApplicantIncome", "CoapplicantIncome"], inplace = True, axis =1)

total.Credit_History.value_counts()
total.Credit_History.fillna(1, inplace=True)

total.Married.fillna("Yes", inplace = True)
total.Dependents.fillna(0, inplace = True)
total.Gender.fillna("missing", inplace = True)
total.Loan_Amount_Term.fillna(360, inplace=True)

train_final = total[:train_len]
test_final = total[train_len:]

test_final.drop("Loan_Status", axis =1, inplace=True)

train_final.to_csv("/media/piyush/New Volume/Data Science/Loan Prediction/data//train_final.csv")
test_final.to_csv("/media/piyush/New Volume/Data Science/Loan Prediction/data//test_final.csv")
