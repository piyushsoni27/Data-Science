# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 23:36:25 2017

@author: Piyush
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

## Reading Data
train = pd.read_csv("/media/piyush/New Volume/Data Science/Analytics Vidhya - Big Mart Sales III/Data/train.csv")
test = pd.read_csv("/media/piyush/New Volume/Data Science/Analytics Vidhya - Big Mart Sales III/Data/test.csv")

train_len = train.shape[0]

total = pd.concat([train, test])

## Processing Numerical Features
med_weight = total.groupby("Item_Type", as_index=False)["Item_Weight"].median()
med_weight.set_index("Item_Type", inplace = True)

total.loc[total.Item_Weight.isnull(),"Item_Weight"] = total.loc[total.Item_Weight.isnull(),"Item_Type"].apply(lambda x: med_weight.loc[str(x)])

mean_visibility = total.groupby("Outlet_Type", as_index=False)["Item_Visibility"].mean()
mean_visibility.set_index("Outlet_Type", inplace=True)

total.loc[total.Item_Visibility==0,"Item_Visibility"] = total.loc[total.Item_Visibility==0,"Outlet_Type"].apply(lambda x: mean_visibility.loc[str(x)])

# Divide MRP into groups so that machine can learn better
total['Item_MRP_Band'] = pd.qcut(total['Item_MRP'], 4)
total[['Item_MRP_Band', 'Item_Outlet_Sales']].groupby(['Item_MRP_Band'], as_index=False).mean().sort_values(by='Item_MRP_Band', ascending=True)

total.loc[total['Item_MRP'] <= 94.012, 'Item_MRP'] = 0
total.loc[(total['Item_MRP'] > 94.012) & (total['Item_MRP'] <= 142.247), 'Item_MRP'] = 1
total.loc[(total['Item_MRP'] > 142.247) & (total['Item_MRP'] <= 185.856), 'Item_MRP']   = 2
total.loc[ total['Item_MRP'] > 185.856, 'Item_MRP'] = 3
total['Item_MRP'] = total['Item_MRP'].astype(int)

total["Outlet_Establishment_Year_from_2018"] = 2018 - total.Outlet_Establishment_Year

## Processing Categorical Variables
total.Item_Fat_Content = total.Item_Fat_Content.apply(lambda s: str(s).lower())
total.Item_Fat_Content[total.Item_Fat_Content == "low fat"] = 'lf'
total.Item_Fat_Content[total.Item_Fat_Content == "regular"] = 'reg'

mode_outlet_size = total.groupby("Outlet_Type", as_index=False)["Outlet_Size"].agg(lambda s: s.value_counts().index[0])
mode_outlet_size.set_index("Outlet_Type", inplace=True)

total.loc[total.Outlet_Size.isnull(), "Outlet_Size"] = total.loc[total.Outlet_Size.isnull(), "Outlet_Type"].apply(lambda row: mode_outlet_size.loc[str(row)].Outlet_Size)

stop = stopwords.words("english")
total.Item_Type = total.Item_Type.apply(lambda s: " ".join([word for word in s.split() if word not in stop]))

# Extract initial Item Code from identifier(Remove numeric Values)
#total["Item_Code"] = total.Item_Identifier.str.replace('\d+','')

total["Item_Category"] = total.Item_Identifier.apply(lambda s: s[:2])

#total["Item_Code"] = total.Item_Identifier.apply(lambda s: s[2:])



cat_vars = ["Item_Fat_Content", "Outlet_Location_Type", "Outlet_Type", 
            "Outlet_Size", "Item_Category", "Item_Type"]

"""
le = LabelEncoder()

for i in cat_vars:
    total[i] = le.fit_transform(total[i])
"""

total.Item_Type.replace(["Breakfast","Hard Drinks","Seafood","Starchy Foods"],
                        "Others", inplace=True)


one_hot = pd.get_dummies(total[cat_vars])

total.drop(["Item_MRP_Band","Outlet_Establishment_Year", "Outlet_Identifier", 
            "Item_Identifier", "Item_Fat_Content", "Outlet_Location_Type", "Outlet_Type", 
            "Outlet_Size", "Item_Category", "Item_Type"], inplace = True, axis=1)

total = pd.concat([total, one_hot], axis=1)

train_final = total[:train_len]

test_final = total[train_len:]
test_final.drop("Item_Outlet_Sales", axis=1, inplace=True)

corr_train = train_final.corr()

#fig, ax = plt.subplots(figsize=(40,40))
#sns.heatmap(corr_train, annot=True, ax=ax)

train_final.to_csv("/media/piyush/New Volume/Data Science/Analytics Vidhya - Big Mart Sales III/Data/train_final.csv")
test_final.to_csv("/media/piyush/New Volume/Data Science/Analytics Vidhya - Big Mart Sales III/Data/test_final.csv")

