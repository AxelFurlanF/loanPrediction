# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:40:51 2018

@author: axelf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
#datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
full_data = [train, test]




#data cleansing
for dataset in full_data:
    #categories to numbers
    
    labelencoder = LabelEncoder()
    labelencoder.fit(["Male","Female","nan"])
    dataset["Gender"] = labelencoder.transform(dataset["Gender"].astype(str))
    
    #impute missing values
    
    imputer = Imputer(missing_values =2, strategy='most_frequent', axis=0)
    dataset["Gender"] = imputer.fit_transform(dataset["Gender"].reshape(-1,1))
    
      #categories to numbers
    labelencoder.fit(["Yes","No","nan"])
    dataset["Married"] = labelencoder.transform(dataset["Married"].astype(str))
    
    #impute missing values
    
    imputer = Imputer(missing_values =2, strategy='most_frequent', axis=0)
    dataset["Married"] = imputer.fit_transform(dataset["Married"].reshape(-1,1))
    
    dataset['Dependents'] = dataset['Dependents'].replace('3+', '3')
    
    labelencoder.fit(["Graduate","Not Graduate"])
    dataset["Education"] = labelencoder.transform(dataset["Education"].astype(str))
    
    #categories to numbers
    labelencoder.fit(["Yes","No","nan"])
    dataset["Self_Employed"] = labelencoder.transform(dataset["Self_Employed"].astype(str))
    
    #impute missing values
    
    imputer = Imputer(missing_values =2, strategy='most_frequent', axis=0)
    dataset["Self_Employed"] = imputer.fit_transform(dataset["Self_Employed"].reshape(-1,1))
    
X = train.iloc[:, :-1].values
y = train.iloc[:, 12].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
test = sc.transform(test)




















