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
    
    imputer = Imputer(missing_values ='NaN', strategy='most_frequent', axis=0)
    dataset["Credit_History"] = imputer.fit_transform(dataset["Credit_History"].reshape(-1,1))
    
     #impute missing values
    
    imputer = Imputer(missing_values ="NaN", strategy='mean', axis=0)
    dataset["LoanAmount"] = imputer.fit_transform(dataset["LoanAmount"].reshape(-1,1))
    
    #impute missing values
    
    imputer = Imputer(missing_values ='NaN', strategy='most_frequent', axis=0)
    dataset["Loan_Amount_Term"] = imputer.fit_transform(dataset["Loan_Amount_Term"].reshape(-1,1))
    
    #impute missing values
    
    imputer = Imputer(missing_values ='NaN', strategy='most_frequent', axis=0)
    dataset["Dependents"] = imputer.fit_transform(dataset["Dependents"].reshape(-1,1))


  
train=pd.concat([train.drop("Property_Area",axis=1), pd.get_dummies(train["Property_Area"], drop_first=True)], axis= 1)
test=pd.concat([test.drop("Property_Area",axis=1), pd.get_dummies(test["Property_Area"], drop_first=True)], axis= 1)  


train['Loan_Status'] = train['Loan_Status'].replace('Y', '1')
train['Loan_Status'] = train['Loan_Status'].replace('N', '0')
        
train=train.drop("Loan_ID",axis=1)
testIds=test["Loan_ID"]
test=test.drop("Loan_ID",axis=1)

X = train.loc[:, train.columns != "Loan_Status"].values
y = train['Loan_Status'].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Feature Scaling
#Escalado con la misma escala train a test set
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
test = sc.transform(test)

"""
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
"""

#RF parece ser bueno y le pega más pero SVM tiró mejor en Kaggle
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
classifier.fit(X_train, y_train)
print(classifier.score(X_train, y_train))

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#Predecir
y_pred_test = classifier.predict(test)
y_pred_test[y_pred_test == '1'] = 'Y'
y_pred_test[y_pred_test == '0'] = 'N'


"""-----------------Armar csv----------------"""
preds = pd.DataFrame({'Loan_ID':testIds, 'Loan_Status':y_pred_test})
preds.to_csv("preds.csv", index=False)












