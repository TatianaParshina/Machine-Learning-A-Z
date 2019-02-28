# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 16:09:10 2018

@author: Таня
"""

# XGBoost
# Part 1 - Data Preprocessing
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_country = LabelEncoder()
X[:, 1] = labelencoder_X_country.fit_transform(X[:, 1])
labelencoder_X_cgender = LabelEncoder()
X[:, 2] = labelencoder_X_cgender.fit_transform(X[:, 2])
# only for country
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# avoid dummy variable trap
X = X[:, 1:]

#splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

#Predicting the Test set results
y_pred = classifier.predict(X_test)

#Making the Confusion Matrix
#function
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, 
                             X = X_train, y = y_train, 
                             cv = 10)
accuracies.mean()
accuracies.std()


































