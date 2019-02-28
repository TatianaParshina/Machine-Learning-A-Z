# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 08:04:56 2018

@author: Таня
"""

#Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#Taking careof missing data
#import class
from sklearn.preprocessing import Imputer
#create object of Imputer class
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
#replace values in array with mean values
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Encoding categorical data (country) 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
labelencoder_X = LabelEncoder()
#encode country  column - 0,1,2 - it is incorrect for country because France is not bigger then Germany
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
#dummy encoding country
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

#encoder purchased
labelencoder_y = LabelEncoder()
#encode purchased  column - 0,1 
y = labelencoder_y.fit_transform(y)



#splitting the dataset into the Training set and Test set
#import library
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#Feature scaling age 20-60 salary 20000 - 90 000. they should have same scale
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

















