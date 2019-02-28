# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 13:30:44 2018

@author: Таня
"""

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoidint the Dummy Variable Trap
#all columns from 1 to the end
X = X[:,1:]

#splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature scaling 
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)

#Building optimal model using Backword Elimination
import statsmodels.formula.api as sm
#x0=1
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
#only will have IV which have impact - statistically significant
X_opt = X[:, [0,1,2,3,4,5]]
#ordinary least squared algorithm
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#step 3
regressor_OLS.summary()

X_opt = X[:, [0,1,3,4,5]]
#ordinary least squared algorithm
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#step 3
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5]]
#ordinary least squared algorithm
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#step 3
regressor_OLS.summary()

X_opt = X[:, [0,3,5]]
#ordinary least squared algorithm
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#step 3
regressor_OLS.summary()

X_opt = X[:, [0,3]]
#ordinary least squared algorithm
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#step 3
regressor_OLS.summary()


#automatic bacword elimination
import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    #all columns - all independent variables
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            #remove independent variable
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)


#Backward Elimination with p-values and Adjusted R Squared:

import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
























