# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 08:33:31 2018

@author: Таня
"""

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
#X will be considered as matrix, 2 is not included
X = dataset.iloc[:, 1:2].values
#y as vector
y = dataset.iloc[:, 2].values

#no splitting because of less data
#splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

#polynomial regression has scaling feature
#Feature scaling 
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#we need 2 models to compare results

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Fitting Polynomial Regression to the dataset
#include polynomial on top of linear regression
from sklearn.preprocessing import PolynomialFeatures
#transform X to new matrix - x, x^2, x^3
poly_reg = PolynomialFeatures (degree = 4)
X_poly = poly_reg.fit_transform(X)
#linear regression for new independent variables
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

#Visualising the Linear Regression results
#real observation
plt.scatter(X, y, color = 'red')
#predictions
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualising the Polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color = 'red')
#predictions
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Predicting a new result with Linear Regression
lin_reg.predict(6.5)

#Predicting a new result with Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform(6.5))
















