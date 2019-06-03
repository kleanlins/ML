#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:51:12 2019

@author: cleandersonlins
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# input variables
y_pos = -1
filename = "Position_Salaries.csv"

# importing the dataset
dataset = pd.read_csv(filename)
X = dataset.iloc[:, 1:y_pos].values
Y = dataset.iloc[:, y_pos].values

# fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression().fit(X, Y)

# fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures 
poly_reg = PolynomialFeatures(degree = 4).fit(X)
X_poly = poly_reg.transform(X)

poly_reg.fit(X_poly, Y)

lin_reg_2 = LinearRegression().fit(X_poly, Y)

# visualizing Linear Regression results
plt.scatter(X, Y, color='r')
plt.plot(X, lin_reg.predict(X), color='b')
plt.title("Truth or bluff")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# visualizing Polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color='r')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='b')
plt.title("Truth or bluff")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# predicting a new result with Linear Regression
lin_reg.predict([[6.5]]) # [[]] -> because it expects a 2D array

# predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))












