#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:45:29 2019

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

# fitting Regression Model to the dataset
# create you regressor here

# predicting a new result
y_pred = reg.predict([[6.5]])

# visualizing Polynomial Regression results
plt.scatter(X, Y, color='r')
plt.plot(X, reg.predict([[6.5]]), color='b')
plt.title("Truth or bluff")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# visualizing Polynomial Regression results for higher resolution
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color='r')
plt.plot(X_grid, reg.predict(X_grid), color='b')
plt.title("Truth or bluff")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()














