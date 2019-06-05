#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:12:04 2019

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
y = dataset.iloc[:, y_pos].values

# fitting Decision Tree Regression Model to the dataset
from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(random_state = 0).fit(X, y)

# predicting a new result
y_pred = reg.predict([[6.5]])

# visualizing Decision Tree Regression results
plt.scatter(X, y, color='r')
plt.plot(X, reg.predict(X), color='b')
plt.title("Truth or bluff")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# visualizing Decision Tree Regression results for higher resolution
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='r')
plt.plot(X_grid, reg.predict(X_grid), color='b')
plt.title("Truth or bluff")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()