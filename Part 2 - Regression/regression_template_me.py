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
y = dataset.iloc[:, y_pos].values

# fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=300, random_state=0).fit(X, y)

# predicting a new result
y_pred = reg.predict([[6.5]])

# visualizing Random Forest Regression results for higher resolution
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='r')
plt.plot(X_grid, reg.predict(X_grid), color='b')
plt.title("Truth or bluff")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()














