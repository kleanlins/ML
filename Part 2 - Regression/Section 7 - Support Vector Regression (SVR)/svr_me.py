#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 13:16:56 2019

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

# applying feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler().fit(X)
sc_Y = StandardScaler().fit(y.reshape(-1,1))

X = sc_X.transform(X)
y = sc_Y.transform(y.reshape(-1, 1))


# fitting Regression Model to the dataset
from sklearn.svm import SVR
reg = SVR(kernel = 'rbf').fit(X, y)

# predicting a new result
y_pred = sc_Y.inverse_transform(reg.predict(sc_X.transform([[6.5]])))

# visualizing Polynomial Regression results
plt.scatter(X, y, color='r')
plt.plot(X, reg.predict(X), color='b')
plt.title("Truth or bluff")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# visualizing Polynomial Regression results for higher resolution
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='r')
plt.plot(X_grid, reg.predict(X_grid), color='b')
plt.title("Truth or bluff")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
