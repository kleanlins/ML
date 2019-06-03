#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 14:59:25 2019

@author: cleandersonlins
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# input variables
y_pos = -1
filename = "50_Startups.csv"

# importing the dataset
dataset = pd.read_csv(filename)
X = dataset.iloc[:, :y_pos].values
Y = dataset.iloc[:, y_pos].values

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('state', OneHotEncoder(), [-1])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

# avoiding the Dummy Variable Trap
X = X[:, 1:]

# splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# fitting Multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)

y_pred = reg.predict(X_test)

# building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(np.ones((50,1), dtype=np.float), X, axis=1)

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
reg_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
reg_OLS.summary()

# removed the highest P value
X_opt = X[:, [0, 1, 3, 4, 5]]
reg_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
reg_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
reg_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
reg_OLS.summary()

X_opt = X[:, [0, 3, 5]]
reg_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
reg_OLS.summary()

X_opt = X[:, [0, 3]]
reg_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
reg_OLS.summary()






