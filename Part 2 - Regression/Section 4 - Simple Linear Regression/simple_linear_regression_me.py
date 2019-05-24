#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 17:40:30 2019

@author: cleandersonlins
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# input variables
y_pos = -1
filename = "Salary_Data.csv"

# importing the dataset
dataset = pd.read_csv(filename, encoding='utf-8')
X = dataset.iloc[:, :y_pos].values
Y = dataset.iloc[:, y_pos].values

# splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

# feature scaling
"""
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler().fit(X_train)
X_train = sc_X.transform(X_train)
X_test = sc_X.transform(X_test)
"""

# fitting simple linear regression into the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression().fit(X_train, y_train)























