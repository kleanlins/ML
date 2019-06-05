#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:15:38 2019

@author: cleandersonlins
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# input variables
y_pos = -1
filename = "Social_Network_Ads.csv"

# importing the dataset
dataset = pd.read_csv(filename)
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, y_pos].values

# splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# applying scaler to our data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler().fit(X_train)
X_train = sc_X.transform(X_train)
X_test = sc_X.transform(X_test)

# fitting Logistic Regression to the training set
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state = 0).fit(X, y)