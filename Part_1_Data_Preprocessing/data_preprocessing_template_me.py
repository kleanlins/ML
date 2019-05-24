#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:28:29 2019

@author: cleandersonlins
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# input variables
y_pos = -1
path = "/home/cleandersonlins/Documents/Estudos/ML/Machine_Learning/Part_1_Data_Preprocessing/"
filename = "Data.csv"

# importing the dataset
dataset = pd.read_csv(path + filename)
X = dataset.iloc[:, :y_pos].values
Y = dataset.iloc[:, y_pos].values

# splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# feature scaling
"""
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler().fit(X_train)
X_train = sc_X.transform(X_train)
X_test = sc_X.transform(X_test)
"""
