#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:28:29 2019

@author: cleandersonlins
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

y_pos = -1
filename = "Data.csv"

# importing the dataset
dataset = pd.read_csv(filename)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# taking care of missing data (by putting mean on the NaN values)
# from sklearn.preprocessing import Imputer -> Deprecated
from sklearn.impute import SimpleImputer
#imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = SimpleImputer()
imputer = imputer.fit(X[:, 1:3])

# imputer = SimpleImputer.fit(X[:, 1:3]) -> same thing as the upper lines

X[:, 1:3] = imputer.transform(X[:, 1:3])

# encoding categorical data
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_x = LabelEncoder()
# X[:, 0] = labelencoder_x.fit_transform(X[:, 0])


#one_hot_encoder = OneHotEncoder(categorical_features = [0]) -> Deprecated
#X = one_hot_encoder.fit_transform(X).toarray()
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('country', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

# makes no difference for the dependent variable to have a "ranking"
labelEncoder_Y = LabelEncoder()
Y = labelEncoder_Y.fit_transform(Y)

# splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# feature scaling

from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()

#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)

sc_X = StandardScaler().fit(X_train) -> works the same
X_train = sc_X.transform(X_train)
X_test = sc_X.transform(X_test)





















