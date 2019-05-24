#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:28:29 2019

@author: cleandersonlins
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
PATH = '/home/cleandersonlins/Documents/Estudos/ML/Machine_Learning/Part_1_Data_Preprocessing
'
dataset = pd.read_csv(PATH + "Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# taking care of missing data
# from sklearn.preprocessing import Imputer -> Deprecated
from sklearn.impute import SimpleImputer
#imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = SimpleImputer()
imputer = imputer.fit(X[:, 1:3])

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

labelEncoder_Y = LabelEncoder()
Y = labelEncoder_Y.fit_transform(Y)
