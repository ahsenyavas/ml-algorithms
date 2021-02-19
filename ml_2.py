# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 16:56:01 2021

@author: Ahsen Yavas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('sample_data.csv')

#print(data)
#print(data.head())
#print(data.describe())

data = data.drop_duplicates()
#print(data.describe())

#print(data.isnull().sum())

# Dropping categorical data rows with missing values
data.dropna(how='any', subset=['Country', 'Purchased'], inplace=True)

# Splitting dataset into independent & dependent variable
X = data.iloc[:, 0:3].values
y = data.iloc[:, 3:].values

#print(X)
#print(y)

# replacing the missing values in the age & salary column with the mean
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

#print(X)

# Handling Categorical Data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('enconder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Encoding the target variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

# Splitting Dataset into Training and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 4:] = sc.fit_transform(X_train[:, 4:])

X_test[:, 4:] = sc.transform(X_test[:, 4:])
print(X_test)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier().fit(X_train, y_train)

y_pred2 = knn_model.predict(X_test)

print(accuracy_score(y_test, y_pred))







