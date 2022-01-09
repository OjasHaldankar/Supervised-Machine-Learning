#!/usr/bin/env python
# coding: utf-8


# # `Supervised ML  :  Support Vector Classifier`


# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.svm import SVC                                 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

# Importing the datasets
train = pd.read_csv("SalaryData_Train(1).csv")
train.head()

train.shape

test = pd.read_csv("SalaryData_Test(1).csv")
test.head()

test.shape


# Combining to get modelling dataset
df = pd.concat([train, test])
df.shape

# Checking for null values
df.isna().sum()

# Encoding the categorical variables / features
from sklearn.preprocessing import LabelEncoder
df = df.apply(LabelEncoder().fit_transform)
df

df['Salary'].value_counts()

# Plotting the output class
import seaborn as sns
sns.countplot(x = 'Salary', data = df)


# Defining Feature & Target Variables
X = df.iloc[:, :13]
Y = df.iloc[:, 13]

# Training & Testing Data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.3, random_state= 0 )

# Initialize Support Vector Classifier
test = SVC()


# Radial Kernel
test = SVC(C= 15, gamma = 0.5)
test.fit(x_train , y_train)
y_pred_rad = test.predict(x_test)
acc_rad = accuracy_score(y_test, y_pred_rad) * 100
print("Accuracy of Radial Kernel =", acc_rad)
confusion_matrix(y_test, y_pred_rad)


# Polynomial Kernel
test = SVC(kernel= "poly")
test.fit(x_train , y_train)
y_pred_rad = test.predict(x_test)
acc_poly = accuracy_score(y_test, y_pred_rad) * 100
print("Accuracy of Polynomial Kernel =", acc_rad)
confusion_matrix(y_test, y_pred_rad)


# Linear Kernel
test = SVC(kernel= "linear")
test.fit(x_train , y_train)
y_pred_rad = test.predict(x_test)
acc_poly = accuracy_score(y_test, y_pred_rad) * 100
print("Accuracy of Linear Kernel =", acc_rad)
confusion_matrix(y_test, y_pred_rad)

