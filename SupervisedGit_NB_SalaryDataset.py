#!/usr/bin/env python
# coding: utf-8

# # `Supervised ML  :  Naive Bayes Classifier`


# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Importing the training data
train_df = pd.read_csv("SalaryData_Train.csv")
train_df.head()

train_df.shape

# Importing testing data
test_df = pd.read_csv("SalaryData_Test.csv")
test_df.head()

test_df.shape

# Concatinating the training & testing datasets vertically to obtain final modelling dataset
df = pd.concat([train_df, test_df])
df.shape

df['Salary'].value_counts()

# Plotiing target column
sns.countplot( x= 'Salary', data= df)

# Label Encoding categorical columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['workclass_le'] = le.fit_transform(df['workclass'])
df['education_le'] = le.fit_transform(df['education'])
df['maritalstatus_le'] = le.fit_transform(df['maritalstatus'])
df['occupation_le'] = le.fit_transform(df['occupation'])
df['relationship_le'] = le.fit_transform(df['relationship'])
df['race_le'] = le.fit_transform(df['race'])
df['sex_le'] = le.fit_transform(df['sex'])
df['native_le'] = le.fit_transform(df['native'])

# Dropping categorical columns 
df = df.drop(['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex', 'native'], axis= 1)
df.head(10)

# Encoding / getting dummy values for 'Salary' column
df['Target'] = pd.get_dummies(df['Salary'], drop_first= True)
df.head(10)

# Dropping 'Salary' column from df dataset
df = df.drop(['Salary'], axis=1)
df.head()


# Defining the feature & target variable
X = df.iloc[:, :13]
Y = df['Target']

# Dividing the dataset into training & testing data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state= 10)

# Initialize Naive Bayes Classifier
nb = MultinomialNB()
nb.fit(X_train, Y_train)

# Calculating training accuracy
Y_pred_train = nb.predict(X_train)
Train_acc = print(accuracy_score(Y_train, Y_pred_train)) 
Train_acc

# Calculating training accuracy
Y_pred_test = nb.predict(X_test)
Test_acc = print(accuracy_score(Y_test, Y_pred_test)) 
Test_acc


print(classification_report(Y_test, Y_pred_test))

