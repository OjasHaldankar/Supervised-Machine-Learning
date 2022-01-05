#!/usr/bin/env python
# coding: utf-8

# # `Classification ML Algorithm  :  Text Classification`
# 
# .................................................................
# 
# **~ Naive Bayes Classifier**
# 
# **~ AdaBoost Classifier**
# 
# **~ Random Forest Classifier**
# 
# **~ Logistic Regression**


# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

# Importing the dataset
df = pd.read_csv('Check.csv')
df

# Information About Dataset : The dataset contains profile vs skillset information for certain profiles which was pre-classified / labelled based on EDA.
df['Domain'].value_counts()

#Defining Target Variable
# {'ReactJS Developer' : 0, 'Peoplesoft Developer' : 1, 'Workday Developer' : 2, 'SQL Developer' : 3}
sns.countplot(x= 'Domain', data= df)

# Defining Feature (Independent Variable)
x = df['Skills']
cv = CountVectorizer()
X = cv.fit_transform(x)
# Defining Target Variable
Y = df['Domain']

# Splitting the dataset into Training & Testing Data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state= 10)


# ### `Model [1]  :  Naive Bayes Classifier` 

#Initializing Naive Bayes Model
from sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB()
NB.fit(X_train, Y_train)

# Calculating Training Accuracy
Y_pred_train_NB = NB.predict(X_train)
Train_acc_NB = accuracy_score(Y_pred_train_NB, Y_train)*100
Train_acc_NB

Y_pred_test_NB = NB.predict(X_test)
Y_pred_test_NB

# Calculating Testing Data Accuracy
Test_acc_NB = accuracy_score(Y_pred_test_NB, Y_test)*100
Test_acc_NB

# Classification Report
print(classification_report(Y_test, Y_pred_test_NB))


# ### `Model [2]  :  Random Forest Classifier (Ensemble Technique)` 

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
# Initializing the RF Classifier
num_trees = 150
max_features = 1 
RF = RandomForestClassifier(n_estimators= num_trees, max_features= max_features)

RF.fit(X_train, Y_train)

# Calculating Training Accuracy
Y_pred_train_RF = RF.predict(X_train)
Train_acc_RF = accuracy_score(Y_pred_train_RF, Y_train)*100
Train_acc_RF

# Calculating Testing Data Accuracy
Y_pred_test_RF = RF.predict(X_test)
Test_acc_RF = accuracy_score(Y_pred_test_RF, Y_test)*100
Test_acc_RF

print(classification_report(Y_test, Y_pred_test_RF))


# ### `Model [3]  :  AdaBoost Classifier` 

from sklearn.ensemble import AdaBoostClassifier
# Initializing the AdaBoost Classifier
num_trees = 150
seed = 5 
AB = AdaBoostClassifier(n_estimators=num_trees, random_state = seed) 
AB.fit(X_train, Y_train)

# Calculating Training Accuracy
Y_pred_train_AB = AB.predict(X_train)
Train_acc_AB = accuracy_score(Y_pred_train_AB, Y_train)*100
Train_acc_AB

# Calculating Testing Data Accuracy
Y_pred_test_AB = AB.predict(X_test)
Test_acc_AB = accuracy_score(Y_pred_test_AB, Y_test)*100
Test_acc_AB

print(classification_report(Y_test, Y_pred_test_AB))


# ### `Model [4]  :  Logistic Regression Classifier` 

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()
lm.fit(X_train, Y_train)

# Calculating Training Accuracy
Y_pred_train_LR = lm.predict(X_train)
Train_acc_LR = accuracy_score(Y_pred_train_LR, Y_train)*100
Train_acc_LR

# Calculating Testing Data Accuracy
Y_pred_test_LR = lm.predict(X_test)
Test_acc_LR = accuracy_score(Y_pred_test_LR, Y_test)*100
Test_acc_LR

print(classification_report(Y_test, Y_pred_test_LR))


Result = {'ML Algorithm':['Naive Bayes', 'Random Forest', 'AdaBoost', 'Logistic Regression' ],
    'Train Accuracy':[Train_acc_NB, Train_acc_RF, Train_acc_AB, Train_acc_LR]
    ,'Test Accuracy':[Test_acc_NB, Test_acc_RF, Test_acc_AB, Test_acc_LR]}

Comp_table = pd.DataFrame(Result)
Comp_table

plt.figure(figsize =(20, 6))
plt.plot(Comp_table['ML Algorithm'], Comp_table['Train Accuracy'])
plt.plot(Comp_table['ML Algorithm'], Comp_table['Test Accuracy'])
plt.title('Model Accuracies',fontdict={'fontsize': 25,'fontweight' : 15,'color' : 'g'})
plt.xlabel('ML Algorithm')
plt.ylabel('Accuracy')
plt.legend(['Train Acc', 'Test Acc'], loc='lower right')
plt.grid()
plt.show();

