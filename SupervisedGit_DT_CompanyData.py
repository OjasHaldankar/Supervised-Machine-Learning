#!/usr/bin/env python
# coding: utf-8

# # `Decision Tree Classifier : [C5.0]`

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split     
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree                                  #For visualisation of the tree
from sklearn.metrics import classification_report

# Importing Dataset
df = pd.read_csv("Company_Data.csv")
df.head(10)

df.describe()

# Converting Categorical data to numeric
df = pd.get_dummies(df,columns=['ShelveLoc','Urban','US'], drop_first=True)
df

# Dividing Target Variable i.e. 'Sales' into buckets [0 to 9] --> Poor Sales & [10 & above] --> Good Sales
df["Sale"] = pd.cut(df["Sales"], bins = [0,10,18], labels = ["Poor Sales", "Good Sales"])
df['Target'] = le.fit_transform(df['Sale'])
df

# Dropping columns not required for analysis
df = df.drop(columns= ['Sale'])
df
df['Target'].value_counts()

import seaborn as sns
sns.countplot(x = 'Target', data = df)

# Defining Feature & Target Variables
X = df.iloc[:, 1:12]
Y = df['Target']

#Dividing data into Training & Testing sections
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state= 1)

#Initializing Decision Tree Classifier
DT = DecisionTreeClassifier(criterion = 'gini', max_depth=3)
DT.fit(X_train,Y_train) 

#plot the Decision Tree
tree.plot_tree(DT);
fn=['Comp Price', 'Income', 'Advertising', 'Population', 'Price', 'Age', 'Education', 'ShelveLoc_Good', 'ShelveLoc_Medium', 'Urban_Yes', 'US_Yes']
cn=['Poor Sales', 'Good Sales']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (5,5), dpi=300)
tree.plot_tree(DT,
               feature_names = fn, 
               class_names=cn,
               filled = True);  

feature_importance = pd.Series(DT.feature_importances_, index=fn).sort_values(ascending=False) 
feature_importance


#Predicting on test data
preds = DT.predict(X_test)     # predicting on test data set 
pd.Series(preds).value_counts() 
preds


# Accuracy of the model
np.mean(preds==Y_test)

