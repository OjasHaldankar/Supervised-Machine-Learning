#!/usr/bin/env python
# coding: utf-8

# # `Decision Tree Classifier` 

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split     
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree                                  
from sklearn.metrics import classification_report

# Importing Dataset
df = pd.read_csv("Fraud_check.csv")
df.head(10)
df.shape

df.describe()

# Features / Independent Variables : 'Undergrad' + 'Marital.Status' + 'City.Population' + 'Work.Experience' + 'Urban'
# Target Variable : 'Taxable.Income' ['10000 to 30000' : Risky] & ['30000 & above' : Good]

# Label Encoding Text Features
le = LabelEncoder()
df['Undergrad_le'] = le.fit_transform(df['Undergrad'])
df['Marital.Status_le'] = le.fit_transform(df['Marital.Status'])
df['Urban_le'] = le.fit_transform(df['Urban'])

#Creating Buckets for Taxable Income based on critical value of 30000
df["tax_income"] = pd.cut(df["Taxable.Income"], bins = [10000,30000,100000], labels = ["Risky", "Good"])
df['Target'] = le.fit_transform(df['tax_income'])
df.head(10)

# Dropping columns not required for analysis
df = df.drop(columns= ['Undergrad', 'Marital.Status', 'Taxable.Income', 'Urban', 'tax_income'])
df

df['Target'].value_counts()
sns.countplot(x = 'Target', data= df)

# Defining Feature & Target Variables
X = df.iloc[:, :5]
Y = df['Target']

#Dividing data into Training & Testing sections
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.3, random_state= 10)

#Initializing Decision Tree Classifier
DT = DecisionTreeClassifier(criterion = 'gini', max_depth=3)
DT.fit(X_train,Y_train) 

#plot the Decision Tree
tree.plot_tree(DT);

fn=['City Population','Work Experience','Undergraduate','Marital Status','Urban']
cn=['Risky', 'Good']
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
print(classification_report(Y_test, preds))


# **`Ensemble Technique : Random Forest Classifier`**

# Implementing Random Forest (Ensemble Technique) Classifier to check whether we get any improvement in the accuracy or not

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

# Initializing the RF Classifier
num_trees = 150
max_features = 5
RF = RandomForestClassifier(n_estimators= num_trees, max_features= max_features)
RF.fit(X_train, Y_train)

# Calculating Testing Data Accuracy
Y_pred_RF = RF.predict(X_test)
from sklearn.metrics import accuracy_score
acc_RF = accuracy_score(Y_pred_RF, Y_test)*100
acc_RF

