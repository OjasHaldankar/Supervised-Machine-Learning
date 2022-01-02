#!/usr/bin/env python
# coding: utf-8

# ## `Supervised Machine Learning : DT Classifier + Random Forest (Ensemble Technique)` 


# **`Model (1)  :  DECISION TREE CLASSIFIER`**

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split      
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree                                 
from sklearn.metrics import classification_report

# Importing the dataset
data = pd.read_csv("Company_Data.csv")
data.head()

data.describe()

# Converting Categorical data to numeric
data = pd.get_dummies(data,columns=['ShelveLoc','Urban','US'], drop_first=True)
data

# Dividing Target Variable i.e. 'Sales' into buckets [0 to 9] --> Poor Sales & [10 & above] --> Good Sales
data["Sale"] = pd.cut(data["Sales"], bins = [0,10,20], labels = ["Poor Sales", "Good Sales"])
le = LabelEncoder()
data['Target'] = le.fit_transform(data['Sale'])
data

# Dropping columns not required for analysis
data = data.drop(columns= ['Sale'])
data['Target'].value_counts()

import seaborn as sns
sns.countplot(x = 'Target', data = data)

# Defining Feature & Target Variables
X = data.iloc[:, 1:12]
Y = data['Target']

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

# Accuracy of the Decision Tree model
acc_DT = np.mean(preds==Y_test)*100
acc_DT



# **`Model [2]  :  Random Forest Classifier (Ensemble Technique)`**

# Importing Libraries
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

num_trees = 50
max_feat = 11

#Initialize Random Forest Classifier
RF = RandomForestClassifier(n_estimators= num_trees, max_features= max_feat)
RF.fit(X_train, Y_train)

from sklearn.metrics import accuracy_score
# Calculating Training Accuracy
Y_pred_train_RF = RF.predict(X_train)
Train_acc_RF = accuracy_score(Y_pred_train_RF, Y_train)*100
Train_acc_RF

# Calculating Testing Data Accuracy
Y_pred_test_RF = RF.predict(X_test)
Test_acc_RF = accuracy_score(Y_pred_test_RF, Y_test)*100
Test_acc_RF

Result = {'ML Algorithm':['Decision Tree', 'Random Forest'],
    'Model Accuracy':[acc_DT, Test_acc_RF]}

table = pd.DataFrame(Result)
table

plt.figure(figsize =(20, 6))
plt.plot(table['ML Algorithm'], table['Model Accuracy'])
plt.title('Model Accuracies',fontdict={'fontsize': 25,'fontweight' : 15,'color' : 'g'})
plt.xlabel('ML Algorithm')
plt.ylabel('Accuracy')
plt.legend(['Accuracy of Model'], loc='lower right')
plt.grid()
plt.show();

