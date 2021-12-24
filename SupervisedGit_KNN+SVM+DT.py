#!/usr/bin/env python
# coding: utf-8

# # `Classification ML Algorithm : KNN, SVM, DT`

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


# ...............................................................................................................................
# 
# ## `ML Model [1]  :  K - Nearest Neighbour Classifier` 
# 
# ...............................................................................................................................

# Initializing KNN classifier
from sklearn.neighbors import KNeighborsClassifier

# Initialize the model by assuming k = 3
KNN = KNeighborsClassifier(n_neighbors = 3)
KNN.fit(X_train, Y_train)

# Calculating Training Accuracy
Y_pred_train_KNN = KNN.predict(X_train)
Train_acc_KNN = accuracy_score(Y_pred_train_KNN, Y_train)*100
Train_acc_KNN

# Calculating Testing Data Accuracy
Y_pred_test_KNN = KNN.predict(X_test)
Test_acc_KNN = accuracy_score(Y_pred_test_KNN, Y_test)*100
Test_acc_KNN

# **`Finding Optimum Value of 'K'`**

# Initialize GridSearchCV
from sklearn.model_selection import GridSearchCV
n_neighbors = np.array(range(1,10))                #Finding optimum value of 'K' in the range of 1 to 10
param_grid = dict(n_neighbors = n_neighbors)

KNN = KNeighborsClassifier()
grid = GridSearchCV(estimator= KNN, param_grid = param_grid)
grid.fit(X_train, Y_train)

print(grid.best_params_)
Train_acc_Knn = grid.best_score_ *100
Train_acc_Knn

# Calculating Testing Data Accuracy
Y_pred_test_Knn = grid.predict(X_test)
Test_acc_Knn = accuracy_score(Y_pred_test_Knn, Y_test)*100
Test_acc_Knn

print(classification_report(Y_test, Y_pred_test_Knn))


# ...............................................................................................................................
# 
# ## `ML Model [2]  :  Support Vector Classifier` 
# 
# ...............................................................................................................................

from sklearn import svm
from sklearn.svm import SVC                                 
from sklearn.model_selection import GridSearchCV

# Initializing SVC
svc = SVC()

param_grid = [{'kernel':['rbf'],'gamma':[50,20,10,5,0.5],'C':[15,14,13,12,11,10,5,1,0.1,0.001] }]
gsv = GridSearchCV(svc ,param_grid,cv=10)
gsv.fit(X_train,Y_train)

gsv.best_params_ , gsv.best_score_ 

svc = SVC(C= 15, gamma = 0.5)
svc.fit(X_train , Y_train)
y_pred_rad = svc.predict(X_test)
acc_rad = accuracy_score(Y_test, y_pred_rad) * 100
print("Accuracy =", acc_rad)

# Linear Kernal
svc1 = SVC(kernel= "linear") 
svc1.fit(X_train , Y_train)
y_pred_lin = svc1.predict(X_test)
acc_lin = accuracy_score(Y_test, y_pred_lin) * 100
print("Accuracy =", acc_lin)

# Polynomial Kernal
svc2 = SVC(kernel= "poly") 
svc2.fit(X_train , Y_train)
y_pred_poly = svc2.predict(X_test)
acc_poly = accuracy_score(Y_test, y_pred_poly) * 100
print("Accuracy =", acc_poly)

Test_acc_SVC = max(acc_rad, acc_lin, acc_poly)
Test_acc_SVC

y_pred_rad_train = svc.predict(X_train)
Train_acc_SVC = accuracy_score(y_pred_rad_train, Y_train)*100
Train_acc_SVC

print(classification_report(Y_test, y_pred_rad))


# ...............................................................................................................................
# 
# ## `ML Model [3]  :  Decision Tree Classifier` 
# 
# ...............................................................................................................................

# Initializing DT classifier
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree  

# Initialize the model
DT = DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
DT.fit(X_train,Y_train) 

#PLot the decision tree
tree.plot_tree(DT);

# Calculating Training Accuracy
Y_pred_train_DT = DT.predict(X_train)
Train_acc_DT = accuracy_score(Y_pred_train_DT, Y_train)*100
Train_acc_DT

#Predicting on test data
Y_pred_test_DT = DT.predict(X_test)                       # predicting on test data set 
pd.Series(Y_pred_test_DT).value_counts()  

# Calculating Testing Data Accuracy
Test_acc_DT =  np.mean(Y_pred_test_DT == Y_test)*100
Test_acc_DT

print(classification_report(Y_test, Y_pred_test_DT))


# **`Comparing The Results`**

Result = {'ML Algorithm':['KNN', 'SVC (Radial Kernal)', 'Decision Tree'],
    'Train Accuracy':[Train_acc_Knn, Train_acc_SVC, Train_acc_DT]
    ,'Test Accuracy':[Test_acc_Knn, Test_acc_SVC,  Test_acc_DT]}

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

