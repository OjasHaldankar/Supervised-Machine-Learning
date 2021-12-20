#!/usr/bin/env python
# coding: utf-8

# # `K - Nearest Neighbours (KNN) Algorithm`

# In[1]:


# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[2]:


data = pd.read_csv("glass.csv")
data.head()


# In[3]:


data.shape


# In[4]:


data['Type'].value_counts()


# In[5]:


# Defining 'Independent Variable' & 'Target Variable'
arr = data.values
x = arr[:, :9]
Y = arr[:, 9]


# In[6]:


# Stndardizing feature values
sc = StandardScaler()
X = sc.fit_transform(x)


# In[7]:


# Dividing the data into Training & Testing sections
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.3, random_state= 42)


# In[8]:


# Initialize the model by assuming k = 3
KNN = KNeighborsClassifier(n_neighbors = 3)
KNN.fit(X_train, Y_train)


# In[10]:


# Calculating Training Accuracy
Y_pred_train_KNN = KNN.predict(X_train)
Train_acc_Knn = accuracy_score(Y_pred_train_KNN, Y_train)*100
Train_acc_Knn


# In[11]:


# Calculating Testing Data Accuracy
Y_pred_test_KNN = KNN.predict(X_test)
Test_acc_Knn = accuracy_score(Y_pred_test_KNN, Y_test)*100
Test_acc_Knn


# **Finding the Optimum Value of 'K'**

# In[12]:


# Initialize GridSearchCV
from sklearn.model_selection import GridSearchCV
n_neighbors = np.array(range(1,10))                #Finding optimum value of 'K' in the range of 1 to 10
param_grid = dict(n_neighbors = n_neighbors)


# In[13]:


KNN = KNeighborsClassifier()
grid = GridSearchCV(estimator= KNN, param_grid = param_grid)
grid.fit(X_train, Y_train)


# In[14]:


print(grid.best_params_)


# In[15]:


Train_acc_Knn_opt = grid.best_score_ *100
Train_acc_Knn_opt


# In[16]:


# Calculating Testing Data Accuracy
Y_pred_test_Knn = grid.predict(X_test)
Test_acc_Knn_opt = accuracy_score(Y_pred_test_Knn, Y_test)*100
Test_acc_Knn_opt


# In[17]:


print(classification_report(Y_test, Y_pred_test_Knn))


# ## `Balancing The Dataset` 

# In[18]:


# Balancing the output class 
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import NearMiss


# In[22]:


# Implementing Oversampling for Handling Imbalanced 
smk = SMOTETomek(random_state = 0)
X_res,Y_res=smk.fit_resample(X,Y)


# In[23]:


# Dividing the data into Training & Testing sections
from sklearn.model_selection import train_test_split
X_res_train, X_res_test, Y_res_train, Y_res_test = train_test_split(X_res, Y_res, test_size= 0.3, random_state= 0)


# In[24]:


KNN = KNeighborsClassifier()
grid = GridSearchCV(estimator= KNN, param_grid = param_grid)
grid.fit(X_res_train, Y_res_train)


# In[25]:


print(grid.best_params_)


# In[26]:


Train_acc_Knn_res = grid.best_score_ *100
Train_acc_Knn_res


# In[27]:


# Calculating Testing Data Accuracy
Y_pred_test_res = grid.predict(X_test)
Test_acc_Knn_res = accuracy_score(Y_pred_test_res, Y_test)*100
Test_acc_Knn_res


# In[28]:


print(classification_report(Y_test, Y_pred_test_res))

