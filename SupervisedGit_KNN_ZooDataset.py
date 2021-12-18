#!/usr/bin/env python
# coding: utf-8

# # `K - Nearest Neighbours (KNN) Algorithm`

# .............................................................................................................................................
# 
# **~ Supervised Machine Learning Algorithm**
# 
# **~ Used for - Classification(Commonly) & Regression as well**
# 
# **~ Important Factors : K value (number of nearest neighbours) & Distance metric (e.g euclidean distance)** 
# 
# .............................................................................................................................................

# In[1]:


# Importing Necessary Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[2]:


# Importing the dataset

df = pd.read_csv("Zoo.csv")
df.head()


# In[3]:


# Dropping Animal Name Column (Not Needed for analysis)
df.drop(df.columns[[0]], axis=1, inplace= True)
df


# In[4]:


df['type'].value_counts()


# In[5]:


# Defining 'Independent Variable' & 'Target Variable'

arr = df.values
x = arr[:, :16]
Y = arr[:, 16]


# In[6]:


# Stndardizing feature values

sc = StandardScaler()

X = sc.fit_transform(x)


# In[7]:


# Dividing the data into Training & Testing sections

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.3, random_state= 0)


# In[8]:


# Initialize the model by assuming k = 4

KNN = KNeighborsClassifier(n_neighbors = 4)
KNN.fit(X_train, Y_train)


# In[9]:


# Calculating Training Accuracy

Y_pred_train_KNN = KNN.predict(X_train)
Train_acc_Knn = accuracy_score(Y_pred_train_KNN, Y_train)*100
Train_acc_Knn


# In[10]:


# Calculating Testing Data Accuracy

Y_pred_test_KNN = KNN.predict(X_test)
Test_acc_Knn = accuracy_score(Y_pred_test_KNN, Y_test)*100
Test_acc_Knn


# In[11]:


print(classification_report(Y_test, Y_pred_test_KNN))


# **`Finding Optimum value of 'K' using 'GridSearchCV'`**

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


# **'From above analysis & model building, we can understand that for K = 1 neighbours, model is having the best testing accuracy of 97%** 
