#!/usr/bin/env python
# coding: utf-8

# # `Support Vector Machines`

# In[1]:


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


# In[2]:


# Reading the 'Forest Fires' file
df = pd.read_csv("forestfires.csv")
df.head(10)


# In[3]:


df.shape


# In[4]:


# EDA : Dropping columns which are not required for analysis i.e. Columns (1) & (2), as its dummy values are already present in the data
df = df.drop(columns= ['month', 'day'])
df


# In[5]:


# Label Encoding Target Variable i.e. size_category column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Target'] = le.fit_transform(df['size_category'])


# In[6]:


df['Target'].value_counts()


# In[8]:


import seaborn as sns
sns.countplot(x = 'Target', data = df )


# In[14]:


df = df.drop(columns = ['size_category'])
df.head()


# In[16]:


# Defining Feature & Target Variables
array = df.values
x = array[:, :28]
y = array[:, 28]


# In[17]:


# Training & Testing Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state= 42 )


# In[18]:


# Radial Kernel
test = SVC()
param_grid = [{'kernel':['rbf'],'gamma':[50,20,10,5,0.5],'C':[15,14,13,12,11,10,5,1,0.1,0.001] }]
gsv = GridSearchCV(test ,param_grid,cv=10)
gsv.fit(x_train,y_train)


# In[19]:


gsv.best_params_ , gsv.best_score_ 


# In[21]:


test = SVC(C= 15, gamma = 50)
test.fit(x_train , y_train)
y_pred = test.predict(x_test)
acc_rad = accuracy_score(y_test, y_pred) * 100
print("Accuracy of Radial Kernel=", acc_rad)
confusion_matrix(y_test, y_pred)


# In[22]:


# Linear Kernel
test = SVC(kernel= "linear") 
test.fit(x_train , y_train)
y_pred = test.predict(x_test)
acc_lin = accuracy_score(y_test, y_pred) * 100
print("Accuracy of Linear Kernel =", acc_lin)
confusion_matrix(y_test, y_pred) 


# In[23]:


# Polynomial Kernal
test = SVC(kernel= "poly") 
test.fit(x_train , y_train)
y_pred = test.predict(x_test)
acc_poly = accuracy_score(y_test, y_pred) * 100
print("Accuracy of Polynomial Kernel =", acc_poly)
confusion_matrix(y_test, y_pred)  


# In[25]:


Result = {'Kernel':['Radial Kernel','Linear Kernel','Polynomial Kernel'],
    'Accuracy':[acc_rad, acc_lin, acc_poly]}
Table = pd.DataFrame(Result)
Table


# In[26]:


Table.sort_values(by= ['Accuracy'], ascending= False)

