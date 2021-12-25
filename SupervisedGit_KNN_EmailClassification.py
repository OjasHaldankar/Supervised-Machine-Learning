#!/usr/bin/env python
# coding: utf-8

# # `K - Nearest Neighbours (KNN) Classifier`

# About the dataset  :  The dataset contains content of some emails which were either 'abusive' or 'non abusive' emails. It is pre classified & labelled. The objective is to develop a classification KNN model for predicting the class of the email i.e. either 'Abusive Email' or 'Non Abusive Email' 

# Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

# Importing the data
KNN = pd.read_csv("Emails.csv")
KNN
KNN = KNN.rename(columns={KNN.columns[0]: 'Email_content'})

# Checking for duplicate entries & removing the duplicates
KNN_model = KNN.drop_duplicates()
KNN_model = KNN_model.reset_index(drop= True)
KNN_model


KNN_model['Class'].value_counts()                     # 0 : 'Non Abusive Class' 1 : 'Abusive Class'
sns.countplot(x= 'Class', data = KNN_model)

# As the target class is imbalanced, performing a comparative study for imbalanced & balanced dataset
# Label Encoding for converting string to required datatype 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
values = np.array(KNN_model['Email_content'])
Encoded = le.fit_transform(values)
Encoded

#Adding array to original dataset
KNN_model['Email_encoded'] = le.fit_transform(values)

# Selecting only the required columns for analysis
KNN_email = KNN_model[['Email_encoded', 'Class']]
KNN_email


# **`Case [1] : Analysis on Imbalanced Dataset`**

# Defining feature & target variable
array_knn = KNN_email.values
x = array_knn[:, 0]
X = x.reshape(-1, 1)
Y = array_knn[:, 1]

#Dividing the dataset into training data & testing data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)

# Initialize the model by assuming k = 6
model = KNeighborsClassifier(n_neighbors=6)
model.fit(X_train, Y_train)

#Prediction
result = model.predict(X_test)
accuracy = np.mean(result==Y_test)
accuracy

print(classification_report(Y_test, result))


# **Finding Optimum Value of 'K'**

# Initialize GridSearchCV
from sklearn.model_selection import GridSearchCV
n_neighbors = np.array(range(1,30))                #Finding optimum value of 'K' in the range of 1 to 30
param_grid = dict(n_neighbors = n_neighbors)

model = KNeighborsClassifier()
grid = GridSearchCV(estimator=model, param_grid = param_grid)
grid.fit(X_train, Y_train)

print(grid.best_score_)
print(grid.best_params_)

# Visualizing the result
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
k_range = range(1, 30)                              #Plot for values of 'K' in the range of 1 to 30
k_scores = []
# for loop
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn, X_train, Y_train, cv=10)
    k_scores.append(scores.mean())
# plot to see clearly
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy for Training Data')
plt.show()

#Prediction for optimum value of K = 11
grid_result = grid.predict(X_test)
acc_imb = np.mean(grid_result==Y_test)
acc_imb

print(classification_report(Y_test, grid_result))
# **The Accuracy of the model by using 'KNN Classifier' for Unbalanced Dataset is '94%' for the optimum value of {K = 11} i.e. 11 nearest neighbours**


# **`Case [2] : Analysis of Balanced Dataset`**

# Balancing the dataset using 'SMOTE' 
from imblearn.combine import SMOTETomek
smote = SMOTETomek(random_state = 1)
X_bal, Y_bal = smote.fit_resample(X,Y)

bal_shape = pd.DataFrame(Y_bal, columns = ['balanced_op'])
bal_shape['balanced_op'].value_counts()
sns.countplot(x= 'balanced_op', data= bal_shape)

#Dividing the dataset into training data & testing data
from sklearn.model_selection import train_test_split
X_train_bal, X_test_bal, Y_train_bal, Y_test_bal = train_test_split(X_bal, Y_bal, test_size = 0.25, random_state = 42)


# **Finding Optimum Value of 'K' for Balanced Dataset**

# Initialize GridSearchCV
n_neighbors1 = np.array(range(1,30))                #Finding optimum value of 'K' in the range of 1 to 30
param_grid1 = dict(n_neighbors = n_neighbors1)

model = KNeighborsClassifier()
grid1 = GridSearchCV(estimator=model, param_grid = param_grid1)
grid1.fit(X_train_bal, Y_train_bal)

print(grid1.best_score_)
print(grid1.best_params_)

# Visualizing the result
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
k_range = range(1, 30)                              #Plot for values of 'K' in the range of 1 to 30
k_scores = []
# for loop
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn, X_train_bal, Y_train_bal, cv=10)
    k_scores.append(scores.mean())    
# plot to see clearly
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy for Balanced Data')
plt.show()

#Prediction
grid1_result = grid1.predict(X_test_bal)
acc_bal = np.mean(grid1_result==Y_test_bal)
acc_bal

print(classification_report(Y_test_bal, grid1_result))
# **The Accuracy of the model by using 'KNN Classifier' for Balanced Dataset is '77%' for the optimum value of {K = 16} i.e. 16 nearest neighbours**



# ## *`Key insights obtained from KNN Classifier Algorithm`* 

# ................................................................................
 
# *1} The Dataset is highly imbalanced. Though overall accuracy for the 'unbalanced data' is higher, the 'Abusive class' has very low recall (0.13) & low f1 score (0.22). As a result, good insights cannot be generated. So accuracy alone cannot be a good measure to analyse the performance.*
# 
# *2) KNN is a 'lazy learning' algorithm. It has a natural tendancy to pick up the patterns in popular class & ignore the least popular class, which generate high accuracy but that is a false sense of performance.*
# 
# *3) In such cases; Confusion Matrix, Precision, Recall & F1 score would be better measures & after balancing the dataset, there is improvement in Precision (0.76), Recall (0.79) & F1 score (0.77).*

# ................................................................................
