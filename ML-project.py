# -*- coding: utf-8 -*-
"""
Created on Sat May 27 17:44:14 2023

@author: Dania
"""

from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import sklearn.metrics

# In[28]:


#load the iris data
data = datasets.load_iris()
print(data.keys())


# In[29]:


x = pd.DataFrame(data = data.data, columns = data.feature_names)
y = pd.DataFrame(data = data.target, columns = ['species'])

print(x.head())
print(y.head())


# In[30]:


#split the dataset
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size = 0.2, random_state= 0)
# Reshape y_train and y_test to 1-dimensional arrays
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()
print("x_train shape", x_train.shape)
print("y_trian shape", y_train.shape)


# In[31]:


#split the validation 
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.35, random_state = 6)
print("x_val", x_val.shape)
print("y_val", y_val.shape)


# In[32]:


#the support vector machine, acc.
clf = svm.SVC(kernel = 'poly', C = 1).fit(x_train, y_train)
train_accuracy = clf.score(x_train, y_train)
validation_accuracy = clf.score(x_val, y_val)
svm_accuracy = clf.score(x_test, y_test)
print("accuracy:",train_accuracy)
print('validation accuracy:',validation_accuracy)
print('SVM testing accuracy:',svm_accuracy)
print("********************************************")
# In[33]:


#predict the class labe of the test set
y_pred = clf.predict(x_test)
print("svm prediction:",y_pred)


# In[34]:


y_pred_val = clf.predict(x_val)
print("svm validation predict",y_pred_val)


# In[35]:


#loss function
def loss_function(y_test, y_pred):
    #y_pred = y_pred.reshape(30, 1)
    return np.mean((y_test - y_pred)**2)

# Calculate the mean squared error (MSE) between predicted and actual values
mse = mean_squared_error(y_val, y_pred_val)
print("Mean Squared Error (MSE)/for validation:", mse)
print("error:",loss_function(y_test, y_pred))
print("---------------------------------------")
# In[36]:

from sklearn.neural_network import MLPClassifier
# Create and train the MLP classifier model
mlp_model = MLPClassifier(hidden_layer_sizes=(78,54), activation='relu', max_iter=600)
mlp_model.fit(x_train, y_train)

train_score = mlp_model.score(x_train, y_train)
mlp_val_accuracy = mlp_model.score(x_val,y_val)
print("mlp validation accuracy:",mlp_val_accuracy)
print("accuracy:",train_score)
mlp_accuracy = mlp_model.score(x_test, y_test)
print('mlp accuracy:',mlp_accuracy)
print("********************************************")


y_pred_mlp_val = mlp_model.predict(x_val)
print("y_pred_mlp_val:",y_pred_mlp_val)
# Predict the target values for the test set
y_pred_mlp = mlp_model.predict(x_test)
print("y_pred_mlp:",y_pred_mlp)
# Calculate the mean squared error (MSE) between predicted and actual values
mse = mean_squared_error(y_val, y_pred_mlp_val)
print("Mean Squared Error (MSE)/for validation:", mse)
print("error:",loss_function(y_test,y_pred_mlp))


print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("for SVM:")
print (sklearn.metrics.classification_report(y_test, y_pred))
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("for MLP:")
print (sklearn.metrics.classification_report(y_test, y_pred_mlp))
