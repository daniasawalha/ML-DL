# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 14:25:59 2023

@author: Dania
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#%matplotlib inline

icecream=pd.read_csv('IceCreamData.csv')

x=icecream[['Temperature']]
y=icecream['Revenue']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
random_state=0)

regression=LinearRegression(fit_intercept=True)
#regression.reshape(-1,1)
regression.fit(x_train,y_train)

print('Linear Model Coeff (m) =' , regression.coef_)
print('Linear Model Coeff (b) =' , regression.intercept_)
y_predict=regression.predict(x_test)
#print(y_predict)

plt.scatter(x_train, y_train, color="blue")
plt.plot(x_train, regression.predict(x_train),color="red")
plt.ylabel('Revenue [$]')
plt.xlabel('Temperature [degC]')
plt.title('Revenue Generated vs. Temperature @Ice Cream Stand (Training)')

plt.scatter(x_test, y_test, color="blue")
plt.plot(x_test, regression.predict(x_test),color="red")
plt.ylabel('Revenue [$]')
plt.xlabel('Temperature [degC]')
plt.title('Revenue Generated vs. Temperature @Ice Cream Stand (Training)')

print('---------0---------')
Temp = -0
Revenue = regression.predict([[Temp]])
print(Revenue)
print('--------35----------')
Temp = 35
Revenue = regression.predict([[Temp]])
print(Revenue)
print('--------55----------')
Temp = 55
Revenue = regression.predict([[Temp]])
print(Revenue)