# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 10:57:11 2023

@author: Dania
"""

import numpy as np
from sklearn.linear_model import LinearRegression

x=np.array([5,15,25,35,45,55]).reshape((-1,1))
y=np.array([5,20,14,32,22,38])

model=LinearRegression()
model.fit(x, y)
r_sq=model.score(x, y)
print(r_sq)
print(model.intercept_)
print(model.coef_)
y_pred=model.predict(x)
print('predicted response:', y_pred, sep='\n')