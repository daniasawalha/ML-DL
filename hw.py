# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 23:34:32 2023

@author: Dania
"""

import numpy as np

# Define the initial weights and learning rate
w = np.array([[0.11, 0.12], [0.21, 0.08]])
v = np.array([0.14, 0.15])
alpha = 0.05

# Define the input and output values
x = np.array([2, 3])
y = np.array([1])


# Define the MLP network function
def MLP(x, w, v, alpha):
    
    z = np.dot(x, w)
    y_pred = np.dot(z, v)
    # Calculate the error
    error =0.5*(np.power( y - y_pred,2))
    
    delta_e =  y_pred -y
    
    # Update the weights
    t=alpha * delta_e * z
    v_new = v-t
    s=np.dot(x,v)
    ww=alpha*delta_e*s
    w_new = w - ww
    
    return w_new, v_new, error,y_pred

def min_error(ww,vv):

    error = 1
    epoch = 0
    while abs(error) > 0.000001:
    
        ww, vv, error ,y_pred= MLP(x, ww, vv, alpha)
        epoch += 1
        print(error)
    # Print the final number of epochs
    
    
    print("Number of epochs:", epoch)
    
    
    
min_error(w,v)