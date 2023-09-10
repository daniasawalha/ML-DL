# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 17:12:28 2023

@author: Dania
"""
import numpy as np 

def nonlin(x,deriv=False):
    if(deriv==True):
        #
        return x*(1-x)
    return(1/(1+np.exp(-x)))
x=np.array([[0,0,1],[0,1,1],[0,1,0],[1,1,1]])
y=np.array([[0,0,1,1]]).T

w=2*np.random.random((3,1))-1
np.random.seed(1)
#training sample is small so what ever is still the same
#try change range/epochs
for iter in range (50):
    #exp
    z=np.dot(x,w)
    z=nonlin(x)
    error=y-z
    delta_z=error*nonlin(z,True)
    w=w+np.dot(x.T,delta_z)
    
print("End of Training")
print(" The output is:")
print(z)