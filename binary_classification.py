# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 13:36:09 2023

@author: Dania
"""
import numpy as np 

def classification(z):
    if z >0:
        return 1
    return 0

w=np.array([1,1])
b=-1
x=np.array([0,0])
print("0 and 0 :",classification(w.dot(x)+b))
x=np.array([0,1])
print("0 and 1 :",classification(w.dot(x)+b))
x=np.array([1,0])
print("1 and 0 :",classification(w.dot(x)+b))
x=np.array([1,1])
print("1 and 1 :",classification(w.dot(x)+b))

b=0
x=np.array([0,0])
print("0 or 0 :",classification(w.dot(x)+b))
x=np.array([0,1])
print("0 or 1 :",classification(w.dot(x)+b))
x=np.array([1,0])
print("1 or 0 :",classification(w.dot(x)+b))
x=np.array([1,1])
print("1 or 1 :",classification(w.dot(x)+b))

def activation(z):
    if z!=0.5:
        return 0
    return 1
print("----------------------------")
b=0
w=np.array([0.5,0.5])
x=np.array([0,0])
print("0 xor 0 :",activation(w.dot(x)+b))
x=np.array([1,0])
print("1 xor 0 :",activation(w.dot(x)+b))
x=np.array([0,1])
print("0 xor 1 :",activation(w.dot(x)+b))
x=np.array([1,1])
print("1 xor 1 :",activation(w.dot(x)+b))