# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:39:54 2023

@author: Dania
"""

import numpy as np


w=np.array([[0.11,0.12],[0.21,0.08]])
v=np.array([0.14,0.15])

alfa=0.05
x=np.array([2,3])
y=np.array([1.0])



        #print("mmm:",qq,"wn:",wn,"vn:",vn)
    
        
def layer(x,ww,vv,allfa):
    #global z,q,e,delta_e
    z=np.dot(x,ww)
    q=np.dot(z,vv)
    
    e=0.5*(np.power((q-y),2))

    #print("kkkkk:",e)
    delta=q-y
    t=alfa*delta*z
    v=vv-(t)
    s=np.dot(x,vv)
    r=alfa*delta*s
    w=ww-r
    return (w,v,e,q)


for iter in range(7):
        
      w, v, error ,y_pred= layer(x,w,v,alfa)
      print(error)
        



        #return(layer(x,ww,vv,y))
#print(step3(v,alfa,delta_e))
#layer(x,w,v,y)
#step3(v,alfa)
#r=step4(v,alfa,x,w)
#print(r)

