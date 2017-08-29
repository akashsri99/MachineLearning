#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 03:11:39 2017

@author: akashsrivastava
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from sklearn.datasets import make_moons


X,y = make_moons(n_samples=1000,noise=0.1,random_state=0)
plt.plot(X[y==0,0],X[y==0,1],'ob',alpha=0.5)
plt.plot(X[y==1,0],X[y==1,1],'xr',alpha=0.5)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
 

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD,Adam


model = Sequential()
model.add(Dense(8,input_shape=(2,),activation='tanh'))
model.add(Dense(8,activation='tanh'))
model.add(Dense(8,activation='tanh'))
model.add(Dense(1,activation='sigmoid'))
model.compile(Adam(lr=0.02),'binary_crossentropy',metrics=['accuracy'])


model.fit(X_train,y_train,epochs=200)


model.evaluate(X_test,y_test)

