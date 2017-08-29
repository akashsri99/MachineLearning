#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 23:14:25 2017

@author: akashsrivastava
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir('/Users/akashsrivastava/Desktop/MachineLearning/DeepLearning/data')

dataset =  pd.read_csv('diabetes.csv')

from sklearn.preprocessing import StandardScaler


X = dataset.iloc[:,0:8].values
y = dataset.iloc[:,8:9].values

from keras.utils import to_categorical
#y1 = to_categorical(y)
scaler = StandardScaler()
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from keras.layers import Dense
from keras.models  import Sequential
from keras.optimizers import SGD,Adam


model = Sequential()
model.add(Dense(32,input_shape = (8,)))
model.add(Dense(32))
model.add(Dense(4))
model.add(Dense(1,activation='sigmoid'))



model.compile(Adam(lr=0.01),'binary_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=100)