#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 02:17:59 2017

@author: akashsrivastava
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os

os.chdir('/Users/akashsrivastava/Desktop/MachineLearning/DeepLearning/data')

dataset = pd.read_csv('user_visit_duration.csv')

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

X = dataset.iloc[:,0:1].values
y = dataset.iloc[:,1:2].values

X_train,X_test,y_train,y_test  = train_test_split(X,y,train_size=0.8,random_state=0)


model = Sequential()
model.add(Dense(units=1,input_shape=(1,),activation='sigmoid'))
model.compile(SGD(lr=0.8),'binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=50)


y_pred = model.predict(X_test)
y_ans = y_pred >0.5

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_ans))

final_y = model.predict(X)
ans = final_y>0.5

plt.scatter(X,y,color='b',alpha=0.5)
plt.scatter(X,ans,color='r')


