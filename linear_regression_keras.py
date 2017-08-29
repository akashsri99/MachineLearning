#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 01:07:26 2017

@author: akashsrivastava
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

os.chdir('/Users/akashsrivastava/Desktop/MachineLearning/DeepLearning/data')

dataset = pd.read_csv('weight-height.csv')
dataset.plot(kind='scatter',x='Height',y='Weight')

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD


X = dataset['Height']
y = dataset['Weight']

model = Sequential()
model.add(Dense(1,input_shape = (1,)))
model.compile(Adam(lr=0.5),'mean_squared_error')
model.fit(X,y,epochs=50)

dataset.plot(kind='scatter',x='Height',y='Weight')
plt.plot(X,model.predict(X),color='r')


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=0)

model.fit(X_train,y_train,epochs=50)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)


from sklearn.metrics import mean_squared_error as mse
print(mse(y_train,train_pred))
print(mse(y_test,test_pred))

