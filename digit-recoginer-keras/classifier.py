fr#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 03:15:51 2017

@author: akashsrivastava
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import to_categorical

import os
os.chdir('/Users/akashsrivastava/Desktop/MachineLearning/kaggle/digit-recoginer-keras')

dataset = pd.read_csv('train.csv')


Y1 = dataset.iloc[:,0].values
Y = to_categorical(Y1)
X = dataset.iloc[:,1:785].values


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import Adam




model = Sequential()
model.add(Dense(512,input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512,activation='tanh'))
model.add(Dense(10,activation='softmax'))
model.compile(Adam(lr=0.04),'categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train,Y_train,epochs=20,verbose=10)

y_pred = model.predict(X_test)
y_ans = np.argmax(y_pred,axis=1)

model.evaluate(X_test,y_pred)



