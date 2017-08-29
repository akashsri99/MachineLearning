# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_circles

X,y = make_circles(n_samples= 1000,noise=0.1,factor=0.2,random_state=0)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD


model = Sequential()
model.add(Dense(4,input_shape=(2,),activation='tanh'))
model.add(Dense(1,activation='sigmoid'))
model.compile(SGD(lr=0.5),'binary_crossentropy',metrics=['accuracy'])


model.fit(X,y,epochs=20)
