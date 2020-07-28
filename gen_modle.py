#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 15:57:33 2020

@author: majingbo
"""


# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from keras.layers import Input, BatchNormalization
from keras.layers.core import Dense
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import keras


#def my_metric_f2n(y_true, y_pred):
#    squared_difference = R2[ii] = 1 - ((Ypred-Ytrue)**2).sum()/((Ytrue - Ytrue.mean())**2).sum()
#    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`

def R2(y_true, y_pred):
    R2 = 1 - ((y_pred-y_true)**2).sum()/((y_true - y_true.mean())**2).sum()
    return R2  # Note the `axis=-1`


def R(y_true, y_pred):
    x = y_true
    y = y_pred
    cov = ((x-x.mean())*(y-y.mean())).mean()
    R = cov/(x.var()*y.var()).sqrt()
#    y_pred = pd.Series(y_pred)
#    R = y_true.corr(y_pred)
#    R = 1 - ((y_pred-y_true)**2).sum()/((y_true - y_true.mean())**2).sum()
    return R  # Note the `axis=-1`


def normalization(data):
    data2 = data.copy()
    for ii in np.arange(data.shape[1]):
        data2.iloc[:, ii] = ((data.iloc[:, ii] -  data.iloc[:, ii].min())/(data.iloc[:, ii].max() -  data.iloc[:, ii].min()))
        
    return data2
        

#model = load_model('model.h5')

readings = Input(shape=(4, ))
x = Dense(8, activation="linear", kernel_initializer="glorot_uniform")(readings)
x = Dense(32, activation="relu", kernel_initializer="glorot_uniform")(x)
x = Dense(64, activation="relu", kernel_initializer="glorot_uniform")(x)
x = Dense(32, activation="relu", kernel_initializer="glorot_uniform")(x)
x = Dense(16, activation="relu", kernel_initializer="glorot_uniform")(x)
x = Dense(8, activation="relu", kernel_initializer="glorot_uniform")(x)
benzene = Dense(4, activation="softplus", kernel_initializer="glorot_uniform")(x)

model = Model(inputs=[readings], outputs=[benzene])
model.compile(loss='mse', optimizer='adam', metrics=[])

NUM_EPOCHS = 10000
BATCH_SIZE = 100

folder = os.getcwd()
train_data = normalization(pd.read_csv(os.path.join(folder, 'train_data.csv')))
evalation_data = normalization(pd.read_csv(os.path.join(folder, 'evaluation_data.csv')))

my_callbacks = [keras.callbacks.CSVLogger(os.path.join(folder, 'train_process.csv'))]

history = model.fit(train_data.iloc[:, 0:4], train_data.iloc[:, 4:8],
                    batch_size=BATCH_SIZE, 
                    epochs=NUM_EPOCHS,
                    callbacks = my_callbacks,
                    validation_data=[evalation_data.iloc[:, 0:4], evalation_data.iloc[:, 4:8]])

model.save(os.path.join(folder, 'model.h5'))