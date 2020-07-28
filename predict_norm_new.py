# -*- coding: utf-8 -*-
"""

"""
from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model, load_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from mykeras import R, R2

#def R2(y_true, y_pred):
#    R2 = 1 - ((y_pred-y_true)**2).sum()/((y_true - y_true.mean())**2).sum()
#    return R2  # Note the `axis=-1`
#
#
#def R(y_true, y_pred):
#    x = y_true
#    y = y_pred
#    cov = ((x-x.mean())*(y-y.mean())).mean()
#    R = cov/(x.var()*y.var()).sqrt()
##    y_pred = pd.Series(y_pred)
##    R = y_true.corr(y_pred)
##    R = 1 - ((y_pred-y_true)**2).sum()/((y_true - y_true.mean())**2).sum()
#    return R  # Note the `axis=-1


def norm(data):
    data2 = data.copy()
    for ii in np.arange(data.shape[1]):
        data2.iloc[:, ii] = input_norm.iloc[0, ii]*data.iloc[:, ii] + input_norm.iloc[1, ii]        
    return data2

def norm_output(data):
    data2 = data.copy()
    for ii in np.arange(data.shape[1]):
        data2.iloc[:, ii] = output_norm.iloc[0, ii]*data.iloc[:, ii] + output_norm.iloc[1, ii]        
    return data2


def reverse_norm(data):
    data2 = data.copy()
    for ii in np.arange(data.shape[1]):
        data2.iloc[:, ii] = (data.iloc[:, ii] - output_norm.iloc[1, ii])/output_norm.iloc[0, ii]
    return data2

global model
folder = os.getcwd()
model = load_model(os.path.join(folder, 'model.h5'))

global input_norm, output_norm
train_data = pd.read_csv(os.path.join(folder, 'train_data.csv'))
input_norm = pd.read_csv(os.path.join(folder, 'input_norm.txt'), header=None)
output_norm = pd.read_csv(os.path.join(folder, 'output_norm.txt'), header=None)

train_data = pd.read_csv(os.path.join(folder, 'train_data.csv'))

train_data = train_data.iloc[0:2, :]
Xtest = train_data.iloc[:, 0:4]
Ytest = train_data.iloc[:, 4:8]
Ytest2 = Ytest
Xtest2 = Xtest

Xtest = norm(Xtest)
Ytest = norm_output(Ytest)
Ypredicted = model.predict(Xtest)
Ypredicted = pd.DataFrame(Ypredicted)
Ypredicted2 = reverse_norm(Ypredicted)
accuracy = pd.DataFrame(abs(np.array(Ypredicted)/np.array(Ytest)-1))
accuracy2 = pd.DataFrame(abs(np.array(Ypredicted2)/np.array(Ytest2)-1))


output_all = pd.concat([Xtest2, Ytest2, Ypredicted2, accuracy], axis=1)
#output_all = output_all[]

all_accuracy = []
step = 0.01
uplimit = 100
for ii in np.arange(accuracy.shape[1]):
    variable = accuracy.iloc[:, ii]
    number = np.zeros([uplimit+1, 1])
    for jj in np.arange(uplimit):
        number[jj] = (variable[variable<((jj+1)*step)].size - variable[variable<(jj*step)].size)/accuracy.shape[0]
        
    plt.plot(number)
    plt.show()



#output_all = pd.concat([Xtest2, Ytest2, Ypredicted2, accuracy], axis=1)
output_all.to_csv(os.path.join(folder, 'train_data_delta.csv'), header=None, index=None)
pd.DataFrame(number).to_csv(os.path.join(folder, 'train_data_number.csv'), header=None, index=None)
################################################################################
#evaluation_data = pd.read_csv(os.path.join(folder, 'evaluation_data.csv'))
#
#Xtest = evaluation_data.iloc[:, 0:4]
#Ytest = evaluation_data.iloc[:, 4:8]
#Ytest2 = Ytest
#Xtest2 = Xtest
#
#Xtest = norm(Xtest)
#Ytest = norm_output(Ytest)
#Ypredicted = model.predict(Xtest)
#Ypredicted = pd.DataFrame(Ypredicted)
#Ypredicted2 = reverse_norm(Ypredicted)
#accuracy = pd.DataFrame(abs(np.array(Ypredicted)/np.array(Ytest)-1))
#
#all_accuracy = []
#step = 0.01
#uplimit = 100
#for ii in np.arange(accuracy.shape[1]):
#    variable = accuracy.iloc[:, ii]
#    number = np.zeros([uplimit+1, 1])
#    for jj in np.arange(uplimit):
#        number[jj] = (variable[variable<((jj+1)*step)].size - variable[variable<(jj*step)].size)/accuracy.shape[0]
#        
#    plt.plot(number)
#    plt.show()
#
#
#
#output_all = pd.concat([Xtest2, Ytest2, Ypredicted2, accuracy], axis=1)
#output_all.to_csv(os.path.join(folder, 'evaluation_data_delta.csv'), header=None, index=None)
#pd.DataFrame(number).to_csv(os.path.join(folder, 'evaluation_data_number.csv'), header=None, index=None)

