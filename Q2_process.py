# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 08:09:32 2020

@author: 马靖博
genetic algorithm (GA) 
"""

import pandas as pd
import numpy as np
import random
from keras.models import load_model
import itertools
import os
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
"""

"""        
def cal_binary(value):
    down = sample_down
    up = sample_up
    binary = ''
    for ii in np.arange(len(value)):
        digit_value = int((2**digit-1)*(value[ii]-down[ii])/(up[ii]-down[ii]))
        binary_piece = bin(digit_value)
        binary_piece = binary_piece[2:]
        binary_piece = (digit-len(binary_piece))*'0'+binary_piece
        binary = binary + binary_piece
    return binary

def cal_value(binary):
    down = sample_down
    up = sample_up
    value = np.zeros((1,len(down)))
#    print(value.size)
    for ii in np.arange(value.size):
        binary_piece = binary[(ii*digit):(ii*digit+digit)]
        value[0,ii] = int(binary_piece,2)/int(2**digit-1)*(up[ii]-down[ii])+down[ii]
    return value[0,:].tolist()

def gen_sample(sample_number):     
#    all_samples = []
    value = np.zeros([sample_number, len(sample_down)])
    for ii in np.arange(sample_number):
        for jj in np.arange(len(sample_down)):
            value[ii,jj] = random.uniform(sample_down[jj], sample_up[jj])
    all_samples = value.tolist()
    return all_samples

def choose_sample(sample_all):
    num = random.choice(np.arange(len(sample_all)))
    return sample_all[num]

def variation_sample(one_sample):
    binary = cal_binary(one_sample)
    for ii in np.arange(variation_length):   
        num = random.choice(np.arange(len(binary)))
        if binary[num] == '0':
            binary = binary[0:num]+'1'+binary[(num+1):]
        elif binary[num] == '1':
            binary = binary[0:num]+'0'+binary[(num+1):]
    one_sample = cal_value(binary)
    return one_sample

def cross_sample(one_sample, two_sample):
    binary_one = cal_binary(one_sample)
    binary_two = cal_binary(two_sample)
    cross_length_max = 10
    cross_length = random.choice(np.arange(cross_length_max))
    cross_start = random.choice(np.arange(len(binary_one) - cross_length))
    change_two = binary_two[cross_start:(cross_start+cross_length)]
    binary_one = binary_one[0:cross_start] + change_two + binary_one[(cross_start+cross_length):]
#    print(binary_one)
    one_sample = cal_value(binary_one)
    return one_sample
"""

"""
def norm(data):
    data2 = data.copy()
    for ii in np.arange(data.shape[1]):
        data2.iloc[:, ii] = input_norm.iloc[0, ii]*data.iloc[:, ii] + input_norm.iloc[1, ii]
        
    return data2

def reverse_norm(data):
    data2 = data.copy()
    for ii in np.arange(data.shape[1]):
        data2.iloc[:, ii] = (data.iloc[:, ii] - output_norm.iloc[1, ii])/output_norm.iloc[0, ii]
    return data2

def cal_output(sample):
    fix_variable = np.array([vapor_concentration, 
                             fuel_electrode_flow, 
                             temperature])*np.ones([sample.shape[0],1])
    Xtest = pd.DataFrame(np.append(fix_variable, sample, axis=1))

    Xtest = norm(Xtest)
    Ypredicted = model.predict(Xtest)
    Ypredicted = pd.DataFrame(Ypredicted)
    Ypredicted = reverse_norm(Ypredicted)
    return np.array(Ypredicted)
"""

"""
def one_change(all_sample):
    new_sample = choose_sample(all_sample)

    if random.random() <= cross_probability:
        sample_cross = choose_sample(all_sample)
        new_sample = cross_sample(new_sample, sample_cross)
        
    if random.random() <= variation_probability:
        new_sample = variation_sample(new_sample)
        
    return new_sample
    
def enlarge_sample(all_sample, mag):
    enlarge_sample = all_sample.copy()
    for ii in np.arange(mag*len(all_sample)):
        new_sample = one_change(all_sample)
        enlarge_sample.append(new_sample)
    new_sample = gen_sample(len(all_sample))
    enlarge_sample = enlarge_sample + new_sample
    return enlarge_sample

def sort_sample(sample):
    sample_input = pd.DataFrame(np.array(sample))
    sample_input.columns = ['V']
    sample_output = pd.DataFrame(cal_output(sample_input))
    sample_output.columns = ['cd',
                             'H2',
                             'CO',
                             'Q']
    sample_sort = pd.DataFrame(-abs(sample_output.Q))
    sample_sort.columns = ['sort']
    sample_all = pd.concat([sample_input, sample_output, sample_sort], axis=1)
#    sample_all = pd.DataFrame(np.append(sample_all, sample_sort, axis=1))
#    sample_all.columns = ['V',
#                          'current density',
#                        'H2 mole faction',
#                        'CO',
#                        'Q',
#                        'sort']
    hh = sample_all
    hh1 = hh[abs(hh['Q'])<Q_limit]
    hh1 = hh1[abs(hh1['V'])>0.9]
#    hh1 = hh[abs(hh['Q'])>-Q_limit]
    hh2 = hh1.sort_values(by='sort', ascending=False)
#    print('在热平衡条件下，电压是：', hh2.iloc[0,0])
#    print(hh2.shape)
    return np.array(hh2.iloc[0:sample_num, 0:1]), hh2.iloc[0, 0:-1]
    
global variation_probability, variation_length, cross_probability
variation_probability = 0.2
variation_length = 5
cross_probability = 0.8

"""
可调节参数
"""



global sample_down, sample_up, digit, sample_num
sample_num = 100
sample_down = [0.8] #变量1和变量2的下限
sample_up = [2] #变量1和变量2的上限
digit = 20

global model
folder = folder = os.getcwd()
model = load_model(os.path.join(folder, 'model.h5'))

global input_norm, output_norm
input_norm = pd.read_csv(os.path.join(folder, 'input_norm.txt'), header=None)
output_norm = pd.read_csv(os.path.join(folder, 'output_norm.txt'), header=None)

global anode_velocity, tem_gradient_max
anode_velocity = 16
Q_limit = 0.1

circle_num = 40
enlarge_num = 20
"""
程序开始运行
"""
#vapor_concentration_all = np.arange(0.1, 0.91, 0.2)
vapor_concentration_all = np.array([0.5])
fuel_electrode_flow_all = np.array([500])
#temperature_all = np.arange(1023, 1124, 20)
temperature_all = np.array([1023])

global vapor_concentration, fuel_electrode_flow, temperature
vapor_concentration = 0.1
fuel_electrode_flow = 400
temperature = 1023

parameter = list(itertools.product(vapor_concentration_all, temperature_all))
output_data = pd.DataFrame(np.zeros([len(parameter), 7]))
for hh in np.arange(fuel_electrode_flow_all.size):
    #temperature = temperature_all[hh]
    fuel_electrode_flow = fuel_electrode_flow_all[hh]
    
#    fig1 = plt.figure()
#    ax1 = Axes3D(fig1)
#    fig2 = plt.figure()
#    ax2 = Axes3D(fig2)
    pros = []
    for pp in tqdm(parameter):
        vapor_concentration, temperature = pp
        sample = gen_sample(sample_num)
        sample1 = sample
        for ii in np.arange(circle_num):
#            print('这是第', ii+1, '次循环')
            sample2 = enlarge_sample(sample1, enlarge_num)
            sample3, result= sort_sample(sample2)
            sample1 = list(sample3)
            pros.append(result.Q)
        voltage, current_density, h2mole, COmole, Q= result
        output =  pd.Series(np.array([vapor_concentration,
                        fuel_electrode_flow,
                        temperature,
                        voltage,
                        current_density,
                        h2mole,
                        COmole,
                        Q]))
        gg = parameter.index(pp)
        output_data.iloc[gg, :] = output
#        print('第',gg,'/',len(parameter)*temperature_all.size,'次计算完成')
#        print(output)
#        print(output_data.iloc[gg:gg+1, :])
#    pd.DataFrame(pros).to_csv(os.path.join(folder, 'process'+'.csv'), header=None, index=None)
    plt.plot(pros)
    pd.DataFrame(pros).to_csv(os.path.join(folder, 'process.csv'),header=None,index=None)
#    ax1.scatter(output_data.iloc[:, 0], output_data.iloc[:, 2], output_data.iloc[:, 3])
#    ax2.scatter(output_data.iloc[:, 0], output_data.iloc[:, 2], output_data.iloc[:, 4])
#    
        

        
        



    output_data.to_csv(os.path.join(folder, str(fuel_electrode_flow)+'Q2.txt'),header=None,index=None)
#    output_data_final = pd.DataFrame(np.append(output_data, sample, axis=1))
    
    
    
    
    

