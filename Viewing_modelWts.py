import os 
import sys 
import math 

import h5py
import csv 

import numpy as np 
import pandas as pd 


os.system('cls' if os.name  =='nt' else 'clear')

file_path = 'my_model.h5'
f = h5py.File(file_path, 'r')

print("Keys in the HDF5 file:", list(f.keys()))

model_weights_group = f['model_weights']
print("\nKeys in 'model_weights' group:", list(model_weights_group.keys())) 

layer_name = 'conv1d'
if layer_name in model_weights_group:
    layer_group = model_weights_group[layer_name]

    print(f"Keys within '{layer_name}':", list(layer_group.keys()))

#print(layer_group['sequential']['conv1d_2'].keys())

#print(layer_group['sequential']['conv1d_2']['kernel'][:],'\n')
#print('conv bias value',layer_group['sequential']['conv1d_2']['bias'][:]) 

#print(layer_group['sequential'].keys())
#print(layer_group['sequential']['dense_1'].keys())
#print(layer_group['sequential']['dense_1']['kernel'][:])
#print(layer_group['sequential']['dense_1']['bias'][:]) 


conv_data_weights = layer_group['sequential']['conv1d']['kernel'][:]
print(conv_data_weights.flatten())
#conv_data_weights.to_csv('conv_1_weights.csv')

