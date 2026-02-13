import os 
import sys 
import math 

import numpy as np 
import pandas as pd 

import keras 

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Convolution1D



def Write_weights(model):
    wt = open('weights.csv', 'w')
    bs = open('biases.csv', 'w')
    for idx,i in enumerate(model.layers):
        if(isinstance(i, Conv1D) or isinstance(i, Dense)):
            weights = i.get_weights()[0]
            biases = i.get_weights()[1]
            weights = weights.flatten()
            biases = biases.flatten()
            print(weights.shape, biases.shape)
            wt.write(','.join(map(str,weights.tolist()))+"\n")
            bs.write(','.join(map(str,biases.tolist()))+"\n")
    #np.savetxt("weight" + str(idx)+".csv" , weights , fmt='%s', delimiter=',')
    #np.savetxt("bias" + str(idx) +".csv" , biases , fmt='%s', delimiter=',')

    return wt, bs 


model = Sequential() 
input_shape = (360, 1)  # Assuming the 1D data is reshaped to (360, 1) and adding a channel dimension

model.add(Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=input_shape))
model.add(MaxPooling1D(strides=2))
 
model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(strides=2))

model.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D( strides=2))

    # model.add(Conv1D(filters=128, kernel_size=17, strides=1, padding='same', activation='relu'))
    # model.add(MaxPooling1D( strides=2))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
# model.add(Dense(50, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.load_weights('my_model.h5')

wt, bs = Write_weights(model=model) 
print(bs) 

