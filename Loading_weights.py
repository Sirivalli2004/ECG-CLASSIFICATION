import os 
import sys 
import math  

import numpy as np 

import keras 

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Convolution1D

os.system('cls' if os.name =='nt' else 'clear')


model = Sequential() 
input_shape = (360, 1)  # Assuming the 1D data is reshaped to (360, 1) and adding a channel dimension

model.add(Conv1D(filters=16, kernel_size=11, strides=1, padding='same', activation='relu', input_shape=input_shape))
model.add(MaxPooling1D(strides=2))

model.add(Conv1D(filters=32, kernel_size=13, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(strides=2))

model.add(Conv1D(filters=64, kernel_size=15, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D( strides=2))

    # model.add(Conv1D(filters=128, kernel_size=17, strides=1, padding='same', activation='relu'))
    # model.add(MaxPooling1D( strides=2))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
# model.add(Dense(50, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.load_weights('my_model.h5')
print('model weights are loaded')