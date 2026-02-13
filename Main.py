import os
import sys
import math

import pywt
import csv
import glob
import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 


from scipy import stats
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling1D,Conv1D, MaxPooling1D, Convolution1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam


from utils import Generate_fileList, Sample_signaReading, denoise, Generate_classDistribution, Pie_chart, class_imbalance, After_imbalanceRemoval
from Data_Generation import Generate_Data

os.environ["CUDA_VISIBLE_DEVICES"]='0'
plt.rcParams["figure.figsize"] = (10,6)


if __name__ == '__main__':

    os.system('cls' if os.name == 'nt' else 'clear')

    classes = ['N', 'L', 'R', 'A', 'V']
    n_classes = len(classes)
    count_classes = [0]*n_classes

    print(count_classes)  

    path = r'C:\Users\VPG\Desktop\SRIVALLI\MIT_ECGClassification\Data'

    records, annotations = Generate_fileList(path=path) 
    print(records[:5], annotations[:5]) 

    emp_signal = Sample_signaReading(records=records) 

    plt.plot(emp_signal[:700])
    plt.xlabel('Samples', fontsize=15)
    plt.ylabel('Voltage (Millivolts)', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('before_preprocessing') 

    emp_signal=denoise(emp_signal)
    plt.plot(emp_signal[:700])
    plt.xlabel('Samples', fontsize=15)
    plt.ylabel('Voltage (Millivolts)', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('after_denoising')

    emp_signal = stats.zscore(emp_signal)
    plt.plot(emp_signal[:700])
    plt.xlabel('Samples', fontsize=15)
    plt.ylabel('Voltage (Millivolts)', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('after_norm')

    X, y, classes, count_classes = Generate_Data(records=records, annotations=annotations, 
                                                                 classes=classes, count_classes=count_classes) 
    
    X, y = np.array(X), np.array(y)
    print(X.shape, y.shape, count_classes)

    i=0
    while y[i]!=0:
        i=i+1

    plt.plot(X[i])
    plt.title('N (Normal beat) ', fontsize=20)
    plt.xlabel('samples',fontsize=15)
    plt.ylabel('Voltage (Millivolts)',fontsize=15)
    plt.savefig('N') 

    Generate_classDistribution(y) 
    Pie_chart(y) 

    X = np.array(X)
    X_reshaped = X.reshape(-1,360,)

    X_df = pd.DataFrame(X_reshaped)
    y_df = pd.DataFrame(y)
    X_new_df = pd.concat([X_df,y_df],axis=1)

    ax=list(range(361))
    X_new_f = X_new_df.set_axis(ax, axis='columns') 

    X_new_df = class_imbalance(X_new_df=X_new_f)
    After_imbalanceRemoval(X_new_df=X_new_df)

    X_new_df.to_csv('balanced_data.csv')

    i=0
    while y[i]!=0:
        i=i+1

    plt.plot(X[i])
    plt.xlabel('samples',fontsize=15)
    plt.ylabel('Voltage (Millivolts)',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig('N')

    i=0
    while y[i]!=1:
        i=i+1

    plt.plot(X[i])
    plt.xlabel('samples',fontsize=15)
    plt.ylabel('Voltage (Millivolts)',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig('L')

    i=0
    while y[i]!=2:
        i=i+1

    plt.plot(X[i])
    plt.xlabel('samples',fontsize=15)
    plt.ylabel('Voltage (Millivolts)',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig('R')

    i=0
    while y[i]!=3:
        i=i+1

    plt.plot(X[i])
    plt.xlabel('samples',fontsize=15)
    plt.ylabel('Voltage (Millivolts)',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig('A')

    i=0
    while y[i]!=4:
        i=i+1

    plt.plot(X[i])
    plt.xlabel('samples',fontsize=15)
    plt.ylabel('Voltage (Millivolts)',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig('V')

    train, test = train_test_split(X_new_df, test_size=0.20, random_state=7)

    print("X_train : ", np.shape(train))
    print("X_test  : ", np.shape(test))

    target_train=train[train.shape[1]-1]
    target_test=test[test.shape[1]-1]
    y_train=to_categorical(target_train)
    y_test=to_categorical(target_test)
    print(np.shape(y_train), np.shape(y_test), y_train[:10], target_train[:10])

    X_train = train.iloc[:,:train.shape[1]-1].values
    X_test = test.iloc[:,:test.shape[1]-1].values
    X_train = X_train.reshape(len(X_train), X_train.shape[1],1)
    X_test = X_test.reshape(len(X_test), X_test.shape[1],1)
    print(np.shape(X_train), np.shape(X_test))

    model = Sequential()

    # Adjust input shape to include a channel dimension for 2D input
    input_shape = (360, 1)  # Assuming the 1D data is reshaped to (360, 1) and adding a channel dimension

    model.add(Conv1D(filters=8, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(strides=2))

    model.add(Conv1D(filters=8, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling1D(strides=2))

    model.add(Conv1D(filters=8, kernel_size=3, strides=1, padding='same', activation='relu'))
    #model.add(MaxPooling1D( strides=2))
    model.add(GlobalAveragePooling1D(name="gap"))

    # model.add(Conv1D(filters=128, kernel_size=17, strides=1, padding='same', activation='relu'))
    # model.add(MaxPooling1D( strides=2))

    #model.add(Flatten())
    #model.add(Dropout(0.5))
    #model.add(Dense(32, activation='relu'))
   # model.add(Dense(50, activation='relu'))
    #model.add(Dense(5, activation='softmax'))
    
  model.add(Dense(8, activation='sigmoid',name="dense_hidden"))
  model.add(Dense(1, activation='sigmoid',name="dense_output"))
  #model.load_weights('/content/drive/MyDrive/vedanth_mtech/ECG_DATA/8_8_8_(8_1)30ep(1).weights.h5')
  # learning_rate=0.00001
  model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
  model.summary()


    #print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy']) 
    history = model.fit(X_train, y_train, batch_size=30, epochs=60, verbose=1, validation_data=(X_test, y_test))

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy',fontsize=20)
    plt.xticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.ylabel('accuracy',fontsize=15)
    plt.xlabel('epoch',fontsize=15)
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy_plot')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss', fontsize=20)
    plt.ylabel('loss', fontsize=15)
    plt.xlabel('epoch', fontsize=15)
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss_plot')
    plt.show()

    model.save('my_model.h5')

    score = model.evaluate(X_test, y_test)

    print('Test Loss:', score[0])
    print('Test accuracy:', score[1])

    from sklearn.metrics import confusion_matrix 

    y_true=[]


    for element in y_test:
        y_true.append(np.argmax(element))
    prediction_proba=model.predict(X_test)
    prediction=np.argmax(prediction_proba,axis=1)
    ax=plt.subplot()
    custCnnConfMat = confusion_matrix(y_true, prediction)
    sns.heatmap(custCnnConfMat, annot=True,fmt='d', cmap='Greens',ax=ax)
    ax.set_title('Confusion Matrix')
    # Check if the number of classes matches the number of tick labels
    if len(classes) == custCnnConfMat.shape[0]:
        ax.xaxis.set_ticklabels(classes)
        ax.yaxis.set_ticklabels(classes)
    else:
        print("Warning: Number of classes does not match the confusion matrix shape.")
    plt.savefig('cm')

    from sklearn.metrics import classification_report

    cf = classification_report(y_true, prediction, target_names=classes,digits=4)
    print(cf)


















    

    
