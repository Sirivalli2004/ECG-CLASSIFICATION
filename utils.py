import os 
import sys 
import math 

import csv 
import pywt
import glob 
import random 


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.utils import resample


def Generate_fileList(path=None):
    records = []
    annotations = []
    filenames = os.listdir(path)

    for f in filenames:
        filename, ext = os.path.splitext(f)
        if ext == '.csv':
            records.append(os.path.join(path, f))
        else:
            annotations.append(os.path.join(path, f))

    return records, annotations

def Sample_signaReading(records):
    emp_signal=[]

    with open(records[6],'r') as f:
        reader = csv.reader(f, delimiter=',',quotechar='|')
        row_index= -1
        for row in reader:
            if row_index >=0:
                emp_signal.insert(row_index, int(row[1]))
            row_index += 1
    
    return emp_signal

def denoise(data):
    w = pywt.Wavelet('sym4')
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    threshold = 0.04 # Threshold for filtering

    coeffs = pywt.wavedec(data, 'sym4', level=maxlev)
  #  print(len(coeffs))
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))

    datarec = pywt.waverec(coeffs, 'sym4')
    return datarec 

def Generate_classDistribution(y):
    y_df = pd.DataFrame(y)
    #y_df.head()
    per_class = y_df[y_df.shape[1]-1].value_counts()
    #print(per_class)
    plt.figure(figsize=(30,10))
    my_circle=plt.Circle( (0,0), 0.7, color='white')
    plt.pie(per_class, labels=['N', 'L', 'R', 'V', 'A'], colors=['#2085ec','#72b4eb','#0a417a','#8464a0','#cea9bc'],autopct='%1.1f%%', textprops={'fontsize':15})
    p=plt.gcf()
    p.gca().add_artist(my_circle)
    plt.show()
    p.savefig('Before_piechart')


def Pie_chart(y):
    y_df = pd.DataFrame(y)
    #y_df.head()
    per_class = y_df[y_df.shape[1]-1].value_counts()
    #print(per_class)
    plt.figure(figsize=(30,10))
    my_circle=plt.Circle( (0,0), 0.7, color='white')
    plt.pie(per_class, labels=['N', 'L', 'R', 'V', 'A'], colors=['#2085ec','#72b4eb','#0a417a','#8464a0','#cea9bc'],autopct='%1.1f%%', textprops={'fontsize':15})
    plt.show()
    plt.savefig('Before_piechart') 

def class_imbalance(X_new_df):
    df_0=(X_new_df[X_new_df[X_new_df.shape[1]-1]==0]).sample(n=7000,random_state=42)
    df_1=X_new_df[X_new_df[X_new_df.shape[1]-1]==1]
    df_2=X_new_df[X_new_df[X_new_df.shape[1]-1]==2]
    df_3=X_new_df[X_new_df[X_new_df.shape[1]-1]==3]
    df_4=X_new_df[X_new_df[X_new_df.shape[1]-1]==4]

    df_1_upsample=resample(df_1,replace=True,n_samples=7000,random_state=125)
    df_2_upsample=resample(df_2,replace=True,n_samples=7000,random_state=77)
    df_3_upsample=resample(df_3,replace=True,n_samples=7000,random_state=103)
    df_4_upsample=resample(df_4,replace=True,n_samples=7000,random_state=59)

    X_new_df=pd.concat([df_0,df_1_upsample,df_2_upsample,df_3_upsample,df_4_upsample])
    return X_new_df 

def After_imbalanceRemoval(X_new_df):
    per_class = X_new_df[X_new_df.shape[1]-1].value_counts()
    #print(per_class)
    plt.figure(figsize=(30,10))
    my_circle=plt.Circle( (0,0), 0.7, color='white')
    plt.pie(per_class, labels=['N', 'L', 'R', 'V', 'A'], colors=['#2085ec','#72b4eb','#0a417a','#8464a0','#cea9bc'],autopct='%1.1f%%',textprops={'fontsize':15})
    plt.show()
    plt.savefig('after_piechart')