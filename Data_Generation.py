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

from scipy import stats

from utils import denoise

def Generate_Data(records, annotations, classes, count_classes):
    X = []
    y = []
    window_size=180
    for r in range(0,len(records)):
        signals = [ ]

        with open(records[r],'r') as f:
            reader = csv.reader(f, delimiter=',',quotechar='|')
            row_index= -1
            for row in reader:
                if row_index >=0:
                    signals.insert(row_index, int(row[1]))
                row_index += 1

        signals = denoise(signals)
        signals = stats.zscore(signals)

        with open(annotations[r], 'r') as fileID:
            data = fileID.readlines()
            beat = list()

            for d in range(1, len(data)):
                splitted = data[d].split(' ')
                splitted = filter(None, splitted)
                next(splitted)
                pos = int(next(splitted))
                arrhythmia_type = next(splitted)

                if(arrhythmia_type in classes):
                    arrhythmia_index = classes.index(arrhythmia_type)
                    count_classes[arrhythmia_index] += 1
                if(window_size <= pos and pos < (len(signals) - window_size)):
                    beat = signals[pos-window_size:pos+window_size]
                    X.append(beat)
                    y.append(arrhythmia_index)

    return X, y, classes, count_classes
