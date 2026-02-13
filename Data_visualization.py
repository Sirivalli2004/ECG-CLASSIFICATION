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


if __name__ == '__main__':

    os.system('cls' if os.name == 'nt' else 'clear')

    # Load your dataset (replace with your actual file path)
    data = pd.read_csv(os.path.join('Data','100.csv')) # e.g., 'ecg_data.csv'
    print(data.columns)
    signal_mlii = data[data.columns[1]].values  # Primary signal from MLII lead
    signal_v5 = data[data.columns[2]].values     # Optional: V5 lead (can be used as a second channel)

    # # For this example, we'll use MLII only; adjust if you want both leads
    signal = signal_mlii
    sample_rate = 360  # Assuming MIT-BIH standard sampling rate
    print(f"Loaded signal with {len(signal)} samples")