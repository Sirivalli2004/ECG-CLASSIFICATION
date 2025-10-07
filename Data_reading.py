import os 
import sys 
import math 

import glob 
import wfdb 
import shutil 
import warnings


import numpy as np 
import pandas as pd 

 
from tqdm import tqdm 
from lightgbm import LGBMClassifier

from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score


from feature_compution import extract_features
from performance_measure_estimation import split_data, find_optimal_threshold 


warnings.filterwarnings('ignore', category=FutureWarning)


dx_dict = {
    '426783006': 'SNR', # Normal sinus rhythm
    '164889003': 'AF', # Atrial fibrillation
    '270492004': 'IAVB', # First-degree atrioventricular block
    '164909002': 'LBBB', # Left bundle branch block
    '713427006': 'RBBB', # Complete right bundle branch block
    '59118001': 'RBBB', # Right bundle branch block
    '284470004': 'PAC', # Premature atrial contraction
    '63593006': 'PAC', # Supraventricular premature beats
    '164884008': 'PVC', # Ventricular ectopics
    '429622005': 'STD', # ST-segment depression
    '164931005': 'STE', # ST-segment elevation
}
classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']


def combine_dataset(data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for sub_dir in glob(os.path.join(data_dir, 'g*')):
        hea_files = glob(os.path.join(sub_dir, '*.hea'))
        mat_files = glob(os.path.join(sub_dir, '*.mat'))

        for file_path in hea_files + mat_files:
            base_name = os.path.basename(file_path)
            dest_path = os.path.join(output_dir, base_name)
            shutil.copy(file_path, dest_path)
    
    print(f"Dataset files copied to {output_dir}")

def gen_reference_csv(data_dir, reference_csv='reference.csv'):
    if not os.path.exists(reference_csv):
        recordpaths = glob.glob(os.path.join(data_dir, '*.hea'))
        results = []  

        for recordpath in recordpaths:
            patient_id = recordpath.split('/')[-1][:-4]
            _, meta_data = wfdb.rdsamp(recordpath[:-4])
            sample_rate = meta_data['fs']
            signal_len = meta_data['sig_len']
            age = meta_data['comments'][0]
            sex = meta_data['comments'][1]
            dx = meta_data['comments'][2]
            age = age[5:] if age.startswith('Age: ') else np.NaN
            sex = sex[5:] if sex.startswith('Sex: ') else 'Unknown'
            dx = dx[4:] if dx.startswith('Dx: ') else ''
            results.append([patient_id, sample_rate, signal_len, age, sex, dx])
        
        df = pd.DataFrame(data=results, columns=['patient_id', 'sample_rate', 'signal_len', 'age', 'sex', 'dx'])
        df.sort_values('patient_id').to_csv(reference_csv, index=None)

def gen_label_csv(label_csv, reference_csv, dx_dict, classes):
    if not os.path.exists(label_csv): 
        results = [] 
        df_reference = pd.read_csv(reference_csv)
        for _, row in df_reference.iterrows(): 
            patient_id = row['patient_id'] 
            dxs = [dx_dict.get(code, '') for code in row['dx'].split(',')] 
            labels = [0] * 9 
            for idx, label in enumerate(classes): 
                if label in dxs:
                    labels[idx] = 1
            
            # print(labels, patient_id) 
            # break 
            
            results.append([patient_id] + labels) 

        df = pd.DataFrame(data=results, columns=['patient_id'] + classes) 
        n = len(df)
        folds = np.zeros(n, dtype=np.int8)
        for i in range(10): 
            start = int(n * i / 10)
            end = int(n * (i + 1) / 10)
            folds[start:end] = i + 1
        
        df['fold'] = np.random.permutation(folds)
        columns = df.columns
        df['keep'] = df[classes].sum(axis=1)
        df = df[df['keep'] > 0]
        df[columns].to_csv(label_csv, index=None) 

def generate_features_csv(features_csv, data_dir, patient_ids):
    print('Generating expert features...')
    ecg_features = []
    for patient_id in tqdm(patient_ids):
        ecg_data, _ = wfdb.rdsamp(os.path.join(data_dir, patient_id))
        ecg_features.append(extract_features(ecg_data))

    df = pd.DataFrame(ecg_features, index=patient_ids)
    df.index.name = 'patient_id' 
    df.to_csv(features_csv)
    return df




if __name__ == '__main__':

    os.system('cls' if os.name == 'nt' else 'clear') 

    data_dir = 'WFDB' 
    labels_csv = 'labels.csv'
    reference_csv = 'reference.csv'
    features_csv =  'features.csv'
    classifier = 'all' 



    classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
    
    gen_reference_csv(data_dir=data_dir, reference_csv=reference_csv)
    gen_label_csv(labels_csv, reference_csv, dx_dict, classes)

    df_labels = pd.read_csv(labels_csv)
    patient_ids = df_labels['patient_id'].tolist()

    if not os.path.exists(features_csv):
        df_X = generate_features_csv(features_csv, data_dir, patient_ids)
    else:
        df_X = pd.read_csv(features_csv)
    df_X = df_X.merge(df_labels[['patient_id', 'fold']], on='patient_id')

    seed = 42
    train_folds, val_folds, test_folds = split_data(seed=seed)
    feature_cols = df_X.columns[1:-1] # remove patient id and fold

    X_train = df_X[df_X['fold'].isin(train_folds)][feature_cols].to_numpy()
    X_val = df_X[df_X['fold'].isin(val_folds)][feature_cols].to_numpy()
    X_test = df_X[df_X['fold'].isin(test_folds)][feature_cols].to_numpy()

    y_train = df_labels[df_labels['fold'].isin(train_folds)][classes].to_numpy()
    y_val = df_labels[df_labels['fold'].isin(val_folds)][classes].to_numpy()
    y_test = df_labels[df_labels['fold'].isin(test_folds)][classes].to_numpy() 

    if classifier == 'all':
        classifiers = ['LR', 'RF', 'LGB', 'MLP']
    else:
        classifiers = [classifier] 
    
    for classifier in classifiers:
        if classifier == 'LR':
            model = LogisticRegression(solver='lbfgs', max_iter=1000)
        elif classifier == 'RF':
            model = RandomForestClassifier(n_estimators=300, max_depth=10)
        elif classifier == 'LGB':
            model = LGBMClassifier(n_estimators=100)
        else:
            model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
        if classifier != 'MLP':
            model = OneVsRestClassifier(model)

        print(f'Start training {classifier}...')
        model.fit(X_train, y_train)
    
        y_val_scores = model.predict_proba(X_val)
        y_test_scores = model.predict_proba(X_test)
        
        f1s = []
        thresholds = []
        print('Finding optimal thresholds on validation dataset...')

        for i in range(len(classes)):
            # find optimal threshold on validation dataset
            y_val_score = y_val_scores[:, i]
            threshold = find_optimal_threshold(y_val[:, i], y_val_score)
            # apply optimal threshold to test dataset
            y_test_score = y_test_scores[:, i]
            y_test_pred = y_test_score > threshold
            f1 = f1_score(y_test[:, i], y_test_pred)
            thresholds.append(threshold)
            f1s.append(f1)
        np.set_printoptions(precision=3)
        print(f'{classifier} F1s:', f1s)
        print('Avg F1:', np.mean(f1s))
    






