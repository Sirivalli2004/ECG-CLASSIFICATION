# 🩺 ECG Arrhythmia Classification using MIT-BIH & PTB-XL Datasets  
### (1D → 2D Scalogram Conversion for Enhanced Accuracy)

This repository focuses on **automatic ECG arrhythmia detection** using both **MIT-BIH Arrhythmia Database** and **PTB-XL Dataset**, integrating **Machine Learning (ML)** and **Deep Learning (DL)** approaches.

The 1D ECG signals are transformed into **2D scalogram images** using the **Continuous Wavelet Transform (CWT)**, enabling CNNs to capture **time–frequency patterns** of heartbeats for improved classification accuracy.

---

## 🚀 Project Highlights
- **Datasets:** MIT-BIH Arrhythmia & PTB-XL (PhysioNet)  
- **Representation:** 1D → 2D Scalogram via Continuous Wavelet Transform (CWT)  
- **Approaches:**  
  - 🧠 **Machine Learning:** Random Forest, XGBoost, SVM, KNN  
  - ⚙️ **Deep Learning:** 2D CNN, ResNet50, EfficientNet, CNN + Attention (CBAM / SE / Transformer)  
- **Goal:** Achieve robust ECG arrhythmia classification with improved accuracy and generalization  
- **Current Work:** Ongoing model enhancement and cross-dataset validation between MIT-BIH and PTB-XL  

---

## 🧩 Workflow

---

## 🩸 Dataset Information

### 🩹 MIT-BIH Arrhythmia Database
| Property | Description |
|-----------|-------------|
| Sampling Frequency | 360 Hz |
| Lead | Lead II |
| Classes (AAMI EC57) | N, S, V, F, Q |

### ❤️ PTB-XL Dataset
| Property | Description |
|-----------|-------------|
| Sampling Frequency | 500 Hz |
| Leads | 12-lead ECG |
| Classes | Normal, MI, STTC, HYP, CD |
| Size | 21 837 records from 18 885 patients |

---

## 🧠 Preprocessing Pipeline
1. **Filtering:** Band-pass (0.5–40 Hz) and wavelet-based denoising  
2. **R-Peak Detection:** Pan-Tompkins / WFDB method  
3. **Beat Segmentation:** 180–200 samples centered at R-peak  
4. **Normalization:** Z-score normalization  
5. **Scalogram Generation:**  
   - Continuous Wavelet Transform (CWT) with *Morlet* wavelet  
   - Saved as **224 × 224** RGB images  

---

## ⚡ Model Architectures

### 🧮 Machine Learning (1D Features)
- Feature extraction: statistical + morphological (mean RR interval, kurtosis, skewness, RMS)
- Models:  
  - Random Forest  
  - XGBoost  
  - Support Vector Machine (SVM)  
  - K-Nearest Neighbors (KNN)

### 🤖 Deep Learning (2D Scalograms)
Example CNN Architecture:
```python
Input: 224x224x3
↓
Conv2D(32, 3x3) → ReLU → MaxPool
↓
Conv2D(64, 3x3) → ReLU → MaxPool
↓
Conv2D(128, 3x3) → ReLU → MaxPool
↓
Flatten → Dense(128, ReLU) → Dropout(0.5)
↓
Dense(5, Softmax)
ECG-Classification/
│
├── data/
│   ├── mitbih/
│   ├── ptbxl/
│   ├── beats/
│   └── scalograms/
│
├── src/
│   ├── preprocessing.py
│   ├── cwt_scalogram.py
│   ├── ml_models.py
│   ├── dl_models.py
│   ├── train_ml.py
│   ├── train_dl.py
│   ├── evaluate.py
│
├── models/
│   └── best_model.h5
│
├── notebooks/
│   ├── mitbih_scalogram.ipynb
│   ├── ptbxl_scalogram.ipynb
│
├── results/
│   ├── confusion_matrix.png
│   ├── training_curves.png
│
├── requirements.txt
├── README.md
└── LICENSE

---

Would you like me to give you the **`requirements.txt`** and a **ready-to-run `cwt_scalogram.py`** script next (for generating 2D scalogram images from ECG signals)?  
That way your GitHub repo will be fully functional and professional.

