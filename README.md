# Glioma Grading Model (GBM vs LGG)

## Overview
This project trains a deep learning model to classify brain tumor grades (Low-Grade Glioma [LGG] vs Glioblastoma Multiforme [GBM]) using top 8 genetic and clinical features from TCGA data. It includes SMOTE balancing, feature scaling, dropout, LeakyReLU activations, and L2 regularization to improve performance.

## Features Used
IDH1`, `Age_at_diagnosis`, `PIK3CA`, `ATRX`, `PTEN`, `CIC`, `EGFR`, `TP53`

## Pipeline
1. Load TCGA dataset (`TCGA_InfoWithGrade.csv`)
2. Select top 8 features
3. Standardize features with `StandardScaler`
4. Balance classes with `SMOTE`
5. Train/Test split with stratification
6. Train neural network with:
   - 3 hidden layers (64-128-64 neurons)
   - LeakyReLU activations
   - Dropout (0.3–0.4)
   - L2 regularization
7. Evaluate with Accuracy, ROC-AUC, Confusion Matrix, and Classification Report
8. Save trained model and scaler

## Run
```bash
pip install -r requirements.txt
python glioma_grading.py
