Glioblastoma Detection (GBM vs LGG)

Overview
This project uses genetic and clinical features to classify brain tumors into:
- GBM (Glioblastoma Multiforme – high-grade)
- LGG (Low-Grade Glioma)

The model applies a neural network with SMOTE balancing, LeakyReLU activations, and dropout regularization for improved accuracy.

Features Used
IDH1, Age_at_diagnosis, PIK3CA, ATRX, PTEN, CIC, EGFR, TP53

Installation

git clone https://github.com/yourusername/glioblastoma-detection.git
cd glioblastoma-detection
pip install -r requirements.txt
