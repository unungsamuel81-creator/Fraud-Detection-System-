# Fraud Detection System

Machine Learning project to detect credit card fraud.

## Problem
Credit card fraud dataset id highly imbalanced: 99.83% legit vs 0.17% fraud. Accuracy alone is misleading.

## Solution
- EDA to understand imbalance
- StandardScaler for Amount feature
- SMOTE to balance training data: 394 fraud -> 227,451 fraud
- RandomForestClassifier (100 trees)
- Evaluated with Confusion Matrix, Precision, Recall, ROC-AUC

## Results
- Before SMOTE; {0: 227451, 1: 394}
- After SMOTE : {0: 227451, 1: 227451}
- ROC-AUC: ~0.98

## How to Run
'''bash
pip install -r requirements.txt
python fraud_analysis.py