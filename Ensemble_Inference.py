import pandas as pd
import numpy as np
import rdkit as rd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import sys

from rdkit import Chem, DataStructs
from rdkit.Chem import rdchem, Descriptors, rdMolDescriptors, AllChem, rdFingerprintGenerator
from rdkit.Chem import rdFingerprintGenerator as fpg

from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, StratifiedKFold, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, classification_report, brier_score_loss, confusion_matrix 

from xgboost import XGBClassifier

X_ecfp = np.array(df_01['ECFP_BitVect'].tolist())
X_tpsa = df_01['TPSA'].values.reshape(-1, 1)
X_logp = df_01['LogP'].values.reshape(-1, 1)
X_combined = np.hstack([X_ecfp, X_tpsa, X_logp])

y = df_01['DILIst_Classification']
X_train_val, X_test, y_train_val, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_val_normalized = scaler.transform(X_val)
X_test_normalized = scaler.transform(X_test)

f1_scores = np.array([0.77, 0.76, 0.71, 0.77])  # F1 scores for RF, SVM, LR, XGB
roc_auc_scores = np.array([0.87, 0.84, 0.74, 0.85])  # ROC AUC scores
balanced_acc_scores = np.array([0.81, 0.80, 0.76, 0.81])  # Balanced accuracy
ece_scores = np.array([0.03, 0.06, 0.04, 0.05])  # Expected Calibration Error (lower is better)

f1_norm = f1_scores / np.max(f1_scores)
roc_auc_norm = roc_auc_scores / np.max(roc_auc_scores)
balanced_acc_norm = balanced_acc_scores / np.max(balanced_acc_scores)

ece_norm = (1 - (ece_scores / np.max(ece_scores)))

aggregated_scores = (f1_norm + roc_auc_norm + balanced_acc_norm + ece_norm) / 4

weights = aggregated_scores / np.sum(aggregated_scores)

print("Model Weights based on aggregated performance metrics:", weights)

final_results = []

chunk_size = 10000 
num_rows = df_02.shape[0]

for start in range(0, num_rows, chunk_size):
    end = min(start + chunk_size, num_rows)
    
    X_combined_normalized_chunk = np.load(f"{save_dir}/normalized_chunk_{start}_{end}.npy")
    
    rf_probs = rf_model.predict_proba(X_combined_normalized_chunk)[:, 1]
    svm_probs = svm_model.predict_proba(X_combined_normalized_chunk)[:, 1]
    lr_probs = lr_model.predict_proba(X_combined_normalized_chunk)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_combined_normalized_chunk)[:, 1]

    ensemble_probs = (weights[0] * rf_probs +
                      weights[1] * svm_probs +
                      weights[2] * lr_probs +
                      weights[3] * xgb_probs)

    ensemble_pred = (ensemble_probs >= 0.5).astype(int)

    chunk_results = pd.DataFrame({
        'DILI_Risk_Prediction': ensemble_pred,
        'DILI_Risk_Probability': ensemble_probs
    })

    final_results.append(chunk_results)

    chunk_results.to_csv(f"{result_dir}/predictions_chunk_{start}_{end}.csv", index=False)

final_results_df = pd.concat(final_results, ignore_index=True)

df_02['DILI_Risk_Prediction'] = final_results_df['DILI_Risk_Prediction']
df_02['DILI_Risk_Probability'] = final_results_df['DILI_Risk_Probability']
