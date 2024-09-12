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

svm_param_grid = {
    'C': [0.01, 0.1, 1.0, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

svm_grid_search = GridSearchCV(estimator=SVC(probability=True, random_state=42),
                               param_grid=svm_param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=2)
svm_grid_search.fit(X_train_normalized, y_train)

best_svm = svm_grid_search.best_estimator_

calibrated_svm = CalibratedClassifierCV(best_svm, method='isotonic', cv='prefit')
calibrated_svm.fit(X_val_normalized, y_val)  # Use the validation set for calibration

y_test_pred_svm = calibrated_svm.predict(X_test_normalized)
y_test_prob_svm = calibrated_svm.predict_proba(X_test_normalized)[:, 1]

f1_svm = f1_score(y_test, y_test_pred_svm)
roc_auc_svm = roc_auc_score(y_test, y_test_prob_svm)
balanced_acc_svm = balanced_accuracy_score(y_test, y_test_pred_svm)

print(f"SVM Model Performance on Test Set (with Isotonic Calibration):")
print(f"F1 Score: {f1_svm:.4f}")
print(f"ROC AUC: {roc_auc_svm:.4f}")
print(f"Balanced Accuracy: {balanced_acc_svm:.4f}")
