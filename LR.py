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

lr_param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': [0.01, 0.1, 1.0, 10, 100],
    'solver': ['liblinear', 'saga', 'lbfgs'],
    'max_iter': [100, 200, 500]
}

lr_grid_search = GridSearchCV(estimator=LogisticRegression(random_state=42),
                              param_grid=lr_param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=2)
lr_grid_search.fit(X_train_normalized, y_train)

best_lr = lr_grid_search.best_estimator_

calibrated_lr = CalibratedClassifierCV(best_lr, method='isotonic', cv='prefit')
calibrated_lr.fit(X_val_normalized, y_val)  # Use the validation set for calibration

y_test_pred_lr = calibrated_lr.predict(X_test_normalized)
y_test_prob_lr = calibrated_lr.predict_proba(X_test_normalized)[:, 1]

f1_lr = f1_score(y_test, y_test_pred_lr)
roc_auc_lr = roc_auc_score(y_test, y_test_prob_lr)
balanced_acc_lr = balanced_accuracy_score(y_test, y_test_pred_lr)

print(f"Logistic Regression Model Performance on Test Set (with Isotonic Calibration):")
print(f"F1 Score: {f1_lr:.4f}")
print(f"ROC AUC: {roc_auc_lr:.4f}")
print(f"Balanced Accuracy: {balanced_acc_lr:.4f}")
