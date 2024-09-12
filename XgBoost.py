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

xgb_param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

xgb_grid_search = GridSearchCV(estimator=XGBClassifier(eval_metric='logloss', random_state=42),
                               param_grid=xgb_param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=2)
xgb_grid_search.fit(X_train_normalized, y_train)

best_xgb = xgb_grid_search.best_estimator_

calibrated_xgb = CalibratedClassifierCV(best_xgb, method='isotonic', cv='prefit')
calibrated_xgb.fit(X_val_normalized, y_val)  # Use the validation set for calibration

y_test_pred_xgb = calibrated_xgb.predict(X_test_normalized)
y_test_prob_xgb = calibrated_xgb.predict_proba(X_test_normalized)[:, 1]

f1_xgb = f1_score(y_test, y_test_pred_xgb)
roc_auc_xgb = roc_auc_score(y_test, y_test_prob_xgb)
balanced_acc_xgb = balanced_accuracy_score(y_test, y_test_pred_xgb)

print(f"XGBoost Model Performance on Test Set (with Isotonic Calibration):")
print(f"F1 Score: {f1_xgb:.4f}")
print(f"ROC AUC: {roc_auc_xgb:.4f}")
print(f"Balanced Accuracy: {balanced_acc_xgb:.4f}")
