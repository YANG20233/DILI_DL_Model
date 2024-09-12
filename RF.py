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

rf_param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

rf_grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=rf_param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)
rf_grid_search.fit(X_train_normalized, y_train)

best_rf = rf_grid_search.best_estimator_

calibrated_rf = CalibratedClassifierCV(best_rf, method='isotonic', cv='prefit')
calibrated_rf.fit(X_val_normalized, y_val)

y_test_pred_rf = calibrated_rf.predict(X_test_normalized)
y_test_prob_rf = calibrated_rf.predict_proba(X_test_normalized)[:, 1]

f1_rf = f1_score(y_test, y_test_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_test_prob_rf)
balanced_acc_rf = balanced_accuracy_score(y_test, y_test_pred_rf)

print(f"Random Forest Model Performance on Test Set (with Isotonic Calibration):")
print(f"F1 Score: {f1_rf:.4f}")
print(f"ROC AUC: {roc_auc_rf:.4f}")
print(f"Balanced Accuracy: {balanced_acc_rf:.4f}")
