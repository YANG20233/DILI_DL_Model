import h5py
import pandas as pd
import numpy as np
import rdkit as rd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

from rdkit import Chem, DataStructs
from rdkit.Chem import rdchem, Descriptors, rdMolDescriptors, AllChem, rdFingerprintGenerator
from rdkit.Chem import rdFingerprintGenerator as fpg

from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean, pdist
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, brier_score_loss, confusion_matrix 
from sklearn.pipeline import Pipeline

SMILES_df = pd.read_csv('DILIST_SMILES.csv')
ZINC_df = pd.read_csv('ZINC20_qed_0.5.csv')

# Function to generate the bitvectors as Tanimoto's similarity requires bit vectors as input.
# Run this code everytime the file is loaded again
def generate_bitvector(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    circular_bit_fp = fpg.GetMorganGenerator(radius=2).GetFingerprint(molecule)
    return circular_bit_fp

# Apply the function to generate the bit vector for each molecule in your DataFrames
ZINC_df['ECFP_BitVect'] = ZINC_df['Canonical_SMILES'].apply(generate_bitvector)
SMILES_df['ECFP_BitVect'] = SMILES_df['Canonical_SMILES'].apply(generate_bitvector)

# Prepare the feature matrix
bit_matrix = np.array([list(fp.ToBitString()) for fp in SMILES_df['ECFP_BitVect']], dtype=np.int8)
tpsa_logp_matrix = SMILES_df[['TPSA', 'LogP']].values
combined_matrix = np.hstack([bit_matrix, tpsa_logp_matrix])

# Target labels (DILI annotations)
y = SMILES_df['DILIst_Classification'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(combined_matrix, y, test_size=0.3, random_state=42)

# Train the Random Forest model for generating tree-based embeddings
rf_for_embedding = RandomForestClassifier(n_estimators=100, random_state=42)
rf_for_embedding.fit(X_train, y_train)

# Get the leaf indices for each sample
train_leaf_indices = rf_for_embedding.apply(X_train)
test_leaf_indices = rf_for_embedding.apply(X_test)

# One-hot encode the leaf indices
encoder = OneHotEncoder(categories='auto')
train_leaves = encoder.fit_transform(train_leaf_indices).toarray()  # Convert to dense array
test_leaves = encoder.transform(test_leaf_indices).toarray()  # Convert to dense array

# Debugging output to verify shapes
print("X_train shape:", X_train.shape)
print("Train leaves shape:", train_leaves.shape)

# Combine the original features with the one-hot encoded leaf indices
X_train_combined = np.hstack([X_train, train_leaves])
X_test_combined = np.hstack([X_test, test_leaves])

# Initialize and train the Logistic Regression model on the combined features
log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
log_reg_model.fit(X_train_combined, y_train)

# Evaluate the model
y_pred_proba = log_reg_model.predict_proba(X_test_combined)[:, 1]
y_pred = log_reg_model.predict(X_test_combined)

# Print evaluation metrics
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'AUC-ROC: {roc_auc_score(y_test, y_pred_proba)}')
print(classification_report(y_test, y_pred))

# Combine ECFP bit vectors with molecular descriptors from ZINC_df
zinc_bit_matrix = np.array([list(fp.ToBitString()) for fp in ZINC_df['ECFP_BitVect']], dtype=np.int8)
zinc_tpsa_logp_matrix = ZINC_df[['TPSA', 'LogP']].values
zinc_combined_matrix = np.hstack([zinc_bit_matrix, zinc_tpsa_logp_matrix])

# Use the trained Random Forest model to generate tree-based embeddings for ZINC_df
zinc_leaf_indices = rf_for_embedding.apply(zinc_combined_matrix)

# One-hot encode the leaf indices for ZINC_df
zinc_leaves = encoder.transform(zinc_leaf_indices).toarray()  # Convert to dense array

# Combine the original features with the one-hot encoded leaf indices for ZINC_df
zinc_combined_features = np.hstack([zinc_combined_matrix, zinc_leaves])

# Predict the probabilities using the trained logistic regression model
zinc_pred_proba = log_reg_model.predict_proba(zinc_combined_features)[:, 1]

# Add the probabilities to the ZINC_df for easier analysis
ZINC_df['Probability_Bin'] = zinc_pred_proba
