import pandas as pd
import numpy as np
import sys
import faiss

from rdkit.Chem import AllChem, rdchem, Descriptors, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator as fpg
from rdkit import Chem, DataStructs
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray 
from rdkit.Chem import QED

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances

def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)  # Try to convert SMILES to RDKit Mol
    if mol is None:  # If mol is None, it's an invalid SMILES
        return False
    return True
 
df['is_valid'] = df['SMILES'].apply(is_valid_smiles)

df_cleaned = df[df['is_valid']].drop(columns=['is_valid']).reset_index(drop=True)

print("Original dataframe shape:", df.shape)
print("Cleaned dataframe shape:", df_cleaned.shape)

problematic_smiles = []

def check_smiles(smiles):
    if smiles is None: 
        return None, "Error: NoneType SMILES"
    
    mol = Chem.MolFromSmiles(smiles) 
    if mol:  # If valid molecule
        unconnected_h_count = sum(
            1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1 and atom.GetDegree() == 0
        )
        if unconnected_h_count > 0:
            problematic_smiles.append(smiles) 
            return smiles, f"Error: {unconnected_h_count} unconnected hydrogens"
        else:
            return smiles, "OK"
    else:
        return None, "Error: Invalid SMILES"

df['checked_smiles'], df['flag'] = zip(*df['Canonical_SMILES'].apply(check_smiles))

print(df[['Canonical_SMILES', 'checked_smiles', 'flag']])

if problematic_smiles:
    print("Problematic SMILES causing unconnected hydrogen warnings:")
    for smiles in problematic_smiles:
        print(smiles)
else:
    print("No problematic SMILES found.")

df_cleaned = df[~df['Canonical_SMILES'].isin(problematic_smiles)].reset_index(drop=True)

print(f"Original DataFrame shape: {df.shape}")
print(f"Cleaned DataFrame shape: {df_cleaned.shape}")

print(df_cleaned[['Canonical_SMILES', 'checked_smiles', 'flag']])

problematic_smiles_02 = []

def check_for_bond_stereo_issues(smiles):
    if smiles is None:  # Handle NoneType SMILES early
        return None, "Error: NoneType SMILES"
    
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE and bond.GetStereo() != Chem.BondStereo.STEREONONE:
                # Check if there are single bonds around the double bond with conflicting directions
                begin_atom = bond.GetBeginAtom()
                end_atom = bond.GetEndAtom()

                begin_neighbors = [n for n in begin_atom.GetNeighbors() if mol.GetBondBetweenAtoms(begin_atom.GetIdx(), n.GetIdx()).GetBondType() == Chem.BondType.SINGLE]
                end_neighbors = [n for n in end_atom.GetNeighbors() if mol.GetBondBetweenAtoms(end_atom.GetIdx(), n.GetIdx()).GetBondType() == Chem.BondType.SINGLE]
                
                for n in begin_neighbors:
                    direction = mol.GetBondBetweenAtoms(begin_atom.GetIdx(), n.GetIdx()).GetBondDir()
                    if direction != Chem.BondDir.NONE:
                        # Conflicting bond direction found
                        problematic_smiles_02.append(smiles)
                        return smiles, "Error: Conflicting bond directions"
                
                for n in end_neighbors:
                    direction = mol.GetBondBetweenAtoms(end_atom.GetIdx(), n.GetIdx()).GetBondDir()
                    if direction != Chem.BondDir.NONE:
                        # Conflicting bond direction found
                        problematic_smiles_02.append(smiles)
                        return smiles, "Error: Conflicting bond directions"

        return smiles, "OK"
    else:
        return None, "Error: Invalid SMILES"

df_cleaned['checked_smiles'], df_cleaned['flag'] = zip(*df_cleaned['Canonical_SMILES'].apply(check_for_bond_stereo_issues))

filtered_df = df_cleaned[~df_cleaned['Canonical_SMILES'].isin(problematic_smiles_02)].reset_index(drop=True)

print("Original dataframe shape:", df_cleaned.shape)
print("Filtered dataframe shape:", filtered_df.shape)


def convert_batch_to_float32(df, start, end, column):
    batch = []
    for bitvect in df[column].iloc[start:end]:
        arr = np.zeros((1,), dtype=np.int32)  
        ConvertToNumpyArray(bitvect, arr) 
        batch.append(arr.astype(np.float32))  
    return np.vstack(batch)  # Stack into a single NumPy array

def tanimoto_similarity(dot_products, bit_sum, indices, epsilon=1e-10):
    denominator = bit_sum[:, None] + bit_sum[indices] - dot_products
    tanimoto_sim = dot_products / (denominator + epsilon) 
    return tanimoto_sim

def batch_similarity_filtering(df, column, batch_size=50000, threshold=0.8, nprobe=10):
    d = len(df[column].iloc[0]) 
    nlist = 100  
  
    index_ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist)
    res = faiss.StandardGpuResources()  
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index_ivf)
    first_batch = convert_batch_to_float32(df, 0, batch_size, column)
    gpu_index.train(first_batch)
    gpu_index.nprobe = nprobe
    filtered_indices = []  

    for i in range(0, len(df), batch_size):
        start = i
        end = min(i + batch_size, len(df))
        ecfp_batch = convert_batch_to_float32(df, start, end, column)
        gpu_index.add(ecfp_batch)
        dot_products, indices = gpu_index.search(ecfp_batch, 2)  # Search for the 2 nearest neighbors
        bit_sum = np.sum(ecfp_batch, axis=1)
        tanimoto_sim = tanimoto_similarity(dot_products[:, 1], bit_sum, indices[:, 1])
        mask = tanimoto_sim < threshold
        filtered_indices.extend(np.where(mask)[0] + start)  # Add start to adjust batch indexing
    return filtered_indices


def pass_1_intra_batch_filtering(df, column, batch_size=50000, threshold=0.8, nprobe=10):
    all_filtered_indices = []
    
    for i in range(0, len(df), batch_size):
        # Filter each batch
        batch_indices = batch_similarity_filtering(df.iloc[i:i+batch_size], column, batch_size=batch_size, threshold=threshold, nprobe=nprobe)
        all_filtered_indices.extend([i + idx for idx in batch_indices])  # Adjust indices relative to the full dataset
    return all_filtered_indices

def pass_2_cross_batch_filtering(df, column, batch_size=50000, threshold=0.8, nprobe=10):
    final_filtered_indices = batch_similarity_filtering(df, column, batch_size=batch_size, threshold=threshold, nprobe=nprobe)
    return final_filtered_indices

filtered_indices_pass_1 = pass_1_intra_batch_filtering(df, 'ECFP_BitVect', batch_size=50000, threshold=0.8, nprobe=10)
filtered_df_pass_1 = df.iloc[filtered_indices_pass_1].reset_index(drop=True)
filtered_indices_pass_2 = pass_2_cross_batch_filtering(filtered_df_pass_1, 'ECFP_BitVect', batch_size=50000, threshold=0.8, nprobe=10)
final_filtered_df = filtered_df_pass_1.iloc[filtered_indices_pass_2].reset_index(drop=True)

print("Original dataframe shape:", df.shape)
print("Filtered dataframe shape after two-pass FAISS processing:", final_filtered_df.shape)

