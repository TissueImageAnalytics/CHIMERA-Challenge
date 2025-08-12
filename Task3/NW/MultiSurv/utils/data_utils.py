import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
#from config import *
import glob

import re
from config import * #CLINICAL_CSV, EVENT_COLUMN, TIME_COLUMN, CLINICAL_FEATURES, FOLDS_CSV

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    mappings = {
        "sex": {"Male": 0, "Female": 1},
        "tumor": {"Primary": 0},
        "stage": {"T1HG": 1, "TaHG": 0},
        "grade": {"G2": 2, "G3": 3},
        "reTUR": {"No": 0, "Yes": 1},
        "variant": {"UCC": 0, "UCC + Variant": 1},
        "EORTC": {"High risk": 0, "Highest risk": 1},
        "BRS": {"BRS1": 1, "BRS2": 2, "BRS3": 3}
    }

    df_encoded = df.copy()

    for col, mapping in mappings.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype(str).str.strip()  # Remove extra whitespace
            df_encoded[col] = df_encoded[col].map(mapping)

    return df_encoded

def load_clinical():
    clinical_df = pd.read_csv(CLINICAL_CSV)
    clinical_df['Case_ID'] = clinical_df['Case_ID'].astype(str)
    clinical_df = clinical_df.dropna(subset=[EVENT_COLUMN, TIME_COLUMN])

    cols_to_use = ['Case_ID'] + CLINICAL_FEATURES + [EVENT_COLUMN, TIME_COLUMN]
    clinical_df = clinical_df[cols_to_use]
    clinical_df = clinical_df.rename(columns={EVENT_COLUMN: 'event', TIME_COLUMN: 'duration'})
    clinical_df = encode_clinical_features(clinical_df)

    #clinical_df.to_csv('/home/u1970167/chimera/task3/clinical/features/before_coerce.csv')
    # Convert remaining clinical features to numeric
    clinical_df[CLINICAL_FEATURES] = clinical_df[CLINICAL_FEATURES].apply(pd.to_numeric, errors='coerce')
    #clinical_df.to_csv('/home/u1970167/chimera/task3/clinical/features/after_coerce.csv')

    # Drop rows with any missing values in clinical features
    clinical_df = clinical_df.dropna(subset=CLINICAL_FEATURES)
    clinical_df = clinical_df.reset_index(drop=True)
    return clinical_df

def load_rna_features(case_ids):
    rna_df = pd.read_csv(RNA_CSV)
    rna_df['Case_ID'] = rna_df['Case_ID'].astype(str)

    # Drop rows with missing feature values
    rna_df = rna_df.dropna().reset_index(drop=True)

    # Set index for lookup
    rna_df = rna_df.set_index('Case_ID')

    # Determine feature columns from any row (since 'Case_ID' is index now)
    feature_cols = rna_df.columns.tolist()

    # Warn about missing case IDs
    available_ids = set(rna_df.index)
    missing_ids = [cid for cid in case_ids if cid not in available_ids]
    if missing_ids:
        print(f"Warning: Missing RNA features for case IDs: {missing_ids}")

    # Build aligned feature matrix (fill with zeros if missing)
    rna_features = np.stack([
        rna_df.loc[cid].values if cid in rna_df.index else np.zeros(len(feature_cols), dtype=np.float32)
        for cid in case_ids
    ])

    return rna_features

def load_wsi_features(case_ids, csv_path="path/to/wsi_features.csv"):

    df = pd.read_csv(csv_path)
    df['Slide_ID'] = df['Slide_ID'].astype(str)

    # Remove trailing '_HE' to get Case_ID
    df['Case_ID'] = df['Slide_ID'].str.replace('_HE$', '', regex=True)

    #print(df[['Case_ID', 'Slide_ID']].head(3))

    def extract_slide_num(x):
        parts = x.split('_')
        last = parts[-1].split('.')[0]  # remove file extension if present
        return int(last) if last.isdigit() else 0  # fallback to 0

    df['Slide_Num'] = df['Slide_ID'].apply(extract_slide_num)

    feature_cols = [col for col in df.columns if col.startswith("dim_")]

    selected_features = []
    for cid in case_ids:
        case_df = df[df['Case_ID'] == cid]
        if case_df.empty:
            print(f"[WARNING] No WSI found for case: {cid}")
            selected_features.append(np.zeros(len(feature_cols), dtype=np.float32))
            continue

        # Pick the slide with the lowest Slide_Num
        selected_row = case_df.sort_values("Slide_Num").iloc[0]
        selected_features.append(selected_row[feature_cols].values.astype(np.float32))

    return np.stack(selected_features)

def load_wsi_features_mult(case_ids, csv_path="path/to/wsi_features.csv"):
    """
    Aggregates WSI features per case and returns aligned numpy array.
    
    Args:
        case_ids (list of str): List of Case_IDs to extract features for.
        csv_path (str): Path to the WSI features CSV file.

    Returns:
        np.ndarray: Array of shape [len(case_ids), feature_dim]
    """
    df = pd.read_csv(csv_path)
    df['Slide_ID'] = df['Slide_ID'].astype(str)
    #print('df cols: ', df.columns)
    df['Case_ID'] = df['Slide_ID'].apply(lambda x: str(x).split('_')[0])

    # Average features per case
    feature_cols = [col for col in df.columns if col.startswith("dim_")]
    agg_df = df.groupby("Case_ID")[feature_cols].mean()
    #print('agg_df head: ', agg_df.head(5))

    # Ensure all requested case_ids are present
    missing_ids = [cid for cid in case_ids if cid not in agg_df.index]
    if missing_ids:
        print(f"Warning: Missing WSI features for case IDs: {missing_ids}")

    # Collect aligned features (zero-vector if missing)
    wsi_features = []
    for cid in case_ids:
        if cid in agg_df.index:
            wsi_features.append(agg_df.loc[cid].values)
        else:
            wsi_features.append(np.zeros(len(feature_cols), dtype=np.float32))  # fallback

    return np.stack(wsi_features)

def load_folds():
    #fold_file = os.path.join(FOLDS_DIR, "folds.csv")

    if os.path.exists(FOLDS_CSV):
        folds_df = pd.read_csv(FOLDS_CSV, dtype={'Case_ID': str})
        folds = []
        for i in range(NUM_FOLDS):
            fold_cases = folds_df[folds_df['fold'] == i]['Case_ID'].tolist()
            folds.append(fold_cases)
        print(f"Loaded folds from {FOLDS_CSV}")
    else:
        print(f"{FOLDS_CSV} doesn't exist")    
        
    return folds

def create_folds(events, case_ids):
    #fold_file = os.path.join(FOLDS_DIR, "folds.csv")
    if os.path.exists(FOLDS_CSV):
        folds_df = pd.read_csv(FOLDS_CSV, dtype={'Case_ID': str})
        folds = []
        for i in range(NUM_FOLDS):
            fold_cases = folds_df[folds_df['fold'] == i]['Case_ID'].tolist()
            folds.append(fold_cases)
        print(f"Loaded folds from {FOLDS_CSV}")
    else:
        skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
        fold_assignments = []
        folds = [[] for _ in range(NUM_FOLDS)]
        for fold_idx, (_, val_idx) in enumerate(skf.split(case_ids, events)):
            val_cases = [case_ids[i] for i in val_idx]
            folds[fold_idx] = val_cases
            fold_assignments.extend([(case_ids[i], fold_idx) for i in val_idx])
        folds_df = pd.DataFrame(fold_assignments, columns=['Case_ID', 'fold'])
        #os.makedirs(FOLDS_DIR, exist_ok=True)
        folds_df.to_csv(FOLDS_CSV, index=False)
        print(f"Created and saved folds to {FOLDS_CSV}")
    return folds

def get_feature_dimensionalities(clinical_df):
    clinical_features = clinical_df.drop(columns=['Case_ID', 'duration', 'event'], errors='ignore')
    clinical_dim = clinical_features.shape[1]
    return clinical_dim
