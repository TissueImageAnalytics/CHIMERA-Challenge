import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from config import *
import glob

import re
from config import MIXED_COLS, CLINICAL_CSV, EVENT_COLUMN, TIME_COLUMN, CLINICAL_FEATURES

def convert_mixed_column_to_numeric(series):
    def parse_value(val):
        match = re.match(r"(\d+)([a-zA-Z]*)", str(val))
        if match:
            base = int(match.group(1))
            suffix = match.group(2)
            if suffix:
                suffix_value = (ord(suffix.lower()) - ord('a') + 1) / 10
                return base + suffix_value
            else:
                return float(base)
        return None  # Or np.nan if you want to filter it later

    return series.apply(parse_value)

def load_clinical():
    clinical_df = pd.read_csv(CLINICAL_CSV)
    clinical_df['Case_ID'] = clinical_df['Case_ID'].astype(str)
    clinical_df = clinical_df.dropna(subset=[EVENT_COLUMN, TIME_COLUMN])

    cols_to_use = ['Case_ID'] + CLINICAL_FEATURES + [EVENT_COLUMN, TIME_COLUMN]
    clinical_df = clinical_df[cols_to_use]
    clinical_df = clinical_df.rename(columns={EVENT_COLUMN: 'event', TIME_COLUMN: 'duration'})

    # Convert mixed columns first
    for col in MIXED_COLS:
        if col in clinical_df.columns:
            clinical_df[col] = convert_mixed_column_to_numeric(clinical_df[col])

    # Convert remaining clinical features to numeric
    clinical_df[CLINICAL_FEATURES] = clinical_df[CLINICAL_FEATURES].apply(pd.to_numeric, errors='coerce')

    # Drop rows with any missing values in clinical features
    clinical_df = clinical_df.dropna(subset=CLINICAL_FEATURES)
    clinical_df = clinical_df.reset_index(drop=True)
    return clinical_df

def load_mri_features(case_ids):
    feats = []
    missing = []

    mod_str = "_".join(MODALITIES)
    feature_dir = os.path.join(MRI_FEATURE_DIR, f"ROI_{APPLY_ROI}_{mod_str}")

    for case_id in case_ids:
        try:
            if COMBINE_MODALITIES:
                # Match files like case1_001.npy, case1_abc.npy, etc.
                pattern = os.path.join(feature_dir, f"{case_id}_*.npy")
                matched_files = sorted(glob.glob(pattern))
                if matched_files:
                    feat = np.load(matched_files[0])  # pick the first match
                else:
                    raise FileNotFoundError(f"No combined MRI feature found for {case_id}")
            else:
                # Look for per-modality files like case1_001_t2w.npy, case1_xyz_adc.npy, etc.
                # Need to match on all modalities for the same scan
                pattern = os.path.join(feature_dir, f"{case_id}_*_{MODALITIES[0]}.npy")
                base_paths = sorted(glob.glob(pattern))

                selected_feat = None
                for base_path in base_paths:
                    base_prefix = base_path.rsplit("_", 1)[0]  # remove "_t2w.npy"
                    try:
                        modality_feats = []
                        for mod in MODALITIES:
                            mod_file = f"{base_prefix}_{mod}.npy"
                            if os.path.isfile(mod_file):
                                modality_feats.append(np.load(mod_file))
                            else:
                                raise FileNotFoundError
                        selected_feat = np.concatenate(modality_feats)
                        break  # success
                    except FileNotFoundError:
                        continue

                if selected_feat is not None:
                    feat = selected_feat
                else:
                    raise FileNotFoundError(f"No complete modality set for {case_id}")

            feats.append(feat)

        except Exception as e:
            print(f" Error loading MRI features for {case_id}: {e}")
            missing.append(case_id)
            feats.append(np.zeros((M_FEATURE_DIM,), dtype=np.float32))  # fallback

    if missing and VERBOSE:
        print(f" Missing MRI features for {len(missing)} cases: {missing}")

    return np.stack(feats)

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

def load_wsi_features(case_ids, csv_path="path/to/wsi_features.csv"):
    """
    Select a single WSI feature per case using Slide_ID naming convention.
    Picks the WSI with the lowest numeric suffix (e.g., Case001_001 over Case001_002).

    Args:
        case_ids (list of str): List of Case_IDs to extract features for.
        csv_path (str): Path to the WSI features CSV file.

    Returns:
        np.ndarray: Array of shape [len(case_ids), feature_dim]
    """
    df = pd.read_csv(csv_path)
    df['Slide_ID'] = df['Slide_ID'].astype(str)
    df['Case_ID'] = df['Slide_ID'].apply(lambda x: x.split('_')[0])

    # Extract slide number (e.g., Case001_003 → 3)
    df['Slide_Num'] = df['Slide_ID'].apply(
        lambda x: int(x.split('_')[-1].split('.')[0]) if '_' in x else 0
    )

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

def load_radiomic_features(case_ids, csv_path):
    df = pd.read_csv(csv_path)
    df['Case_ID'] = df['Case_ID'].astype('str')
    df = df[df['Case_ID'].isin(case_ids)].sort_values('Case_ID')
    features = df.drop(columns=['Case_ID']).values
    return features

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
