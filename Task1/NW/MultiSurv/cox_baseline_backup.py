import os
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
from utils.data_utils import load_clinical, create_or_load_folds
from config import *

# Load clinical data using your utility
clinical_df = load_clinical()

# Subset only clinical features + Case_ID + event and duration columns
cols_to_use = ['Case_ID'] + CLINICAL_FEATURES + ['event', 'duration']
clinical_df = clinical_df[cols_to_use]

# Extract event and duration arrays for fold creation
events = clinical_df['event'].values
case_ids = clinical_df['Case_ID'].astype(str).tolist()

# Use your utility to load or create folds consistently
folds_case_ids = create_or_load_folds(events, case_ids)

# Convert folds from case_ids to train/test indices
fold_indices = []
for val_case_ids in folds_case_ids:
    val_idx = clinical_df[clinical_df['Case_ID'].isin(val_case_ids)].index.tolist()
    train_idx = clinical_df[~clinical_df['Case_ID'].isin(val_case_ids)].index.tolist()
    fold_indices.append((train_idx, val_idx))

def evaluate_cox_model(df, feature_cols, fold_indices, label, normalise=True):
    c_indices = []
    for fold, (train_idx, val_idx) in enumerate(fold_indices):
        X_train = df.loc[train_idx, feature_cols].copy()
        X_val = df.loc[val_idx, feature_cols].copy()

        if normalise:
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=train_idx)
            X_val = pd.DataFrame(scaler.transform(X_val), columns=feature_cols, index=val_idx)

        train_df = X_train.copy()
        train_df['duration'] = df.loc[train_idx, 'duration']
        train_df['event'] = df.loc[train_idx, 'event']

        val_df = X_val.copy()
        val_df['duration'] = df.loc[val_idx, 'duration']
        val_df['event'] = df.loc[val_idx, 'event']

        cph = CoxPHFitter()
        cph.fit(train_df, duration_col='duration', event_col='event')

        partial_hazards = cph.predict_partial_hazard(val_df)
        c_index = concordance_index(val_df['duration'], -partial_hazards, val_df['event'])
        c_indices.append(c_index)

        if VERBOSE:
            print(f"{label} - Fold {fold+1} C-index: {c_index:.4f}")

    avg_cindex = np.mean(c_indices)
    std_cindex = np.std(c_indices)
    print(f"\nAverage C-index across all folds: {avg_cindex:.4f} ± {std_cindex:.4f}")
    return c_indices, avg_cindex

# Run evaluation on clinical features only
c_indices, avg_c_index = evaluate_cox_model(
    clinical_df, CLINICAL_FEATURES, fold_indices, label="Clinical Only", normalise=True
)

# Optionally, save results to a CSV file
results_df = pd.DataFrame({
    'Fold': [f'Fold {i+1}' for i in range(len(c_indices))],
    'C-index': c_indices
})
results_df.loc[len(c_indices)] = ['Average', avg_c_index]

os.makedirs(OUTPUT_DIR, exist_ok=True)
results_csv_path = os.path.join(OUTPUT_DIR, "cox_clinical_results.csv")
results_df.to_csv(results_csv_path, index=False)

print(f"Results saved to: {results_csv_path}")
