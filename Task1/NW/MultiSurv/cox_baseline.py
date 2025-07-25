import os
import pickle
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
from utils.data_utils import load_clinical, load_folds
from config import *

os.makedirs(COX_BASELINE_DIR, exist_ok=True)

# Load clinical data
clinical_df = load_clinical()

# Columns to use
cols_to_use = ['Case_ID'] + CLINICAL_FEATURES + ['event', 'duration']
clinical_df = clinical_df[cols_to_use]

# Load folds from CSV using your function
folds_case_ids = load_folds()

# Convert folds from Case_IDs to train/test indices
fold_indices = []
for val_case_ids in folds_case_ids:
    val_idx = clinical_df[clinical_df['Case_ID'].isin(val_case_ids)].index.tolist()
    train_idx = clinical_df[~clinical_df['Case_ID'].isin(val_case_ids)].index.tolist()
    fold_indices.append((train_idx, val_idx))


def train_cv_cox_model(df, feature_cols, fold_indices):
    """
    Train CoxPH models on each fold and save models + scalers.
    Evaluate on validation sets and print fold-wise & average C-index.
    """
    c_indices = []
    for fold, (train_idx, val_idx) in enumerate(fold_indices):
        X_train = df.loc[train_idx, feature_cols].copy()
        X_val = df.loc[val_idx, feature_cols].copy()

        if SCALE_DATA:
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=train_idx)
            X_val = pd.DataFrame(scaler.transform(X_val), columns=feature_cols, index=val_idx)
        else:
            scaler = None

        train_df = X_train.copy()
        train_df['duration'] = df.loc[train_idx, 'duration']
        train_df['event'] = df.loc[train_idx, 'event']

        val_df = X_val.copy()
        val_df['duration'] = df.loc[val_idx, 'duration']
        val_df['event'] = df.loc[val_idx, 'event']

        cph = CoxPHFitter()
        cph.fit(train_df, duration_col='duration', event_col='event')

        # Save model and scaler for this fold
        model_path = os.path.join(COX_BASELINE_DIR, f"cox_model_fold{fold+1}.pkl")
        scaler_path = os.path.join(COX_BASELINE_DIR, f"scaler_fold{fold+1}.pkl")

        with open(model_path, 'wb') as f:
            pickle.dump(cph, f)
        if scaler:
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)

        # Evaluate on validation set
        partial_hazards = cph.predict_partial_hazard(val_df)
        c_index = concordance_index(val_df['duration'], -partial_hazards, val_df['event'])
        c_indices.append(c_index)

        if VERBOSE:
            print(f"Train - Fold {fold+1} C-index: {c_index:.4f}")

    avg_cindex = np.mean(c_indices)
    std_cindex = np.std(c_indices)
    print(f"\nAverage C-index across all folds: {avg_cindex:.4f} ± {std_cindex:.4f}")

    # Save results CSV
    results_df = pd.DataFrame({
        'Fold': [f'Fold {i+1}' for i in range(len(c_indices))],
        'C-index': c_indices
    })
    results_df.loc[len(c_indices)] = ['Average', avg_cindex]
    results_csv_path = os.path.join(COX_BASELINE_DIR, "cox_clinical_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Training results saved to: {results_csv_path}")

    return c_indices, avg_cindex


def inference_ensemble(df, feature_cols, fold_count=NUM_FOLDS):
    """
    Perform inference on entire df using ensemble of saved models.
    Average the partial hazards predicted by each fold model.
    Returns ensemble c-index.
    """
    # Prepare input features
    X = df[feature_cols].copy()

    partial_hazards_list = []

    for fold in range(fold_count):
        model_path = os.path.join(COX_BASELINE_DIR, f"cox_model_fold{fold+1}.pkl")
        scaler_path = os.path.join(COX_BASELINE_DIR, f"scaler_fold{fold+1}.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model for fold {fold+1} not found. Train models first!")

        with open(model_path, 'rb') as f:
            cph = pickle.load(f)

        if SCALE_DATA and os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols, index=df.index)
        else:
            X_scaled = X

        pred_partial_hazard = cph.predict_partial_hazard(X_scaled)
        partial_hazards_list.append(pred_partial_hazard)

    # Ensemble by averaging partial hazards (linear scale)
    ensemble_partial_hazard = pd.concat(partial_hazards_list, axis=1).mean(axis=1)

    c_index = concordance_index(df['duration'], -ensemble_partial_hazard, df['event'])
    print(f"\nEnsemble Inference C-index on entire dataset: {c_index:.4f}")

    return c_index


def main(mode="train"):
    """
    Main entry point to either train or infer.
    mode: 'train' or 'inference'
    """
    if mode == "train":
        print("Starting 5-fold cross-validation training...")
        train_cv_cox_model(clinical_df, CLINICAL_FEATURES, fold_indices)

    elif mode == "inference":
        print("Starting inference using ensemble of 5 folds on entire dataset...")
        inference_ensemble(clinical_df, CLINICAL_FEATURES, fold_count=len(fold_indices))

    else:
        raise ValueError("Invalid mode! Use 'train' or 'inference'.")

if __name__ == "__main__":
    mode = 'train' # 'train' | 'inference'
    
    main(mode)