import numpy as np
import os
import optuna
from lifelines.utils import concordance_index as cindex
from sksurv.metrics import concordance_index_censored
from models.model import TransductiveSR as TSRR
from config import *
import torch
import json
from utils.data_utils import (
    load_clinical, load_mri_features, load_wsi_features, load_wsi_features_mult,
    create_folds, get_feature_dimensionalities, load_folds, maybe_scale
)

def safe_to_array(arr, n_samples):
    # if arr is None, return empty array with shape (n_samples, 0)
    if arr is None:
        return np.empty((n_samples, 0))
    return arr

# === Optuna Objective ===
def objective(trial):
    lambda_w = trial.suggest_loguniform('lambda_w', 1e-2, 1.0)
    lambda_u = trial.suggest_loguniform('lambda_u', 1e-2, 1.0)
    LR = trial.suggest_loguniform('lr', 1e-4, 1e-1)
    latent_dim = trial.suggest_int('latent_dim', 16, 128)
    dropout = trial.suggest_uniform('dropout', 0.0, 0.5)

    clinical_df = load_clinical()
    clinical_dim = get_feature_dimensionalities(clinical_df)

    y_event = clinical_df['event'].values
    case_ids = clinical_df['Case_ID'].astype(str).tolist()

    folds = load_folds() if os.path.exists(FOLDS_CSV) else create_folds(y_event, case_ids)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p = 2
    cv_scores = []

    for fold_idx in range(NUM_FOLDS):
        print(f"\n=== Fold {fold_idx + 1}/{NUM_FOLDS} ===")

        val_ids = folds[fold_idx]
        train_ids = [cid for cid in case_ids if cid not in val_ids]

        train_clinical = clinical_df[clinical_df['Case_ID'].isin(train_ids)].sort_values('Case_ID')
        val_clinical = clinical_df[clinical_df['Case_ID'].isin(val_ids)].sort_values('Case_ID')

        train_clin_array = train_clinical.drop(columns=['Case_ID', 'duration', 'event']).values if USE_CLINICAL_FEATURES else None
        val_clin_array = val_clinical.drop(columns=['Case_ID', 'duration', 'event']).values if USE_CLINICAL_FEATURES else None

        train_mri_array = load_mri_features(train_clinical['Case_ID'].tolist()) if USE_MRI_FEATURES else None
        val_mri_array = load_mri_features(val_clinical['Case_ID'].tolist()) if USE_MRI_FEATURES else None

                
        # WSI features
        if AGGREG_CASE_WSI:
            train_wsi_array = load_wsi_features_mult(train_clinical['Case_ID'].tolist(), csv_path=WSI_FEATURES_CSV) if USE_WSI_FEATURES else None
            val_wsi_array   = load_wsi_features_mult(val_clinical['Case_ID'].tolist(), csv_path=WSI_FEATURES_CSV) if USE_WSI_FEATURES else None
        else:
            train_wsi_array = load_wsi_features(train_clinical['Case_ID'].tolist(), csv_path=WSI_FEATURES_CSV) if USE_WSI_FEATURES else None
            val_wsi_array   = load_wsi_features(val_clinical['Case_ID'].tolist(), csv_path=WSI_FEATURES_CSV) if USE_WSI_FEATURES else None

        if SCALE_DATA:
            if USE_CLINICAL_FEATURES:
                train_clin_array, val_clin_array = maybe_scale("clinical", train_clin_array, val_clin_array, fit=True, fold=fold_idx)
            if USE_MRI_FEATURES:
                train_mri_array, val_mri_array = maybe_scale("mri", train_mri_array, val_mri_array, fit=True, fold=fold_idx)
            if USE_WSI_FEATURES:
                train_wsi_array, val_wsi_array = maybe_scale("wsi", train_wsi_array, val_wsi_array, fit=True, fold=fold_idx)

        T_train = train_clinical['duration'].values
        E_train = train_clinical['event'].values
        T_val   = val_clinical['duration'].values
        E_val   = val_clinical['event'].values

        n_train = train_clinical.shape[0]
        n_val = val_clinical.shape[0]

        print(f"Train - positives (1): {np.sum(E_train == 1)}, negatives (0): {np.sum(E_train == 0)}")
        print(f"Val   - positives (1): {np.sum(E_val == 1)}, negatives (0): {np.sum(E_val == 0)}")

        # safe conversion to arrays that can concatenate
        train_clin_array = safe_to_array(train_clin_array, n_train)
        val_clin_array = safe_to_array(val_clin_array, n_val)

        train_mri_array = safe_to_array(train_mri_array, n_train)
        val_mri_array = safe_to_array(val_mri_array, n_val)

        train_wsi_array = safe_to_array(train_wsi_array, n_train)
        val_wsi_array = safe_to_array(val_wsi_array, n_val)

        # concatenate in fixed order: clinical, mri, wsi 
        X_train = np.concatenate([train_clin_array, train_mri_array, train_wsi_array], axis=1)
        X_val = np.concatenate([val_clin_array, val_mri_array, val_wsi_array], axis=1)

        c_dim = clinical_dim if USE_CLINICAL_FEATURES else 0
        m_dim = MRI_FEATURE_DIM if USE_MRI_FEATURES else 0
        w_dim = WSI_FEATURE_DIM if USE_WSI_FEATURES else 0

        tsr_model = TSRR(
            lambda_w=lambda_w,
            lambda_u=lambda_u,
            p=p,
            Tmax=2000,
            lr=LR,
            dropout=dropout,
            latent_dim=latent_dim,
            structure=TSR_STRUCTURE,
            clin_dim=c_dim, mri_dim=m_dim, wsi_dim=w_dim,
            device=device
        )

        tsr_model.fit(X_train, T_train, E_train, X_val, plot_loss=False)
        Z_val = tsr_model.decision_function(X_val)

        cindex_val, _, _, _, _ = concordance_index_censored(
            E_val.astype(bool), T_val, -Z_val
        )
        cv_scores.append(cindex_val)

    return np.mean(cv_scores)

def optuna_optim():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=TUNING_TRIALS)
    best_params = study.best_params
    print("\nBest hyperparameters:", best_params)
    params_path = os.path.join(GLOBAL_DIR, 'params.json')

    # Save to JSON
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=4)

    print(f"Hyperparameters saved to {params_path}")