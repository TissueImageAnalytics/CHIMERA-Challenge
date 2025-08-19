import os
import numpy as np
import pandas as pd
import torch
from config import *
from utils.data_utils import (
    load_clinical, load_mri_features, load_wsi_features, load_wsi_features_mult,
    create_folds, get_feature_dimensionalities, load_folds, maybe_scale, safe_to_array
)

import torch.optim as optim
#from sampler import EventBalancedBatchSampler
import matplotlib.pyplot as plt
from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import StandardScaler
import joblib
import json
from run_optuna import optuna_optim
from models.model import TransductiveSR as TSRR
from lifelines.statistics import logrank_test
from scipy.stats import combine_pvalues

def main():
    params_path = os.path.join(GLOBAL_DIR, 'params.json')

    # Load JSON back into best_params
    with open(params_path, 'r') as f:
        best_params = json.load(f)

    clinical_df = load_clinical()
    clinical_dim = get_feature_dimensionalities(clinical_df)

    y_event = clinical_df['event'].values
    case_ids = clinical_df['Case_ID'].astype(str).tolist()

    folds = load_folds() if os.path.exists(FOLDS_CSV) else create_folds(y_event, case_ids)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_cindices = []
    run_stds = []
    
    for run_num in range(RUNS):
        val_cindices = []
        val_pvalues = []
        threshold = 0
        p = 2
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
                    train_clin_array, val_clin_array, clin_scaler = maybe_scale("clinical", train_clin_array, val_clin_array, fit=True, fold=fold_idx, return_scaler=True)
                    scaler_path = os.path.join(GLOBAL_DIR, f"clinical_scaler_RUN{run_num}_FOLD{fold_idx}.pkl")
                joblib.dump(clin_scaler, scaler_path)
                if USE_MRI_FEATURES:
                    train_mri_array, val_mri_array, mri_scaler = maybe_scale("mri", train_mri_array, val_mri_array, fit=True, fold=fold_idx, return_scaler=True)
                    scaler_path = os.path.join(GLOBAL_DIR, f"mri_scaler_RUN{run_num}_FOLD{fold_idx}.pkl")
                    joblib.dump(mri_scaler, scaler_path)
                if USE_WSI_FEATURES:
                    train_wsi_array, val_wsi_array, wsi_scaler = maybe_scale("wsi", train_wsi_array, val_wsi_array, fit=True, fold=fold_idx, return_scaler=True)             
                    scaler_path = os.path.join(GLOBAL_DIR, f"wsi_scaler_RUN{run_num}_FOLD{fold_idx}.pkl")
                    joblib.dump(wsi_scaler, scaler_path)

            T_train = train_clinical['duration'].values
            E_train = train_clinical['event'].values
            T_val   = val_clinical['duration'].values
            E_val   = val_clinical['event'].values

            n_train = train_clinical.shape[0]
            n_val = val_clinical.shape[0]

            print(f"Train - positives (1): {np.sum(E_train == 1)}, negatives (0): {np.sum(E_train == 0)}")
            print(f"Val   - positives (1): {np.sum(E_val == 1)}, negatives (0): {np.sum(E_val == 0)}")

            # safe conversion to arrays
            train_clin_array = safe_to_array(train_clin_array, n_train)
            val_clin_array   = safe_to_array(val_clin_array, n_val)

            train_mri_array  = safe_to_array(train_mri_array, n_train)
            val_mri_array    = safe_to_array(val_mri_array, n_val)

            train_wsi_array  = safe_to_array(train_wsi_array, n_train)
            val_wsi_array    = safe_to_array(val_wsi_array, n_val)

            # collect only active modalities
            modalities_train, modalities_val = [], []

            if USE_CLINICAL_FEATURES:
                modalities_train.append(train_clin_array)
                modalities_val.append(val_clin_array)

            if USE_MRI_FEATURES:
                modalities_train.append(train_mri_array)
                modalities_val.append(val_mri_array)

            if USE_WSI_FEATURES:
                modalities_train.append(train_wsi_array)
                modalities_val.append(val_wsi_array)

            # final design matrices
            X_train = np.concatenate(modalities_train, axis=1)
            X_val   = np.concatenate(modalities_val, axis=1)

            c_dim = clinical_dim if USE_CLINICAL_FEATURES else 0
            m_dim = MRI_FEATURE_DIM if USE_MRI_FEATURES else 0
            w_dim = WSI_FEATURE_DIM if USE_WSI_FEATURES else 0

            tsr_model = TSRR(
                lambda_w=best_params['lambda_w'],
                lambda_u=best_params['lambda_u'],
                p=p,
                Tmax=2000,
                lr=best_params['lr'],
                dropout=best_params['dropout'],
                latent_dim=best_params['latent_dim'],
                structure=TSR_STRUCTURE,
                clin_dim=c_dim, mri_dim=m_dim, wsi_dim=w_dim,
            )

            #tsr_model.to(device)

            tsr_model.fit(X_train, T_train, E_train, X_val, plot_loss=False)

            # === Save model per fold & run ===
            model_path = os.path.join(
                GLOBAL_DIR, f"model_RUN{run_num}_FOLD{fold_idx}.pkl"
            )
            joblib.dump(tsr_model, model_path)

            # === Evaluation ===
            Z_test = tsr_model.decision_function(X_val)
            cindex_val, _, _, _, _ = concordance_index_censored(E_val.astype(bool), T_val, -Z_test)
            val_cindices.append(cindex_val)

            results_df = pd.DataFrame({'Prediction': Z_test, 'Time': T_val, 'Event': E_val})
            low_group = results_df[results_df['Prediction'] <= threshold]
            high_group = results_df[results_df['Prediction'] > threshold]

            result = logrank_test(
                low_group['Time'], high_group['Time'],
                event_observed_A=low_group['Event'],
                event_observed_B=high_group['Event']
            )
            val_pvalues.append(result.p_value)

        # === Log results for this run ===
        mean_score = round(np.mean(val_cindices), 3)
        std_score = round(np.std(val_cindices), 2)
        _, combined_p = combine_pvalues(val_pvalues, method='fisher')

        run_cindices.append((run_num, mean_score, val_cindices))
        run_stds.append(std_score)

        # Save all run metrics and params
        results_path = os.path.join(
            GLOBAL_DIR, f"PARMS_RESULTS_{TUNING_TRIALS}_RUN_{run_num}.csv"
        )

        results_df = {
            "lambda_w": best_params["lambda_w"],
            "lambda_u": best_params["lambda_u"],
            "lr": best_params["lr"],
            "latent_dim": best_params["latent_dim"],
            "dropout": best_params["dropout"],
            "mean_cindex": mean_score,
            "std_cindex": std_score,
            "combined_p": combined_p,
            "cindices": str(val_cindices),
            "p_values": str(val_pvalues)
        }
        pd.DataFrame([results_df]).to_csv(results_path, index=False)
        print(f"Saved results to {results_path}")

    print(f"\n Runs: {RUNS}=======Modalities===========")
    print(f"\nRuns Modalities: Clinical({USE_CLINICAL_FEATURES}), MRI({USE_MRI_FEATURES}), WSI({USE_WSI_FEATURES})")
    avg_cindices_only = [c[1] for c in run_cindices]  # extract just the average C-index per run
    
    # Identify best run based on average C-index
    best_run = max(run_cindices, key=lambda x: x[1])  # (run_num, avg_cindex, [fold_cindices])
    best_run_num = best_run[0]
    best_run_cindices = best_run[2]

    # Save best run info
    best_run_info = {
        "best_run": best_run_num,
        "best_run_avg_cindex": best_run[1],
        "fold_cindices": best_run_cindices
    }
    with open(os.path.join(GLOBAL_DIR, "best_run.json"), "w") as f:
        json.dump(best_run_info, f, indent=2)
    
    print("Folds C-index ± std")
    formatted = [f"{c:.3f}±{s:.2f}" for c, s in zip(avg_cindices_only, run_stds)]
    print(", ".join(formatted))
    print(f"\nRuns Average C-index: {round(np.mean(avg_cindices_only), 3)} ± {round(np.std(avg_cindices_only), 3)}")

    
if __name__ == "__main__":
    os.makedirs(GLOBAL_DIR, exist_ok=True)
    print(f"\n=== Step 1: Hyperparam optimisation using Optuna ===")
    optuna_optim()
    print(f"\n=== Step 2: {NUM_FOLDS}-folds cross ===")
    main()
