import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from config import *
from utils.data_utils import load_clinical, load_mri_features, load_radiomic_features, load_wsi_features, load_wsi_features_mult, create_folds, get_feature_dimensionalities, load_folds
from train_eval import SurvivalDataset, train_one_epoch, evaluate, save_model
from models.model import MultimodalSurvivalModel
import torch.optim as optim
from sampler import EventBalancedBatchSampler  # Import custom sampler
import matplotlib.pyplot as plt
from mri_feature_extraction import extract_MRI_features
from radiomic_feature_extraction import extract_radiomic_features
from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import StandardScaler
import joblib
import json

def plot_learning_curve(train_losses, val_cindices, fold_idx):
    plt.figure(figsize=(10, 5))
    epochs = np.arange(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_cindices, label='Validation C-Index')

    plt.xlabel('Epoch')
    plt.title(f'Fold {fold_idx} Training Curve')
    plt.legend()
    plt.grid(True)

    os.makedirs(GLOBAL_DIR, exist_ok=True)
    plt_path = os.path.join(GLOBAL_DIR, f"fold_{fold_idx}_learning_curve.png")
    plt.savefig(plt_path)
    plt.close()

def get_or_fit_scaler(name, train_array, fit=True, run=0, fold=0):
    os.makedirs(GLOBAL_DIR, exist_ok=True)
    scaler_path = os.path.join(GLOBAL_DIR, f"{name}_scaler_run_{run}_{fold}.pkl")
    
    if fit:
        scaler = StandardScaler()
        scaler.fit(train_array)
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)

    return scaler

def maybe_scale(name, train_array, val_array, fit=True, run=0, fold=0):
    scaler = get_or_fit_scaler(name, train_array, fit=fit, run=run, fold=fold)
    train_scaled = scaler.transform(train_array) if train_array is not None else None
    val_scaled = scaler.transform(val_array) if val_array is not None else None
    return train_scaled, val_scaled

def bin_durations(durations, bin_edges):
    durations_tensor = torch.tensor(durations, dtype=torch.float32)
    durations_bin = torch.bucketize(durations_tensor, bin_edges) - 1
    durations_bin = torch.clamp(durations_bin, 0, len(bin_edges) - 2)
    return durations_bin.numpy()  # convert back to numpy if needed

def main():
    clinical_df = load_clinical()
    clinical_dim = get_feature_dimensionalities(clinical_df)

    y_event = clinical_df['event'].values
    case_ids = clinical_df['Case_ID'].astype(str).tolist()

    if os.path.exists(FOLDS_CSV):
        folds = load_folds()
    else:
        create_folds(y_event, case_ids)

    run_cindices = []
    run_stds = []

    min_time = 5
    max_time = 250
    
    bin_edges = torch.linspace(min_time, max_time, steps=TIME_BINS+1)  # 31 edges for 30 bins

    for run_num in range(RUNS):
        val_cindices = []
        for fold_idx in range(NUM_FOLDS):
            print(f"\n=== Fold {fold_idx + 1}/{NUM_FOLDS} ===")

            val_ids = folds[fold_idx]
            train_ids = [cid for cid in case_ids if cid not in val_ids]

            train_clinical = clinical_df[clinical_df['Case_ID'].isin(train_ids)].sort_values('Case_ID')
            val_clinical = clinical_df[clinical_df['Case_ID'].isin(val_ids)].sort_values('Case_ID')

            # Clinical features
            train_clin_array = train_clinical.drop(columns=['Case_ID', 'duration', 'event']).values if USE_CLINICAL_FEATURES else None
            val_clin_array = val_clinical.drop(columns=['Case_ID', 'duration', 'event']).values if USE_CLINICAL_FEATURES else None

            # MRI features
            train_mri_array = load_mri_features(train_clinical['Case_ID'].tolist()) if USE_MRI_FEATURES else None
            val_mri_array = load_mri_features(val_clinical['Case_ID'].tolist()) if USE_MRI_FEATURES else None

            # WSI features
            if AGGREG_CASE_WSI:
                train_wsi_array = load_wsi_features_mult(train_clinical['Case_ID'].tolist(), csv_path=WSI_FEATURES_CSV) if USE_WSI_FEATURES else None
                val_wsi_array   = load_wsi_features_mult(val_clinical['Case_ID'].tolist(), csv_path=WSI_FEATURES_CSV) if USE_WSI_FEATURES else None
            else:
                train_wsi_array = load_wsi_features(train_clinical['Case_ID'].tolist(), csv_path=WSI_FEATURES_CSV) if USE_WSI_FEATURES else None
                val_wsi_array   = load_wsi_features(val_clinical['Case_ID'].tolist(), csv_path=WSI_FEATURES_CSV) if USE_WSI_FEATURES else None

            # Radiomic features (handled separately)
            train_radiomic_array, val_radiomic_array = None, None
            if USE_RADIOMIC_FEATURES:
                train_radiomic_array = load_radiomic_features(train_clinical['Case_ID'].tolist(), csv_path=RADIOMIC_CSV)
                val_radiomic_array = load_radiomic_features(val_clinical['Case_ID'].tolist(), csv_path=RADIOMIC_CSV)

                if SCALE_DATA:
                    train_radiomic_array, val_radiomic_array = maybe_scale("radiomic", train_radiomic_array, val_radiomic_array, fit=True, run=run_num, fold=fold_idx)

            # Optional scaling (only for clinical, MRI, WSI)
            if SCALE_DATA:
                if USE_CLINICAL_FEATURES:
                    train_clin_array, val_clin_array = maybe_scale("clinical", train_clin_array, val_clin_array, fit=True, run=run_num, fold=fold_idx)
                if USE_MRI_FEATURES:
                    train_mri_array, val_mri_array = maybe_scale("mri", train_mri_array, val_mri_array, fit=True, run=run_num, fold=fold_idx)
                if USE_WSI_FEATURES:
                    train_wsi_array, val_wsi_array = maybe_scale("wsi", train_wsi_array, val_wsi_array, fit=True, run=run_num, fold=fold_idx)

            # Concatenate radiomic features to clinical AFTER scaling
            if USE_RADIOMIC_FEATURES:
                if train_clin_array is not None:
                    train_clin_array = np.concatenate([train_clin_array, train_radiomic_array], axis=1)
                    val_clin_array = np.concatenate([val_clin_array, val_radiomic_array], axis=1)
                else:
                    train_clin_array = train_radiomic_array
                    val_clin_array = val_radiomic_array

            train_durations_raw = train_clinical['duration'].values
            val_durations_raw = val_clinical['duration'].values

            train_durations = bin_durations(train_durations_raw, bin_edges)
            val_durations = bin_durations(val_durations_raw, bin_edges)

            train_events = train_clinical['event'].values
            val_events = val_clinical['event'].values
            # Print counts for training events
            print(f"Train - positives (1): {np.sum(train_events == 1)}, negatives (0): {np.sum(train_events == 0)}")

            # Print counts for validation events
            print(f"Val   - positives (1): {np.sum(val_events == 1)}, negatives (0): {np.sum(val_events == 0)}")

            train_dataset = SurvivalDataset(train_clin_array, train_mri_array, train_wsi_array, train_durations, train_events)
            val_dataset = SurvivalDataset(val_clin_array, val_mri_array, val_wsi_array, val_durations, val_events)

            train_sampler = EventBalancedBatchSampler(events=train_events, batch_size=BATCH_SIZE, drop_last=False)
            train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

            c_dim = clinical_dim if USE_CLINICAL_FEATURES else 0
            m_dim = M_FEATURE_DIM if USE_MRI_FEATURES else 0
            w_dim = W_FEATURE_DIM if USE_WSI_FEATURES else 0

            r_dim = R_FEATURE_DIM if USE_RADIOMIC_FEATURES else 0
            
            if r_dim > 0:
                c_dim += r_dim

            model = MultimodalSurvivalModel(c_dim, m_dim, w_dim,
                                            fusion_type=FUSION_TYPE,
                                            survival_model=SURVIVAL_MODEL,
                                            time_bins=TIME_BINS)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD) 

            best_cindex = -np.inf
            best_epoch = 0
            wait = 0

            # Lists to store loss and C-index for plotting
            train_losses = []
            val_cindices_fold = []

            for epoch in range(1, EPOCHS + 1):
                train_loss = train_one_epoch(model, train_loader, optimizer, SURVIVAL_MODEL, device)
                val_cindex = evaluate(model, val_loader, SURVIVAL_MODEL, device)[0]

                train_losses.append(train_loss)
                val_cindices_fold.append(val_cindex)

                print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val C-index: {val_cindex:.4f}")

                # Early stopping
                if val_cindex > best_cindex:
                    best_cindex = val_cindex
                    best_epoch = epoch
                    save_model(model, fold_idx, run_num)
                    wait = 0
                else:
                    wait += 1

                if wait >= PATIENCE:
                    print(f"Early stopping at epoch {epoch}. Best C-index: {best_cindex:.4f} at epoch {best_epoch}")
                    break

            val_cindices.append(best_cindex)

            # Plot training curve
            #plot_learning_curve(train_losses, val_cindices_fold, fold_idx)

        avg_cindex = round(np.mean(val_cindices), 3)
        std_cindex = round(np.std(val_cindices), 2)
        
        print(f"\nAverage C-index across all folds: {avg_cindex} ± {std_cindex}")

        # Save fold-wise and summary C-index results to CSV
        results_path = os.path.join(GLOBAL_DIR, f"fold_cindices_run{run_num}.csv")

        # Build dataframe
        fold_rows = [{"Fold": f"Fold {i+1}", "C-Index": c} for i, c in enumerate(val_cindices)]
        avg_row = {"Fold": "Average", "C-Index": f"{avg_cindex} ± {std_cindex}"}
        fold_rows.append(avg_row)

        results_df = pd.DataFrame(fold_rows)
        results_df.to_csv(results_path, index=False)

        if VERBOSE:
            print(f"\nSaved fold-wise C-indices to {results_path}")
        
        run_cindices.append((run_num, avg_cindex, val_cindices))
        run_stds.append(std_cindex)

    print(f"\n Runs: {RUNS}=======Modalities===========")
    print(f"Clinical: {USE_CLINICAL_FEATURES}, Radiomic: {USE_RADIOMIC_FEATURES}, MRI: {USE_MRI_FEATURES}, WSI: {USE_WSI_FEATURES}")
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

    # === Overall standard deviation across all folds in all runs ===
    print(f"\nRuns Average C-index: {round(np.mean(avg_cindices_only), 3)} ± {round(np.std(avg_cindices_only), 3)}")


if __name__ == "__main__":
    os.makedirs(GLOBAL_DIR, exist_ok=True)
    ## Set up the paths in config.py
    ## Extract embeddings using a pretraind model
    #print("\n=== MRI deep features extraction ===")
    #extract_MRI_features()
    #print("\n=== MRI handcrafted features extraction ===")
    #extract_radiomic_features()
    
    ## Step 2: Do 5-folds cross-valiation using existing stratified random fold
    #print(f"\n=== Step 2: {NUM_FOLDS}-folds cross ===")
    main()