import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from config import *
from utils.data_utils import (
    load_clinical, load_rna_features, load_wsi_features, load_wsi_features_mult,
    create_folds, get_feature_dimensionalities, load_folds
)
from train_eval import SurvivalDataset, train_one_epoch, evaluate, save_model
from models.model import MultimodalSurvivalModel
import torch.optim as optim
from sampler import EventBalancedBatchSampler
import matplotlib.pyplot as plt
from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

def maybe_reduce_rna(train_rna, val_rna, train_ids, val_ids, fit=True, fold=0):
    os.makedirs(GLOBAL_DIR, exist_ok=True)
    
    pca_path = os.path.join(GLOBAL_DIR, f"rna_pca_{fold}.pkl")
    train_path = os.path.join(GLOBAL_DIR, f"reduced_rna_fold{fold}.npy")
    val_path = os.path.join(GLOBAL_DIR, f"reduced_rna_val_fold{fold}.npy")
    train_ids_path = os.path.join(GLOBAL_DIR, f"reduced_rna_ids_fold{fold}.csv")

    if fit:
        print(f" Fitting PCA on RNA train set (fold {fold})...")
        pca = PCA(n_components=RNA_DIM_REDUCE_TO)
        pca.fit(train_rna)
        joblib.dump(pca, pca_path)

        train_rna = pca.transform(train_rna)
        val_rna = pca.transform(val_rna)

        # Save transformed train/val RNA for potential reuse or inspection
        np.save(train_path, train_rna)
        np.save(val_path, val_rna)
        pd.DataFrame({'Case_ID': train_ids}).to_csv(train_ids_path, index=False)
    else:
        print(f" Loading saved reduced RNA features and PCA model for fold {fold}")
        pca = joblib.load(pca_path)
        train_rna = np.load(train_path)
        val_rna = pca.transform(val_rna)

    return train_rna, val_rna

def bin_durations(durations, bin_edges):
    durations_tensor = torch.tensor(durations, dtype=torch.float32)
    durations_bin = torch.bucketize(durations_tensor, bin_edges) - 1
    durations_bin = torch.clamp(durations_bin, 0, len(bin_edges) - 2)
    return durations_bin.numpy()  # convert back to numpy if needed

def main():
    clinical_df = load_clinical()
    #clinical_dim = get_feature_dimensionalities(clinical_df)

    y_event = clinical_df['event'].values
    case_ids = clinical_df['Case_ID'].astype(str).tolist()

    folds = load_folds() if os.path.exists(FOLDS_CSV) else create_folds(y_event, case_ids)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

            train_clin_array = train_clinical.drop(columns=['Case_ID', 'duration', 'event']).values if USE_CLINICAL_FEATURES else None
            val_clin_array = val_clinical.drop(columns=['Case_ID', 'duration', 'event']).values if USE_CLINICAL_FEATURES else None

            train_rna_array = load_rna_features(train_clinical['Case_ID'].tolist()) if USE_RNA_FEATURES else None
            val_rna_array = load_rna_features(val_clinical['Case_ID'].tolist()) if USE_RNA_FEATURES else None

            if USE_RNA_FEATURES:
                # Step 1: Scale RNA features (save/load scaler per fold)
                train_rna_array, val_rna_array = maybe_scale(
                    "rna", train_rna_array, val_rna_array, fit=True, fold=fold_idx
                )

                # Step 2: Apply PCA on scaled RNA (save/load PCA model per fold)
                if RNA_DIM_REDUCE_TO < RNA_FEATURE_DIM:
                    train_rna_array, val_rna_array = maybe_reduce_rna(
                        train_rna_array, val_rna_array,
                        train_ids=train_clinical['Case_ID'].tolist(),
                        val_ids=val_clinical['Case_ID'].tolist(),
                        fit=True, fold=fold_idx
                    )
                # train_rna_array, val_rna_array = maybe_reduce_rna(
                #     train_rna_array, val_rna_array,
                #     train_ids=train_clinical['Case_ID'].tolist(),
                #     val_ids=val_clinical['Case_ID'].tolist(),
                #     fit=True, fold=fold_idx,
                #     supervised=True,
                #     top_k=500
                # )

            # WSI features
            if AGGREG_CASE_WSI:
                train_wsi_array = load_wsi_features_mult(train_clinical['Case_ID'].tolist(), csv_path=WSI_FEATURES_CSV) if USE_WSI_FEATURES else None
                val_wsi_array   = load_wsi_features_mult(val_clinical['Case_ID'].tolist(), csv_path=WSI_FEATURES_CSV) if USE_WSI_FEATURES else None
            else:
                train_wsi_array = load_wsi_features(train_clinical['Case_ID'].tolist(), csv_path=WSI_FEATURES_CSV) if USE_WSI_FEATURES else None
                val_wsi_array   = load_wsi_features(val_clinical['Case_ID'].tolist(), csv_path=WSI_FEATURES_CSV) if USE_WSI_FEATURES else None

            if SCALE_DATA:
                if USE_CLINICAL_FEATURES:
                    train_clin_array, val_clin_array = maybe_scale("clinical", train_clin_array, val_clin_array, fit=True, run=run_num, fold=fold_idx)
                if USE_WSI_FEATURES:
                    train_wsi_array, val_wsi_array = maybe_scale("wsi", train_wsi_array, val_wsi_array, fit=True, run=run_num, fold=fold_idx)

            if BIN_EDGES:
                train_durations_raw = train_clinical['duration'].values
                val_durations_raw = val_clinical['duration'].values
                train_durations = bin_durations(train_durations_raw, bin_edges)
                val_durations = bin_durations(val_durations_raw, bin_edges)
            else:
                train_durations = train_clinical['duration'].values
                val_durations = val_clinical['duration'].values

            train_events = train_clinical['event'].values
            val_events = val_clinical['event'].values

            # Print counts for training events
            print(f"Train - positives (1): {np.sum(train_events == 1)}, negatives (0): {np.sum(train_events == 0)}")

            # Print counts for validation events
            print(f"Val   - positives (1): {np.sum(val_events == 1)}, negatives (0): {np.sum(val_events == 0)}")

            train_dataset = SurvivalDataset(train_clin_array, train_rna_array, train_wsi_array, train_durations, train_events)
            val_dataset = SurvivalDataset(val_clin_array, val_rna_array, val_wsi_array, val_durations, val_events)

            if FULL_BATCH:
                train_batch_size = len(train_dataset)
            else:
                train_batch_size = BATCH_SIZE

            #train_sampler = EventBalancedBatchSampler(events=train_events, batch_size=train_batch_size, drop_last=False)
            #train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
            val_loader = DataLoader(val_dataset, batch_size=train_batch_size, shuffle=False, drop_last=False)

            if SURVIVAL_MODEL in ['cox', 'deepsurv']:
                train_loader = DataLoader(train_dataset,
                                        batch_size=len(train_dataset),
                                        shuffle=False, drop_last=False)
            else:
                train_sampler = EventBalancedBatchSampler(events=train_events,
                                                        batch_size=BATCH_SIZE,
                                                        drop_last=False)
                train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)


            # c_dim = clinical_dim if USE_CLINICAL_FEATURES else 0
            # r_dim = RNA_DIM_REDUCE_TO if USE_RNA_FEATURES else 0
            # w_dim = WSI_FEATURE_DIM if USE_WSI_FEATURES else 0
            c_dim = train_clin_array.shape[1] if (USE_CLINICAL_FEATURES and train_clin_array is not None) else 0
            r_dim = train_rna_array.shape[1] if (USE_RNA_FEATURES and train_rna_array is not None) else 0
            w_dim = train_wsi_array.shape[1] if (USE_WSI_FEATURES and train_wsi_array is not None) else 0

            model = MultimodalSurvivalModel(c_dim, r_dim, w_dim)

            model.to(device)

            optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

            best_cindex = -np.inf
            best_epoch = 0
            wait = 0

            train_losses = []
            val_cindices_fold = []

            for epoch in range(1, EPOCHS + 1):
                train_loss = train_one_epoch(model, train_loader, optimizer, SURVIVAL_MODEL, device)
                val_cindex = evaluate(model, val_loader, SURVIVAL_MODEL, device)[0]

                train_losses.append(train_loss)
                val_cindices_fold.append(val_cindex)

                print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val C-index: {val_cindex:.4f}")

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
            #plot_learning_curve(train_losses, val_cindices_fold, fold_idx)

        avg_cindex = round(np.mean(val_cindices), 3)
        std_cindex = round(np.std(val_cindices), 2)
        
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
    print(f"Clinical: {USE_CLINICAL_FEATURES}, RNA: {USE_RNA_FEATURES}, WSI: {USE_WSI_FEATURES}")
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
    print(f"\n=== Step 2: {NUM_FOLDS}-folds cross validation===")
    main()
