import os
import numpy as np
import pandas as pd
import torch
from config import *
from utils.data_utils import load_clinical, load_mri_features, load_wsi_features, get_feature_dimensionalities
from models.model import MultimodalSurvivalModel
from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import StandardScaler
import joblib

def get_or_fit_scaler(name, train_array, fit=True, fold=0):
    os.makedirs(GLOBAL_DIR, exist_ok=True)
    scaler_path = os.path.join(GLOBAL_DIR, f"{name}_scaler_{fold}.pkl")
    
    if fit:
        scaler = StandardScaler()
        scaler.fit(train_array)
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)

    return scaler

def maybe_scale(name, train_array, val_array, fit=True, fold=0):
    scaler = get_or_fit_scaler(name, train_array, fit=fit, fold=fold)
    train_scaled = scaler.transform(train_array) if train_array is not None else None
    val_scaled = scaler.transform(val_array) if val_array is not None else None
    return train_scaled, val_scaled

def inference(test_case_ids):
   
    # Load clinical data and get feature dims
    clinical_df = load_clinical()
    clinical_dim = get_feature_dimensionalities(clinical_df)

    # Sort test case IDs to preserve consistent output order
    test_case_ids = sorted(test_case_ids)

    # Get input arrays
    test_clinical = clinical_df[clinical_df['Case_ID'].isin(test_case_ids)].sort_values('Case_ID')
    test_clin_array = test_clinical.drop(columns=['Case_ID', 'duration', 'event']).values if USE_CLINICAL_FEATURES else None
    test_mri_array = load_mri_features(test_case_ids) if USE_MRI_FEATURES else None
    test_wsi_array = load_wsi_features(test_case_ids, csv_path=WSI_FEATURES_CSV) if USE_WSI_FEATURES else None

    # === Apply same scalers ===
    if SCALE_DATA:
        if USE_CLINICAL_FEATURES:
            test_clin_array, _ = maybe_scale("clinical", test_clin_array, test_clin_array, fit=False)
        if USE_MRI_FEATURES:
            test_mri_array, _ = maybe_scale("mri", test_mri_array, test_mri_array, fit=False)
        if USE_WSI_FEATURES:
            test_wsi_array, _ = maybe_scale("wsi", test_wsi_array, test_wsi_array, fit=False)

    # Feature dimensions
    c_dim = clinical_dim if USE_CLINICAL_FEATURES else 0
    m_dim = M_FEATURE_DIM if USE_MRI_FEATURES else 0
    w_dim = W_FEATURE_DIM if USE_WSI_FEATURES else 0

    # Prepare model template
    model = MultimodalSurvivalModel(
        clin_dim=c_dim,
        mri_dim=m_dim,
        wsi_dim=w_dim,
        fusion_type=FUSION_TYPE,
        survival_model=SURVIVAL_MODEL,
        time_bins=TIME_BINS
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Make sure arrays are 2D: (batch_size, feature_dim) so can process a single case as well
    if test_clin_array is not None and test_clin_array.ndim == 1:
        test_clin_array = test_clin_array.reshape(1, -1)

    if test_mri_array is not None and test_mri_array.ndim == 1:
        test_mri_array = test_mri_array.reshape(1, -1)

    if test_wsi_array is not None and test_wsi_array.ndim == 1:
        test_wsi_array = test_wsi_array.reshape(1, -1)

    # Prepare input tensors
    N = len(test_case_ids)
    clin_tensor = torch.tensor(test_clin_array, dtype=torch.float32).to(device) if test_clin_array is not None else torch.zeros((N, c_dim), device=device)
    mri_tensor = torch.tensor(test_mri_array, dtype=torch.float32).to(device) if test_mri_array is not None else torch.zeros((N, m_dim), device=device)
    wsi_tensor = torch.tensor(test_wsi_array, dtype=torch.float32).to(device) if test_wsi_array is not None else torch.zeros((N, m_dim), device=device)

    # Define save dir
    
    if USE_ENSEMBLE:
        # === ENSEMBLE: Load all folds ===
        pmf_all_folds = []

        for fold_idx in range(NUM_FOLDS):
            model_path = os.path.join(GLOBAL_DIR, f"best_model_fold{fold_idx}.pt")
            if not os.path.exists(model_path):
                print(f"[Warning] Model missing for fold {fold_idx}: {model_path}")
                continue

            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            with torch.no_grad():
                out = model(clinical_feat=clin_tensor, mri_feat=mri_tensor, wsi_feat=wsi_tensor)
                pmf_all_folds.append(out.cpu().numpy())

        if not pmf_all_folds:
            raise RuntimeError("No models loaded for ensemble inference.")

        avg_pmf = np.mean(pmf_all_folds, axis=0)
    else:
        # === SINGLE BEST MODEL ===
        # Load fold_cindices.csv
        cindex_path = os.path.join(GLOBAL_DIR, "fold_cindices.csv")
        if not os.path.exists(cindex_path):
            raise FileNotFoundError(f"Missing file: {cindex_path}")

        df = pd.read_csv(cindex_path)
        df = df[df["Fold"].str.contains("Fold", na=False)]
        best_idx = df["C-Index"].astype(float).idxmax()
        best_fold = int(df.iloc[best_idx]["Fold"].split()[1])

        model_path = os.path.join(GLOBAL_DIR, f"best_model_fold{best_fold-1}.pt")
        print(f"Using best single model: Fold {best_fold} → {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            avg_pmf = model(clinical_feat=clin_tensor, mri_feat=mri_tensor, wsi_feat=wsi_tensor).cpu().numpy()

    # Expected time = sum_t p(t) * t
    time_bins = np.arange(TIME_BINS)
    expected_times = np.sum(avg_pmf * time_bins[None, :], axis=1)

    return dict(zip(test_case_ids, expected_times))

if __name__ == "__main__":
    os.makedirs(GLOBAL_DIR, exist_ok=True)
    ## Do inference on all the training set using an ensemble or the best of the 5 folds
    print("\n=== Step: Inference on entire training set ===")
    
    clinical_df = load_clinical()
    all_case_ids = clinical_df['Case_ID'].astype(str).tolist()
    #all_case_ids = all_case_ids[:1]   #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    predictions = inference(all_case_ids)

    # Save predictions
    os.makedirs(GLOBAL_DIR, exist_ok=True)

    output_df = pd.DataFrame({
        'Case_ID': list(predictions.keys()),
        'Predicted_Time': list(predictions.values())
    })

    output_df = output_df.sort_values(by="Predicted_Time", ascending=False)

    save_path = os.path.join(GLOBAL_DIR, "ensemble_train_predictions.csv")
    output_df.to_csv(save_path, index=False)

    print(f"Ensemble predictions saved to: {save_path}")

    # Get true durations and events
    clinical_df_sorted = clinical_df[clinical_df['Case_ID'].isin(all_case_ids)].sort_values('Case_ID')
    durations = clinical_df_sorted['duration'].values
    events = clinical_df_sorted['event'].values
    preds = np.array([predictions[cid] for cid in clinical_df_sorted['Case_ID'].tolist()])

    # Compute C-index
    ## sksurv concordance function expects a risk score but as the challenge expect the algorithms to return time-to-event therefore, their code negates the time-to-event so it becomes a risk score
    ## But if the algorithm is returning a risk score then it should be negated before hand so that the negation of their code is cancelled out and the risk score remains the risk score.
 
    if SURVIVAL_MODEL == 'cox':
        preds = -preds
    
    c_index = concordance_index_censored( 
            events.astype(bool),
            durations,
            -preds
    )
    
    if USE_ENSEMBLE:
        print(f"\nEnsemble C-index on full training set: {c_index[0]:.4f}")
    else:
        print(f"\C-index on full training set using best of 5fold model: {c_index[0]:.4f}")