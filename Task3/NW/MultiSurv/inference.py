import os
import numpy as np
import pandas as pd
import torch
from config import *
from utils.data_utils import load_clinical, load_rna_features, load_wsi_features, load_wsi_features_mult, encode_clinical_features
#from utils.bins import expected_time_from_pmf
from models.model import MultimodalSurvivalModel
from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import StandardScaler
import joblib
import json

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

def extract_clinical_vector(jsdata):
    """
    Accepts a single JSON dict with clinical data.
    Encodes categorical variables using same mappings as training,
    handles numeric/mixed columns, and returns (1, num_features) float32 array.
    """

    # Convert JSON into DataFrame
    df = pd.DataFrame([jsdata], columns=CLINICAL_FEATURES)

    # Apply categorical encoding
    df_encoded = encode_clinical_features(df)

    # Ensure all values are numeric
    df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce')

    # Check for missing values
    if df_encoded.isnull().any().any():
        missing_cols = df_encoded.columns[df_encoded.isnull().any()].tolist()
        raise ValueError(f"Missing or invalid clinical feature(s) in JSON input: {missing_cols}")

    return df_encoded.values.astype(np.float32)

def inference(test_case_ids):
    # Load clinical data and get feature dims
    clinical_df = load_clinical()
    #clinical_dim = get_feature_dimensionalities(clinical_df)

    # Sort test case IDs to preserve consistent output order
    test_case_ids = sorted(test_case_ids)

    # Get input arrays
    test_clinical = clinical_df[clinical_df['Case_ID'].isin(test_case_ids)].sort_values('Case_ID')
    
    test_clin_array = test_clinical.drop(columns=['Case_ID', 'duration', 'event']).values if USE_CLINICAL_FEATURES else None

    #### for single case from clincal json @@@@@@@@@@@@@@@@@@@@@@@@@@
    if INFER_SINGLE:
        test_case = os.listdir(CLINICAL_JSON_DIR)[0]
        input_chimera_clinical_data_of_prostate_cancer_patients = f"{CLINICAL_JSON_DIR}/{test_case}/{test_case}_CD.json"
        with open(input_chimera_clinical_data_of_prostate_cancer_patients, "r") as f:
           jsdata = json.load(f)

        test_clin_array = extract_clinical_vector(jsdata)
        
        ## ToDo: change the RNA and WSI feature loading to challenge format
        ## Temp work around: change the test_case_ids as in the next line
        test_case_ids = [test_case.split('.json')[0]]
    
    test_rna_array = load_rna_features(test_case_ids) if USE_RNA_FEATURES else None

    if AGGREG_CASE_WSI:
        test_wsi_array = load_wsi_features_mult(test_case_ids, csv_path=WSI_FEATURES_CSV) if USE_WSI_FEATURES else None
    else:
        test_wsi_array = load_wsi_features(test_case_ids, csv_path=WSI_FEATURES_CSV) if USE_WSI_FEATURES else None

    # Feature dimensions
    c_dim = test_clin_array.shape[1] if (USE_CLINICAL_FEATURES and test_clin_array is not None) else 0
    r_dim = test_rna_array.shape[1] if (USE_RNA_FEATURES and test_rna_array is not None) else 0
    w_dim = test_wsi_array.shape[1] if (USE_WSI_FEATURES and test_wsi_array is not None) else 0

    # Prepare model template
    model = MultimodalSurvivalModel(
        clin_dim=c_dim,
        rna_dim=r_dim,
        wsi_dim=w_dim,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_run_path = os.path.join(GLOBAL_DIR, "best_run.json")
    if not os.path.exists(best_run_path):
        raise FileNotFoundError(f"Missing {best_run_path}. Please train models first.")
    with open(best_run_path, "r") as f:
        best_run_info = json.load(f)

    best_run = best_run_info["best_run"]
    fold_cindices = best_run_info["fold_cindices"]

    if USE_ENSEMBLE:
        pmf_all_folds = []

        for fold_idx in range(NUM_FOLDS):
            model_path = os.path.join(GLOBAL_DIR, f"best_model_run{best_run}_fold{fold_idx}.pt")

            if not os.path.exists(model_path):
                print(f"[Warning] Model missing for fold {fold_idx}: {model_path}")
                continue

            fold_clin_array, _ = maybe_scale("clinical", test_clin_array, test_clin_array, fit=False, run=best_run, fold=fold_idx) if USE_CLINICAL_FEATURES else (None, None)
            fold_rna_array, _ = maybe_scale("rna", test_rna_array, test_rna_array, fit=False, run=best_run, fold=fold_idx) if USE_RNA_FEATURES else (None, None)
            fold_wsi_array, _ = maybe_scale("wsi", test_wsi_array, test_wsi_array, fit=False, run=best_run, fold=fold_idx) if USE_WSI_FEATURES else (None, None)

            if fold_clin_array is not None and fold_clin_array.ndim == 1:
                fold_clin_array = fold_clin_array.reshape(1, -1)
            if fold_rna_array is not None and fold_rna_array.ndim == 1:
                fold_rna_array = fold_rna_array.reshape(1, -1)
            if fold_wsi_array is not None and fold_wsi_array.ndim == 1:
                fold_wsi_array = fold_wsi_array.reshape(1, -1)

            clin_tensor = torch.tensor(fold_clin_array, dtype=torch.float32).to(device) if c_dim > 0 else None
            rna_tensor  = torch.tensor(fold_rna_array, dtype=torch.float32).to(device) if r_dim > 0 else None
            wsi_tensor  = torch.tensor(fold_wsi_array, dtype=torch.float32).to(device) if w_dim > 0 else None

            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            with torch.no_grad():
                out = model(clinical_feat=clin_tensor, rna_feat=rna_tensor, wsi_feat=wsi_tensor)
                pmf_all_folds.append(out.cpu().numpy())

        if not pmf_all_folds:
            raise RuntimeError("No models loaded for ensemble inference.")
        
        assert all(p.shape == pmf_all_folds[0].shape for p in pmf_all_folds), \
            "Mismatch in PMF shape across folds!"

        avg_pmf = np.mean(pmf_all_folds, axis=0)

    else: ## This is not updated and shold not be used at the moment.
        # Load best run info
        best_run_path = os.path.join(GLOBAL_DIR, "best_run.json")
        if not os.path.exists(best_run_path):
            raise FileNotFoundError(f"Missing {best_run_path}. Please train models first.")
        with open(best_run_path, "r") as f:
            best_run_info = json.load(f)

        best_run = best_run_info["best_run"]
        fold_cindices = best_run_info["fold_cindices"]
        best_fold = int(np.argmax(fold_cindices))  # index of best fold

        model_path = os.path.join(GLOBAL_DIR, f"best_model_run{best_run}_fold{best_fold}.pt")
        print(f"Using best single model: Run {best_run}, Fold {best_fold}, {model_path}")

        test_clin_array, _ = maybe_scale("clinical", test_clin_array, test_clin_array, fit=False, run=0, fold=best_fold) if USE_CLINICAL_FEATURES else (None, None)
        test_rna_array, _ = maybe_scale("rna", test_rna_array, test_rna_array, fit=False, run=0, fold=best_fold) if USE_RNA_FEATURES else (None, None)
        test_wsi_array, _ = maybe_scale("wsi", test_wsi_array, test_wsi_array, fit=False, run=0, fold=best_fold) if USE_WSI_FEATURES else (None, None)

        if test_clin_array is not None and test_clin_array.ndim == 1:
            test_clin_array = test_clin_array.reshape(1, -1)
        if test_rna_array is not None and test_rna_array.ndim == 1:
            test_rna_array = test_rna_array.reshape(1, -1)
        if test_wsi_array is not None and test_wsi_array.ndim == 1:
            test_wsi_array = test_wsi_array.reshape(1, -1)

        clin_tensor = torch.tensor(fold_clin_array, dtype=torch.float32).to(device) if c_dim > 0 else None
        rna_tensor  = torch.tensor(fold_rna_array, dtype=torch.float32).to(device) if r_dim > 0 else None
        wsi_tensor  = torch.tensor(fold_wsi_array, dtype=torch.float32).to(device) if w_dim > 0 else None

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            avg_pmf = model(clinical_feat=clin_tensor, rna_feat=rna_tensor, wsi_feat=wsi_tensor)[0].cpu().numpy()

    ## to test with a single json clinical file
    if INFER_SINGLE:
        time_bins = np.arange(TIME_BINS)
        score = float(np.sum(avg_pmf * time_bins))
        print(f"Score for case {test_case}:  {score}")
        exit()

    ##Expected time = sum_t p(t) * t
    time_bins = np.arange(TIME_BINS)
    #bin_edges = np.load(f"{GLOBAL_DIR}edges.npy")
    expected_times = np.sum(avg_pmf * time_bins[None, :], axis=1)
    #expected_times = expected_time_from_pmf(avg_pmf, bin_edges)

    return dict(zip(test_case_ids, expected_times))

if __name__ == "__main__":
    os.makedirs(GLOBAL_DIR, exist_ok=True)
    ## Do inference on all the training set using an ensemble or the best of the 5 folds
    print("\n=== Step: Inference on entire training set ===")

    ## load held-out test set IDs
    #test_df = pd.read_csv('/home/u1970167/chimera/task3/experiments/folds/test.csv')
    #all_case_ids = test_df['Case_ID'].astype(str).tolist()

    # load all the train set
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

    print(f"Inference predictions saved to: {save_path}")

    # Get true durations and events
    clinical_df_sorted = clinical_df[clinical_df['Case_ID'].isin(all_case_ids)].sort_values('Case_ID')
    durations = clinical_df_sorted['duration'].values
    events = clinical_df_sorted['event'].values
    print(clinical_df_sorted.head(5))
    preds = np.array([predictions[cid] for cid in clinical_df_sorted['Case_ID'].tolist()])

    # Compute C-index
    ## sksurv concordance function expects a risk score but as the challenge expect the algorithms to return time-to-event therefore, their code negates the time-to-event so it becomes a risk score
    ## But if the algorithm is returning a risk score then it should be negated before hand so that the negation of their code is cancelled out and the risk score remains the risk score.
 
    if SURVIVAL_MODEL == 'cox' or SURVIVAL_MODEL == 'deepsurv':
        preds = -preds
    
    c_index = concordance_index_censored( 
            events.astype(bool),
            durations,
            -preds
    )
    
    if USE_ENSEMBLE:
        print(f"\nEnsemble C-index on full training set: {c_index[0]:.4f}")
    else:
        print(f"\nC-index on full training set using best of 5fold model: {c_index[0]:.4f}")