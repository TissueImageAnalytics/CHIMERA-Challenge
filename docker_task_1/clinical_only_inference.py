from data_utils import convert_mixed_column_to_numeric
from config import *
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
from model import MultimodalSurvivalModel
import torch

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("/opt/app/resources")
CLINICAL_MODEL_DIR = Path("/opt/app/resources/task1_submission_clinical")


def get_or_fit_scaler(name, train_array, fit=True, fold=0):
    os.makedirs(CLINICAL_MODEL_DIR, exist_ok=True)
    scaler_path = os.path.join(CLINICAL_MODEL_DIR, f"{name}_scaler_{fold}.pkl")

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


def extract_clinical_vector(jsdata):
    """
    Reads clinical features from a single JSON object (already loaded),
    applies necessary preprocessing (including mixed column handling),
    and returns a (1, num_features) float32 NumPy array.
    """
 
    # Extract raw feature values from JSON
    features = [jsdata.get(feat, None) for feat in CLINICAL_FEATURES]
 
    # Wrap into a DataFrame for consistent preprocessing
    df = pd.DataFrame([features], columns=CLINICAL_FEATURES)
 
    if df[CLINICAL_FEATURES].isnull().any().any():
        # raise ValueError("Missing or invalid clinical feature(s) in JSON input")
        df[CLINICAL_FEATURES] = df[CLINICAL_FEATURES].fillna(0)

    # Apply mixed column conversion
    for col in MIXED_COLS:
        if col in df.columns:
            df[col] = convert_mixed_column_to_numeric(df[col])
    
    df[CLINICAL_FEATURES] = df[CLINICAL_FEATURES].replace("x", 0)
    # Convert all clinical features to numeric, coerce errors to NaN
    df[CLINICAL_FEATURES] = df[CLINICAL_FEATURES].apply(pd.to_numeric, errors='coerce')
 
 
    return df.values.astype(np.float32)
 

def inference(clinical_json_path):
    # # Load clinical data and get feature dims
    # clinical_df = load_clinical()
    # clinical_dim = get_feature_dimensionalities(clinical_df)
 
    # # Get input arrays
    # test_clinical = clinical_df[clinical_df['Case_ID'].isin(test_case_ids)].sort_values('Case_ID')
    # test_clin_array = test_clinical.drop(columns=['Case_ID', 'duration', 'event']).values if USE_CLINICAL_FEATURES else None


    input_chimera_clinical_data_of_prostate_cancer_patients = clinical_json_path
    with open(input_chimera_clinical_data_of_prostate_cancer_patients, "r") as f:
       jsdata = json.load(f)
 
    test_clin_array = extract_clinical_vector(jsdata)
    clinical_dim = test_clin_array.shape[1]

    test_mri_array = None
    test_wsi_array = None
 
   
 
    # Feature dimensions
    c_dim = clinical_dim if USE_CLINICAL_FEATURES else 0
    m_dim = 0
    w_dim = 0
 
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
 
    
    
    if USE_ENSEMBLE:
        # === ENSEMBLE: Load all folds ===
        pmf_all_folds = []
 
        for fold_idx in range(NUM_FOLDS):
            # === Apply same scalers ===
            if SCALE_DATA:
                if USE_CLINICAL_FEATURES:
                    fold_clin_array, _ = maybe_scale("clinical", test_clin_array, test_clin_array, fit=False, fold=fold_idx)
                if USE_MRI_FEATURES:
                    fold_mri_array, _ = maybe_scale("mri", test_mri_array, test_mri_array, fit=False, fold=fold_idx)
                if USE_WSI_FEATURES:
                    fold_wsi_array, _ = maybe_scale("wsi", test_wsi_array, test_wsi_array, fit=False, fold=fold_idx)
            # Prepare input tensors
            N = 1
            fold_clin_tensor = torch.tensor(fold_clin_array, dtype=torch.float32).to(device) if fold_clin_array is not None else torch.zeros((N, c_dim), device=device)
            fold_mri_tensor = None
            fold_wsi_tensor = None

            model_path = os.path.join(CLINICAL_MODEL_DIR, f"best_model_fold{fold_idx}.pt")
            if not os.path.exists(model_path):
                print(f"[Warning] Model missing for fold {fold_idx}: {model_path}")
                continue
 
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
 
            with torch.no_grad():
                out = model(clinical_feat=fold_clin_tensor, mri_feat=fold_mri_tensor, wsi_feat=fold_wsi_tensor)
                pmf_all_folds.append(out.cpu().numpy())
 
        if not pmf_all_folds:
            raise RuntimeError("No models loaded for ensemble inference.")
 
        avg_pmf = np.mean(pmf_all_folds, axis=0)
    else:
        raise NotImplementedError("Single model inference not implemented yet")
    # else:
    #     # === SINGLE BEST MODEL ===
    #     # Load fold_cindices.csv
    #     cindex_path = os.path.join(GLOBAL_DIR, "fold_cindices.csv")
    #     if not os.path.exists(cindex_path):
    #         raise FileNotFoundError(f"Missing file: {cindex_path}")
 
    #     df = pd.read_csv(cindex_path)
    #     df = df[df["Fold"].str.contains("Fold", na=False)]
    #     best_idx = df["C-Index"].astype(float).idxmax()
    #     best_fold = int(df.iloc[best_idx]["Fold"].split()[1])
 
    #     model_path = os.path.join(GLOBAL_DIR, f"best_model_fold{best_fold-1}.pt")
    #     print(f"Using best single model: Fold {best_fold} → {model_path}")
    #     model.load_state_dict(torch.load(model_path, map_location=device))
    #     model.eval()
 
    #     with torch.no_grad():
    #         avg_pmf = model(clinical_feat=clin_tensor, mri_feat=mri_tensor, wsi_feat=wsi_tensor).cpu().numpy()
 
    time_bins = np.arange(TIME_BINS)
    score = float(np.sum(avg_pmf * time_bins))
    print('score: ', score)
    return score









def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))

    

# def extract_clinical_feats():
#     """Read clinical data from JSON file.
#         Simply returns the clinical data as a dictionary.
#     Args:
#         json_file_path (str): Path to the JSON file containing clinical data.
#     Returns:
#         dict: Clinical data.
#     """
#     with open(INPUT_PATH / "chimera-clinical-data-of-prostate-cancer-patients.json", "r") as f:
#         clinical_data = json.loads(f.read())

#     pprint("Clinical data found:")
#     pprint(clinical_data)
#     return clinical_data





def generic_handler():      

    clinical_json_path = INPUT_PATH / "chimera-clinical-data-of-prostate-cancer-patients.json"

    output_time_to_biochemical_recurrence_for_prostate_cancer = inference(clinical_json_path)

    write_json_file(
        location=OUTPUT_PATH
        / "time-to-biochemical-recurrence-for-prostate-cancer-months.json",
        content=output_time_to_biochemical_recurrence_for_prostate_cancer,
    )

    return 0


if __name__ == "__main__":
    generic_handler()