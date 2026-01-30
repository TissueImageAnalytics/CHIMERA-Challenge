import json
import os
from pprint import pprint
from pathlib import Path
import torch

import numpy as np

from glob import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler

import joblib
from post_challenge_data_utils import convert_mixed_column_to_numeric
from post_challenge_config import *

from post_challenge_model import MultimodalSurvivalModel
import h5py

## ==========User defined functions=============== ##
## Changes to the inference.py, assuming this will be the entry point

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")

SURVIVAL_WEIGHTS_PATH = Path("/opt/app/resources/clinv2_wsi")
SCALER_WEIGHTS_PATH = Path("/opt/app/resources/clinv2_wsi")


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))
        

def get_or_fit_scaler(name, train_array, fit=True, fold=0):
    os.makedirs(SCALER_WEIGHTS_PATH, exist_ok=True)
    scaler_path = os.path.join(SCALER_WEIGHTS_PATH, f"{name}_scaler_run_8_{fold}.pkl")
    
    if fit:
        print("Fitting scaler for", name)
        scaler = StandardScaler()
        scaler.fit(train_array)
        joblib.dump(scaler, scaler_path)
    else:
        print("Loading scaler for", name)
        scaler = joblib.load(scaler_path)
    
    return scaler


def maybe_scale(name, train_array, val_array, fit=False, fold=0):
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

    # Apply mixed column conversion
    for col in MIXED_COLS:
        if col in df.columns:
            df[col] = convert_mixed_column_to_numeric(df[col])

    # Convert all clinical features to numeric, coerce errors to NaN
    df[CLINICAL_FEATURES] = df[CLINICAL_FEATURES].apply(pd.to_numeric, errors='coerce')

    if df[CLINICAL_FEATURES].isnull().any().any():
        raise ValueError("Missing or invalid clinical feature(s) in JSON input")

    return df.values.astype(np.float32)


def extract_WSI_feats():

    wsi_dir = INPUT_PATH / "images/prostatectomy-wsi"

    wsi_path_list = glob(str(wsi_dir / "*.tif")) + glob(str(wsi_dir / "*.tiff")) + glob(str(wsi_dir / "*.svs")) + glob(str(wsi_dir / "*.ndpi"))

    pprint("WSI files found:")
    pprint(wsi_path_list)

    # select the first WSI
    wsi_path = wsi_path_list[0]
    pprint(f"Selected WSI: {wsi_path}")

    # TRIDENT Features
    try:
        trident_dir = OUTPUT_PATH / "trident_processed"
        trident_slide_features_titan_dir = trident_dir / "10x_1024px_0px_overlap" / "slide_features_titan"
        pprint(f"TRIDENT slide features directory: {trident_slide_features_titan_dir}")
        pprint(os.listdir(trident_slide_features_titan_dir))
        wsi_features_list = glob(str(trident_slide_features_titan_dir / "*.h5"))
        wsi_feature_path = wsi_features_list[0]
        with h5py.File(wsi_feature_path, 'r') as f:
            features = f['features'][()]
            features = torch.tensor(features, dtype=torch.float32)

        pprint("WSI features extracted.")

        if features.ndim == 1:
            features = features.reshape(1, -1)

        return features

    except Exception as e:
        pprint(f"Error occurred while reading TRIDENT features: {e}")


## ==========Challenge functions=============== ##
## Changes to the inference.py/interf9_handler(), assuming this will be the entry point

## I assume we would need to implement all (i.e. from 0 to 9) of the below handlers but just put the last one

def predict_score(clin_feats, mri_feats, wsi_feats):
    """Predict the score using the model.
    
    Args:
        clin_feats (torch.Tensor): Clinical features.
        mri_feats (torch.Tensor): MRI features.
        wsi_feats (torch.Tensor): WSI features.
    
    Returns:
        float: Predicted score.
    """

    ### Combine radiomic and clinical features
    m_dim = 0
    clin_dim = 2
    w_dim = 768 # For TITAN!


    # Prepare model template
    model = MultimodalSurvivalModel(
        clin_dim=clin_dim,
        mri_dim=m_dim,
        wsi_dim=w_dim,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if USE_ENSEMBLE:
        pmf_all_folds = []

        for fold_idx in range(NUM_FOLDS):
            model_path = os.path.join(SURVIVAL_WEIGHTS_PATH, f"best_model_run8_fold{fold_idx}.pt")
            if not os.path.exists(model_path):
                pprint(f"[Warning] Model missing for fold {fold_idx}: {model_path}")
                continue
            

            fold_clin_array, _ = maybe_scale("clinical", clin_feats, clin_feats, fit=False, fold=fold_idx) if USE_CLINICAL_FEATURES else (None, None)
            fold_mri_array, _ = None, None
            fold_wsi_array, _ = maybe_scale("wsi", wsi_feats, wsi_feats, fit=False, fold=fold_idx) if USE_WSI_FEATURES else (None, None)

            if fold_clin_array is not None and fold_clin_array.ndim == 1:
                fold_clin_array = fold_clin_array.reshape(1, -1)
            if fold_mri_array is not None and fold_mri_array.ndim == 1:
                fold_mri_array = fold_mri_array.reshape(1, -1)
            if fold_wsi_array is not None and fold_wsi_array.ndim == 1:
                fold_wsi_array = fold_wsi_array.reshape(1, -1)

            clin_tensor = torch.tensor(fold_clin_array, dtype=torch.float32).to(device) if fold_clin_array is not None else torch.zeros((1, clin_dim), device=device)
            mri_tensor = None
            wsi_tensor = torch.tensor(fold_wsi_array, dtype=torch.float32).to(device) if fold_wsi_array is not None else torch.zeros((1, w_dim), device=device)


            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            with torch.no_grad():
                out = model(clinical_feat=clin_tensor, mri_feat=None, wsi_feat=wsi_tensor)
                pmf_all_folds.append(out.cpu().numpy())

        if not pmf_all_folds:
            raise RuntimeError("No models loaded for ensemble inference.")


        avg_pmf = np.mean(pmf_all_folds, axis=0)



    time_bins = np.arange(TIME_BINS)
    expected_times = np.sum(avg_pmf * time_bins[None, :], axis=1)

    return expected_times[0]

def generic_handler():      
    input_chimera_clinical_data_of_prostate_cancer_patients = INPUT_PATH / "chimera-clinical-data-of-prostate-cancer-patients.json"
    with open(input_chimera_clinical_data_of_prostate_cancer_patients, "r") as f:
        jsdata = json.load(f)
    clin_feats = extract_clinical_vector(jsdata)
    wsi_feats = extract_WSI_feats() ## user defined function, returns a single vector for the whole case

    
    output_time_to_biochemical_recurrence_for_prostate_cancer = predict_score(clin_feats, None, wsi_feats)
    pprint(f"Predicted time: {output_time_to_biochemical_recurrence_for_prostate_cancer}")

    write_json_file(
        location=OUTPUT_PATH
        / "time-to-biochemical-recurrence-for-prostate-cancer-months.json",
        content=output_time_to_biochemical_recurrence_for_prostate_cancer,
    )

    return 0


if __name__ == "__main__":
    generic_handler()