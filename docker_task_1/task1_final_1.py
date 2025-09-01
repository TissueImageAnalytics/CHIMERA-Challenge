import os
import numpy as np
import pandas as pd
import torch
from config_final_1 import *
from data_utils_final_1 import convert_mixed_column_to_numeric
from model_final_1 import MultimodalSurvivalModel
from sklearn.preprocessing import StandardScaler
import joblib
import json
import SimpleITK as sitk
from pathlib import Path
from glob import glob
from pprint import pprint
import h5py

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")
SURVIVAL_WEIGHTS_PATH = Path("/opt/app/resources/weights_scalers_folder")
SCALER_WEIGHTS_PATH = Path("/opt/app/resources/weights_scalers_folder")

def get_or_fit_scaler(name, train_array, fit=True, run=0, fold=0):
    os.makedirs(SCALER_WEIGHTS_PATH, exist_ok=True)
    scaler_path = os.path.join(SCALER_WEIGHTS_PATH, f"{name}_scaler_run_{run}_{fold}.pkl")
    
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

def load_clinical_vectors_from_jsons(case_ids):
    """
    Given a list of case IDs, loads their clinical JSONs,
    applies preprocessing (including mixed-column parsing),
    and returns a NumPy array of shape (num_cases, num_features).
    """
    
    all_vectors = []
    for cid in sorted(case_ids):
        json_path = os.path.join(CLINICAL_JSON_DIR, f"{cid}.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Missing clinical JSON file for case: {cid}")

        with open(json_path, "r") as f:
            jsdata = json.load(f)

        # Extract raw feature values
        features = []
        for feat in CLINICAL_FEATURES:
            value = jsdata.get(feat, None)
            features.append(value)

        # Wrap into DataFrame for conversion
        df = pd.DataFrame([features], columns=CLINICAL_FEATURES)

        # Apply mixed column conversion
        for col in MIXED_COLS:
            if col in df.columns:
                df[col] = convert_mixed_column_to_numeric(df[col])

        # Convert to float
        df[CLINICAL_FEATURES] = df[CLINICAL_FEATURES].apply(pd.to_numeric, errors='coerce')

        if df[CLINICAL_FEATURES].isnull().any().any():
            raise ValueError(f"Missing or non-numeric value in clinical features for case {cid}")

        all_vectors.append(df.values[0])  # shape: (num_features,)

    return np.stack(all_vectors)  # shape: (num_cases, num_features)

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


def extract_radiomic_feats():
    # state_dict = torch.load(model_dir / "a_tarball_subdirectory" / "MRI_model_wts.pt", map_location='cpu')
    # model.load_state_dict(state_dict)
    # ...
    # mri = model(volumne_tensor)
    # return mri
    t2_mask_dir = INPUT_PATH / "images/prostate-tissue-mask-for-axial-t2-prostate-mri"
    t2_mask_path_list = glob(str(t2_mask_dir / "*.mha"))

    pprint("MRI files found:")
    pprint(t2_mask_path_list)

    # select the first mask
    mask_path = t2_mask_path_list[0]
    mask_img = sitk.ReadImage(mask_path)

    # Calculate volume
    spacing = mask_img.GetSpacing()
    voxel_volume = np.prod(spacing)
    mask_array = sitk.GetArrayFromImage(mask_img)
    total_volume = float(np.sum(mask_array > 0) * voxel_volume)  

    print("Radiomic features extracted.")

    total_volume = np.array([total_volume], dtype=np.float32)  # Convert to numpy array
    # Convert to [1,1] shape for consistency
    if total_volume.ndim == 1:
        total_volume = total_volume.reshape(1, 1)

    return total_volume


def extract_WSI_feats():

    wsi_dir = INPUT_PATH / "images/prostatectomy-wsi"

    wsi_path_list = glob(str(wsi_dir / "*.tif")) + glob(str(wsi_dir / "*.tiff")) + glob(str(wsi_dir / "*.svs")) + glob(str(wsi_dir / "*.ndpi"))

    pprint("WSI files found:")
    pprint(wsi_path_list)

    # select the first WSI
    wsi_path = wsi_path_list[0]
    pprint(f"Selected WSI: {wsi_path}")


    trident_dir = OUTPUT_PATH / "trident_processed"
    trident_slide_features_titan_dir = trident_dir / "10x_1024px_0px_overlap" / "slide_features_titan"
    print(f"TRIDENT slide features directory: {trident_slide_features_titan_dir}")
    print(os.listdir(trident_slide_features_titan_dir))
    wsi_features_list = glob(str(trident_slide_features_titan_dir / "*.h5"))
    wsi_feature_path = wsi_features_list[0]
    with h5py.File(wsi_feature_path, 'r') as f:
        features = f['features'][()]
        features = torch.tensor(features, dtype=torch.float32)

    print("WSI features extracted.")

    if features.ndim == 1:
        features = features.reshape(1, -1)

    return features





def inference():

    input_chimera_clinical_data_of_prostate_cancer_patients = INPUT_PATH / "chimera-clinical-data-of-prostate-cancer-patients.json"
    with open(input_chimera_clinical_data_of_prostate_cancer_patients, "r") as f:
        jsdata = json.load(f)

    test_clin_array = extract_clinical_vector(jsdata)
        


    if USE_RADIOMIC_FEATURES:
        test_radiomic_array = extract_radiomic_feats()

    test_mri_array = None

    test_wsi_array = extract_WSI_feats()

    # Feature dimensions
    c_dim = test_clin_array.shape[1] if (USE_CLINICAL_FEATURES and test_clin_array is not None) else 0
    m_dim = test_mri_array.shape[1] if (USE_MRI_FEATURES and test_mri_array is not None) else 0
    w_dim = test_wsi_array.shape[1] if (USE_WSI_FEATURES and test_wsi_array is not None) else 0

    r_dim = test_radiomic_array.shape[1] if (USE_RADIOMIC_FEATURES and test_radiomic_array is not None) else 0
    
    if r_dim > 0:
        c_dim += r_dim

    # Prepare model template
    model = MultimodalSurvivalModel(
        clin_dim=c_dim,
        mri_dim=m_dim,
        wsi_dim=w_dim,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)



    best_run = 8

    if USE_ENSEMBLE:
        pmf_all_folds = []

        for fold_idx in range(NUM_FOLDS):
            model_path = os.path.join(SURVIVAL_WEIGHTS_PATH, f"best_model_run{best_run}_fold{fold_idx}.pt")

            if not os.path.exists(model_path):
                print(f"[Warning] Model missing for fold {fold_idx}: {model_path}")
                continue

            fold_clin_array, _ = maybe_scale("clinical", test_clin_array, test_clin_array, fit=False, run=best_run, fold=fold_idx) if USE_CLINICAL_FEATURES else (None, None)
            fold_radi_array, _ = maybe_scale("radiomic", test_radiomic_array, test_radiomic_array, fit=False, run=best_run, fold=fold_idx) if USE_RADIOMIC_FEATURES else (None, None)

            # Concatenate radiomic features to clinical AFTER scaling
            if USE_RADIOMIC_FEATURES:
                if fold_clin_array is not None:
                    fold_clin_array = np.concatenate([fold_clin_array, fold_radi_array], axis=1)
                else:
                    fold_clin_array = fold_radi_array

            fold_mri_array, _ = maybe_scale("mri", test_mri_array, test_mri_array, fit=False, run=best_run, fold=fold_idx) if USE_MRI_FEATURES else (None, None)
            fold_wsi_array, _ = maybe_scale("wsi", test_wsi_array, test_wsi_array, fit=False, run=best_run, fold=fold_idx) if USE_WSI_FEATURES else (None, None)

            if fold_clin_array is not None and fold_clin_array.ndim == 1:
                fold_clin_array = fold_clin_array.reshape(1, -1)
            if fold_mri_array is not None and fold_mri_array.ndim == 1:
                fold_mri_array = fold_mri_array.reshape(1, -1)
            if fold_wsi_array is not None and fold_wsi_array.ndim == 1:
                fold_wsi_array = fold_wsi_array.reshape(1, -1)

            clin_tensor = torch.tensor(fold_clin_array, dtype=torch.float32).to(device) if fold_clin_array is not None else torch.zeros((1, c_dim), device=device)
            mri_tensor = torch.tensor(fold_mri_array, dtype=torch.float32).to(device) if fold_mri_array is not None else torch.zeros((1, m_dim), device=device)
            wsi_tensor = torch.tensor(fold_wsi_array, dtype=torch.float32).to(device) if fold_wsi_array is not None else torch.zeros((1, w_dim), device=device)

            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            with torch.no_grad():
                out = model(clinical_feat=clin_tensor, mri_feat=mri_tensor, wsi_feat=wsi_tensor)
                pmf_all_folds.append(out.cpu().numpy())

        if not pmf_all_folds:
            raise RuntimeError("No models loaded for ensemble inference.")

        assert all(p.shape == pmf_all_folds[0].shape for p in pmf_all_folds), \
            "Mismatch in PMF shape across folds!"

        avg_pmf = np.mean(pmf_all_folds, axis=0)



    time_bins = np.arange(TIME_BINS)
    score = float(np.sum(avg_pmf * time_bins))
    print(f"Score:  {score}")
    return score


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def generic_handler():      

    output_time_to_biochemical_recurrence_for_prostate_cancer = inference()

    print(f"Predicted time: {output_time_to_biochemical_recurrence_for_prostate_cancer}")

    write_json_file(
        location=OUTPUT_PATH
        / "time-to-biochemical-recurrence-for-prostate-cancer-months.json",
        content=output_time_to_biochemical_recurrence_for_prostate_cancer,
    )

    return 0


if __name__ == "__main__":
    generic_handler()