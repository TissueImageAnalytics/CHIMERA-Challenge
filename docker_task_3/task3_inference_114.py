import os
import numpy as np
import pandas as pd
import torch
from config114 import *

from model import MultimodalSurvivalModel
from sklearn.preprocessing import StandardScaler
import joblib
import json
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


def encode_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    mappings = {
        "sex": {"Male": 0, "Female": 1},
        "tumor": {"Primary": 0},
        "stage": {"T1HG": 1, "TaHG": 0},
        "grade": {"G2": 2, "G3": 3},
        "reTUR": {"No": 0, "Yes": 1},
        "variant": {"UCC": 0, "UCC + Variant": 1},
        "EORTC": {"High risk": 0, "Highest risk": 1},
        "BRS": {"BRS1": 1, "BRS2": 2, "BRS3": 3}
    }

    df_encoded = df.copy()

    for col, mapping in mappings.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype(str).str.strip()  # Remove extra whitespace
            df_encoded[col] = df_encoded[col].map(mapping)

    return df_encoded


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

    clinical_dim = get_feature_dimensionalities(df_encoded)
    print(f"Clinical feature dimensionality: {clinical_dim}")

    return df_encoded.values.astype(np.float32), clinical_dim



def load_wsi_features():

    wsi_dir = INPUT_PATH / "images/bladder-cancer-tissue-biopsy-wsi"

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

    if not os.path.exists(wsi_feature_path):
        raise FileNotFoundError(f"WSI feature file not found: {wsi_feature_path}")

    with h5py.File(wsi_feature_path, 'r') as f:
        features = f['features'][()]
        features = torch.tensor(features, dtype=torch.float32)

    print("WSI features extracted.")

    if features.ndim == 1:
        features = features.reshape(1, -1)

    return features


def get_feature_dimensionalities(clinical_df):
    clinical_features = clinical_df.drop(columns=['Case_ID', 'duration', 'event'], errors='ignore')
    clinical_dim = clinical_features.shape[1]
    return clinical_dim



def inference():
    input_chimera_clinical_data_of_prostate_cancer_patients = INPUT_PATH / "chimera-clinical-data-of-bladder-cancer-recurrence-patients.json"
    with open(input_chimera_clinical_data_of_prostate_cancer_patients, "r") as f:
        jsdata = json.load(f)

    test_clin_array, clinical_dim = extract_clinical_vector(jsdata)


    test_rna_array = None


    test_wsi_array = load_wsi_features()

    # Feature dimensions
    c_dim = clinical_dim if USE_CLINICAL_FEATURES else 0
    m_dim = RNA_DIM_REDUCE_TO if USE_RNA_FEATURES else 0
    w_dim = WSI_FEATURE_DIM if USE_WSI_FEATURES else 0

    # Prepare model template
    model = MultimodalSurvivalModel(
        clin_dim=c_dim,
        rna_dim=m_dim,
        wsi_dim=w_dim,
        fusion_type=FUSION_TYPE,
        survival_model=SURVIVAL_MODEL,
        time_bins=TIME_BINS
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    best_run = 4

    if USE_ENSEMBLE:
        pmf_all_folds = []

        for fold_idx in range(NUM_FOLDS):
            model_path = os.path.join(SURVIVAL_WEIGHTS_PATH, f"best_model_run{best_run}_fold{fold_idx}.pt")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"[Warning] Model missing for fold {fold_idx}: {model_path}")

            fold_clin_array, _ = maybe_scale("clinical", test_clin_array, test_clin_array, fit=False, run=best_run, fold=fold_idx) if USE_CLINICAL_FEATURES else (None, None)
            fold_rna_array, _ = maybe_scale("rna", test_rna_array, test_rna_array, fit=False, run=best_run, fold=fold_idx) if USE_RNA_FEATURES else (None, None)
            fold_wsi_array, _ = maybe_scale("wsi", test_wsi_array, test_wsi_array, fit=False, run=best_run, fold=fold_idx) if USE_WSI_FEATURES else (None, None)

            if fold_clin_array is not None and fold_clin_array.ndim == 1:
                fold_clin_array = fold_clin_array.reshape(1, -1)
            if fold_rna_array is not None and fold_rna_array.ndim == 1:
                fold_rna_array = fold_rna_array.reshape(1, -1)
            if fold_wsi_array is not None and fold_wsi_array.ndim == 1:
                fold_wsi_array = fold_wsi_array.reshape(1, -1)

            clin_tensor = torch.tensor(fold_clin_array, dtype=torch.float32).to(device) if fold_clin_array is not None else torch.zeros((1, c_dim), device=device)
            rna_tensor = torch.tensor(fold_rna_array, dtype=torch.float32).to(device) if fold_rna_array is not None else torch.zeros((1, m_dim), device=device)
            wsi_tensor = torch.tensor(fold_wsi_array, dtype=torch.float32).to(device) if fold_wsi_array is not None else torch.zeros((1, w_dim), device=device)

            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            with torch.no_grad():
                out = model(clinical_feat=clin_tensor, rna_feat=rna_tensor, wsi_feat=wsi_tensor)
                pmf_all_folds.append(out.cpu().numpy())

        if not pmf_all_folds:
            raise RuntimeError("No models loaded for ensemble inference.")

        avg_pmf = np.mean(pmf_all_folds, axis=0)
   


    time_bins = np.arange(TIME_BINS)
    score = float(np.sum(avg_pmf * time_bins))
    return score



def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def generic_handler():      

    output_likelihood_of_bladder_cancer_recurrence = inference()

    print(f"Predicted score: {output_likelihood_of_bladder_cancer_recurrence}")

    write_json_file(
        location=OUTPUT_PATH / "likelihood-of-bladder-cancer-recurrence.json",
        content=output_likelihood_of_bladder_cancer_recurrence,
    )

    return 0

if __name__ == "__main__":
    generic_handler()