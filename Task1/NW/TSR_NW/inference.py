import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
from TransductiveSR import TransductiveSR as TSRR
import re

# === Configuration ===
TASK = 1
USE_CLINICAL = True
USE_MRI = False
USE_WSI = True
USE_ENSEMBLE = True  # <-- Set to False to use best single model
EMBEDDER = 'prism' ## 'titan' or 'prism'
RUN = 0  # Only used when USE_ENSEMBLE = False
FOLDS = [0, 1, 2, 3, 4]
CENSORING = 120
TUNING_TRIALS = 1 ## used during training. this is needed here to load the required files for inference
RESULT_DIR = f"/home/u1970167/chimera/task1/experiments/TSR_results/Task_{TASK}_Clinical_{USE_CLINICAL}_MRI_{USE_MRI}_WSI_{USE_WSI}_{EMBEDDER}_TUNING_{TUNING_TRIALS}/"
FOLDS_DIR = "/home/u1970167/chimera/task1/experiments/folds/"

# === Task-Specific Paths ===
if TASK == 1:
    if EMBEDDER == "titan":
        WSI_FEATURE_PATH = "/home/u1970167/chimera/task1/pathology/features/titan/Task1_TITAN_1024_embeddings.csv"
    elif EMBEDDER == 'prism':
        WSI_FEATURE_PATH = "/home/u1970167/chimera/task1/pathology/features/prism/Task1_prism_224_embeddings.csv"

    CLINICAL_PATH = "/home/u1970167/chimera/task1/clinical_data.csv"
    TIME_COL = 'time_to_follow-up/BCR'
    EVENT_COL = 'BCR'
    SLIDE_ID_COL = 'Slide_ID'
    MRI_FEATURE_DIR = "/home/u1970167/chimera/task1/radiology/features/"
else:
    raise NotImplementedError("Inference script currently supports TASK=1 only")

# === Clinical Features ===
CLINICAL_FEATURES = [
    "age_at_prostatectomy",
    "primary_gleason",
    "secondary_gleason",
    "ISUP",
    "pre_operative_PSA",
    "capsular_penetration",
    "positive_surgical_margins",
    "invasion_seminal_vesicles",
    "lymphovascular_invasion",
    "pT_stage"
]

MIXED_COLS = ["pT_stage"] ## pT_stage has values such 2, 2a, 2b, 3 etc. these needs to be changed to values like 2.0, 2.1, 2.2, 3.0 etc repectively

EXCLUDE_COLS = ["Case_ID", TIME_COL, EVENT_COL, SLIDE_ID_COL]

def convert_mixed_column_to_numeric(series):
    def parse_value(val):
        match = re.match(r"(\d+)([a-zA-Z]*)", str(val))
        if match:
            base = int(match.group(1))
            suffix = match.group(2)
            if suffix:
                suffix_value = (ord(suffix.lower()) - ord('a') + 1) / 10
                return base + suffix_value
            else:
                return float(base)
        return None  # or np.nan
    return series.apply(parse_value)

# === Load Dataset ===
def load_dataset():
    clinical_data = pd.read_csv(CLINICAL_PATH)
    clinical_data["Case_ID"] = clinical_data["Case_ID"].astype(str)

    cols_to_use = ["Case_ID", TIME_COL, EVENT_COL]
    if USE_CLINICAL:
        cols_to_use += CLINICAL_FEATURES

    clinical_data = clinical_data[cols_to_use]

    # --- Convert mixed columns like 'pT_stage' ---
    for col in MIXED_COLS:
        if col in clinical_data.columns:
            clinical_data[col] = convert_mixed_column_to_numeric(clinical_data[col])

    # --- Ensure all clinical features are numeric ---
    for col in CLINICAL_FEATURES:
        if col in clinical_data.columns:
            clinical_data[col] = pd.to_numeric(clinical_data[col], errors='coerce')

    # --- Drop rows with NaNs in clinical features ---
    clinical_data = clinical_data.dropna(subset=[col for col in CLINICAL_FEATURES if col in clinical_data.columns])

    # --- Merge WSI features if needed ---
    if USE_WSI:
        slide_embeddings = pd.read_csv(WSI_FEATURE_PATH)
        slide_embeddings[SLIDE_ID_COL] = slide_embeddings[SLIDE_ID_COL].astype(str)
        slide_embeddings["Case_ID"] = slide_embeddings[SLIDE_ID_COL].str.split('_').str[0]
        embedding_cols = [col for col in slide_embeddings.columns if col.startswith("dim")]
        wsi_features = slide_embeddings.groupby("Case_ID")[embedding_cols].mean().reset_index()
        clinical_data = pd.merge(clinical_data, wsi_features, on="Case_ID", how="inner")

    # --- Final assembly ---
    df = clinical_data.copy()
    T = np.array(df[TIME_COL])
    E = np.array(df[EVENT_COL])

    E[T > CENSORING] = 0
    T[T > CENSORING] = CENSORING

    # Drop unused columns
    X = df.drop(columns=[col for col in EXCLUDE_COLS if col in df.columns]).values

    return X, T, E

# === Inference ===
def infer():
    X, T, E = load_dataset()
    preds = []

    if USE_ENSEMBLE:
        print("Running ensemble inference across all folds...")

        for fold in FOLDS:
            model_path = os.path.join(RESULT_DIR, f"model_TASK{TASK}_RUN{RUN}_FOLD{fold}.pkl")
            scaler_path = os.path.join(RESULT_DIR, f"scaler_TASK{TASK}_RUN{RUN}_FOLD{fold}.pkl")

            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            X_scaled = scaler.transform(X)
            Z = model.decision_function(X_scaled)
            preds.append(Z)

        Z_final = np.mean(preds, axis=0)

    else:
        print("Running single-model inference from fold 0...")
        model_path = os.path.join(RESULT_DIR, f"model_TASK{TASK}_RUN{RUN}_FOLD0.pkl")
        scaler_path = os.path.join(RESULT_DIR, f"scaler_TASK{TASK}_RUN{RUN}_FOLD0.pkl")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        X_scaled = scaler.transform(X)
        Z_final = model.decision_function(X_scaled)

    cindex_val, _, _, _, _ = concordance_index_censored(E.astype(bool), T, -Z_final)
    print(f"\nInference C-Index: {cindex_val:.4f}")

if __name__ == "__main__":
    infer()
