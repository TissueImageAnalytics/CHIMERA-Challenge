import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
from TransductiveSR import TransductiveSR as TSRR
import re
from config import *

def convert_mixed_column_to_numeric(series):
    def parse_value(val):
        match = re.match(r"(\d+)([a-zA-Z]*)", str(val))
        if match:
            base = int(match.group(1))
            suffix = match.group(2)
            return base + ((ord(suffix.lower()) - ord('a') + 1) / 10) if suffix else float(base)
        return None
    return series.apply(parse_value)

def load_raw_dataframe():
    clinical_data = pd.read_csv(CLINICAL_PATH)
    clinical_data["Case_ID"] = clinical_data["Case_ID"].astype(str)

    cols_to_use = ["Case_ID", TIME_COL, EVENT_COL]
    if USE_CLINICAL:
        cols_to_use += CLINICAL_FEATURES

    clinical_data = clinical_data[cols_to_use]

    for col in MIXED_COLS:
        if col in clinical_data.columns:
            clinical_data[col] = convert_mixed_column_to_numeric(clinical_data[col])

    for col in CLINICAL_FEATURES:
        if col in clinical_data.columns:
            clinical_data[col] = pd.to_numeric(clinical_data[col], errors='coerce')

    clinical_data = clinical_data.dropna(subset=[col for col in CLINICAL_FEATURES if col in clinical_data.columns])

    if USE_WSI:
        slide_embeddings = pd.read_csv(WSI_FEATURE_PATH)
        slide_embeddings[SLIDE_ID_COL] = slide_embeddings[SLIDE_ID_COL].astype(str)
        slide_embeddings["Case_ID"] = slide_embeddings[SLIDE_ID_COL].str.split('_').str[0]
        embedding_cols = [col for col in slide_embeddings.columns if col.startswith("dim")]
        wsi_features = slide_embeddings.groupby("Case_ID")[embedding_cols].mean().reset_index()
        clinical_data = pd.merge(clinical_data, wsi_features, on="Case_ID", how="inner")

    return clinical_data

def split_and_scale_features(df, fit=False, scalers=None, return_scalers=False):
    df_copy = df.copy()
    modality_scalers = scalers if scalers else {}
    clinical_cols = [col for col in df.columns if col in CLINICAL_FEATURES]
    wsi_cols = [col for col in df.columns if col.startswith("dim")]
    mri_cols = [col for col in df.columns if col.startswith("mri_feat_")]

    for modality, cols in zip(["clinical", "wsi", "mri"], [clinical_cols, wsi_cols, mri_cols]):
        if cols:
            if fit:
                scaler = StandardScaler().fit(df_copy[cols])
                df_copy[cols] = scaler.transform(df_copy[cols])
                modality_scalers[modality] = scaler
            else:
                scaler = modality_scalers.get(modality)
                if scaler:
                    df_copy[cols] = scaler.transform(df_copy[cols])

    if return_scalers:
        return df_copy, modality_scalers
    return df_copy

def infer():
    raw_df = load_raw_dataframe()
    T = np.array(raw_df[TIME_COL])
    E = np.array(raw_df[EVENT_COL])
    E[T > CENSORING] = 0
    T[T > CENSORING] = CENSORING
    feature_df = raw_df.drop(columns=[col for col in EXCLUDE_COLS if col in raw_df.columns])

    preds = []

    if USE_ENSEMBLE:
        print("Running ensemble inference across all folds...")

        for fold in ENSEMBLE_FOLDS:
            model_path = os.path.join(RESULT_DIR, f"model_TASK{TASK}_RUN{RUN}_FOLD{fold}.pkl")
            scaler_path = os.path.join(RESULT_DIR, f"scalers_TASK{TASK}_RUN{RUN}_FOLD{fold}.pkl")

            model = joblib.load(model_path)
            scalers = joblib.load(scaler_path)

            X_scaled_df = split_and_scale_features(feature_df, fit=False, scalers=scalers)
            X_scaled = X_scaled_df.values
            Z = model.decision_function(X_scaled)
            preds.append(Z)

        Z_final = np.mean(preds, axis=0)

    else:
        print("Running single-model inference from fold 0...")
        model_path = os.path.join(RESULT_DIR, f"model_TASK{TASK}_RUN{RUN}_FOLD0.pkl")
        scaler_path = os.path.join(RESULT_DIR, f"scalers_TASK{TASK}_RUN{RUN}_FOLD0.pkl")

        model = joblib.load(model_path)
        scalers = joblib.load(scaler_path)

        X_scaled_df = split_and_scale_features(feature_df, fit=False, scalers=scalers)
        X_scaled = X_scaled_df.values
        Z_final = model.decision_function(X_scaled)

    cindex_val, _, _, _, _ = concordance_index_censored(E.astype(bool), T, -Z_final)
    print(f"\nInference C-Index: {cindex_val:.4f}")

if __name__ == "__main__":
    infer()
