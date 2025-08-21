import os
import numpy as np
import pandas as pd
import torch
from config import *
from utils.data_utils import load_clinical, load_mri_features, load_wsi_features, load_wsi_features_mult, get_feature_dimensionalities, convert_mixed_column_to_numeric, safe_to_array
from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import StandardScaler
from models.model import TransductiveSR as TSRR
import joblib
import json

def get_or_fit_scaler(name, train_array, fit=True, run=0, fold=0):
    os.makedirs(GLOBAL_DIR, exist_ok=True)
    scaler_path = os.path.join(GLOBAL_DIR, f"{name}_scaler_RUN{run}_FOLD{fold}.pkl")
    
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

# def load_inference_data():
#     """Load and merge modalities in the same way as training."""
#     df = load_clinical()

#     if USE_WSI_FEATURES:
#         if AGGREG_CASE_WSI:
#             wsi_df = load_wsi_features_mult(case_ids=)
#         else:
#             wsi_df = load_wsi_features(case_ids=)
#         df = df.merge(wsi_df, on="Case_ID", how="inner")

#     if USE_MRI_FEATURES:
#         mri_df = load_mri_features(case_ids=)
#         df = df.merge(mri_df, on="Case_ID", how="inner")

#     # censoring
#     T = np.array(df[TIME_COLUMN])
#     E = np.array(df[EVENT_COLUMN])
#     E[T > CENSORING] = 0
#     T[T > CENSORING] = CENSORING

#     return df, T, E

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

def inference(test_case_ids):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load clinical data and get feature dims
    clinical_df = load_clinical()
    clinical_dim = get_feature_dimensionalities(clinical_df)

    # Sort test case IDs to preserve consistent output order
    test_case_ids = sorted(test_case_ids)

    # Get input arrays
    test_clinical = clinical_df[clinical_df['Case_ID'].isin(test_case_ids)].sort_values('Case_ID')
    
    test_clin_array = test_clinical.drop(columns=['Case_ID', 'duration', 'event']).values if USE_CLINICAL_FEATURES else None

    #### for single case from clincal json @@@@@@@@@@@@@@@@@@@@@@@@@@
    if INFER_SINGLE:
        test_case = os.listdir(CLINICAL_JSON_DIR)[0]
        input_chimera_clinical_data_of_prostate_cancer_patients = f"{CLINICAL_JSON_DIR}/{test_case}"
        with open(input_chimera_clinical_data_of_prostate_cancer_patients, "r") as f:
           jsdata = json.load(f)

        test_clin_array = extract_clinical_vector(jsdata)
        
        ## ToDo: change the MRI and WSI feature loading to challenge format
        ## Temp work around: change the test_case_ids as in the next line
        test_case_ids = [test_case.split('.json')[0]]

    test_mri_array = load_mri_features(test_case_ids) if USE_MRI_FEATURES else None

    if AGGREG_CASE_WSI:
        test_wsi_array = load_wsi_features_mult(test_case_ids, csv_path=WSI_FEATURES_CSV) if USE_WSI_FEATURES else None
    else:
        test_wsi_array = load_wsi_features(test_case_ids, csv_path=WSI_FEATURES_CSV) if USE_WSI_FEATURES else None
    
    ## ## testing multiple cases from json so that confirm the c-index works
    if not INFER_LOCAL:
        test_clin_array = load_clinical_vectors_from_jsons(test_case_ids) 

    best_run_path = os.path.join(GLOBAL_DIR, "best_run.json")
    if not os.path.exists(best_run_path):
        raise FileNotFoundError(f"Missing best_run.json. Please train models first.")
    with open(best_run_path, "r") as f:
        best_run_info = json.load(f)

    best_run = best_run_info["best_run"]
    fold_cindices = best_run_info["fold_cindices"]

    if USE_ENSEMBLE:
        preds = []

        for fold_idx in range(NUM_FOLDS): ## model_TASK1_RUN0_FOLD1
            model_path = os.path.join(GLOBAL_DIR, f"model_RUN{best_run}_FOLD{fold_idx}.pkl")

            if not os.path.exists(model_path):
                print(f"[Warning] Model missing for fold {fold_idx}: {model_path}")
                continue
        
            samples = test_clin_array.shape[0]

            # scale + safe conversion for each modality
            if USE_CLINICAL_FEATURES:
                test_clin_array, _ = maybe_scale("clinical", test_clin_array, test_clin_array,
                                                fit=False, run=best_run, fold=fold_idx)
                test_clin_array = safe_to_array(test_clin_array, samples)
            else:
                test_clin_array = None

            if USE_MRI_FEATURES:
                test_mri_array, _ = maybe_scale("mri", test_mri_array, test_mri_array,
                                                fit=False, run=best_run, fold=fold_idx)
                test_mri_array = safe_to_array(test_mri_array, samples)
            else:
                test_mri_array = None

            if USE_WSI_FEATURES:
                test_wsi_array, _ = maybe_scale("wsi", test_wsi_array, test_wsi_array,
                                                fit=False, run=best_run, fold=fold_idx)
                test_wsi_array = safe_to_array(test_wsi_array, samples)
            else:
                test_wsi_array = None

            # collect only active modalities
            modalities_test = []
            if USE_CLINICAL_FEATURES:
                modalities_test.append(test_clin_array)
            if USE_MRI_FEATURES:
                modalities_test.append(test_mri_array)
            if USE_WSI_FEATURES:
                modalities_test.append(test_wsi_array)

            # final design matrix
            X_val = np.concatenate(modalities_test, axis=1)

            # inference
            model = joblib.load(model_path)
            print('X_val: ', X_val)
            Z = model.decision_function(X_val)
            preds.append(Z)

        avg_preds = np.mean(preds, axis=0)
    
    ## to test with a single json clinical file
    if INFER_SINGLE:
        print(f"Score for case {test_case}:  {avg_preds[0]}")
        exit()

    return dict(zip(test_case_ids, avg_preds))

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

    print(f"Inference predictions saved to: {save_path}")

    # Get true durations and events
    clinical_df_sorted = clinical_df[clinical_df['Case_ID'].isin(all_case_ids)].sort_values('Case_ID')
    durations = clinical_df_sorted['duration'].values
    events = clinical_df_sorted['event'].values
    preds = np.array([predictions[cid] for cid in clinical_df_sorted['Case_ID'].tolist()])

    # Compute C-index
    ## sksurv concordance function expects a risk score but as the challenge expect the algorithms to return time-to-event therefore, their code negates the time-to-event so it becomes a risk score
    ## But if the algorithm is returning a risk score then it should be negated before hand so that the negation of their code is cancelled out and the risk score remains the risk score.
 
    c_index = concordance_index_censored( 
            events.astype(bool),
            durations,
            -preds
    )
    
    if USE_ENSEMBLE:
        print(f"\nEnsemble C-index on full training set: {c_index[0]:.4f}")
    else:
        print(f"\nC-index on full training set using best of 5fold model: {c_index[0]:.4f}")

        