import os
import pickle
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
#from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
from utils.data_utils import load_clinical, load_folds, encode_clinical_features
from config import *
from sksurv.metrics import concordance_index_censored
import json

os.makedirs(COX_BASELINE_DIR, exist_ok=True)

# Load clinical data
clinical_df = load_clinical()

# Columns to use
cols_to_use = ['Case_ID'] + CLINICAL_FEATURES + ['event', 'duration']
clinical_df = clinical_df[cols_to_use]

# Load folds from CSV using your function
folds_case_ids = load_folds()

# Convert folds from Case_IDs to train/test indices
fold_indices = []
for val_case_ids in folds_case_ids:
    val_idx = clinical_df[clinical_df['Case_ID'].isin(val_case_ids)].index.tolist()
    train_idx = clinical_df[~clinical_df['Case_ID'].isin(val_case_ids)].index.tolist()
    fold_indices.append((train_idx, val_idx))

def train_cv_cox_model(df, feature_cols, fold_indices):
    """
    Train CoxPH models on each fold and save models + scalers.
    Evaluate on validation sets and print fold-wise & average C-index.
    """
    c_indices = []

    for fold, (train_idx, val_idx) in enumerate(fold_indices):
        print(f"\n--- Training Fold {fold + 1} ---")
        
        X_train = df.loc[train_idx, feature_cols].copy()
        X_val = df.loc[val_idx, feature_cols].copy()

        if SCALE_DATA:
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=train_idx)
            X_val = pd.DataFrame(scaler.transform(X_val), columns=feature_cols, index=val_idx)
        else:
            scaler = None

        # Combine with target columns
        train_df = X_train.copy()
        train_df['duration'] = df.loc[train_idx, 'duration']
        train_df['event'] = df.loc[train_idx, 'event']

        val_df = X_val.copy()
        val_df['duration'] = df.loc[val_idx, 'duration']
        val_df['event'] = df.loc[val_idx, 'event']

        # === Defensive Checks === #
        if train_df.isnull().values.any():
            print(" NaNs detected in training data. Dropping rows with missing values.")
            train_df = train_df.dropna()

        # Drop constant columns (zero variance)
        nunique = train_df[feature_cols].nunique()
        constant_cols = nunique[nunique <= 1].index.tolist()
        if constant_cols:
            print(f" Dropping constant features: {constant_cols}")
            train_df.drop(columns=constant_cols, inplace=True)
            val_df.drop(columns=constant_cols, inplace=True)
            feature_cols = [col for col in feature_cols if col not in constant_cols]

        # Fit Cox model
        #cph = CoxPHFitter()
        cph = CoxPHFitter(penalizer=PENALIZER, l1_ratio=L1_RATIO)
        
        cph.fit(train_df, duration_col='duration', event_col='event')
        
        # Save model and scaler
        model_path = os.path.join(COX_BASELINE_DIR, f"cox_model_fold{fold+1}.pkl")
        scaler_path = os.path.join(COX_BASELINE_DIR, f"scaler_fold{fold+1}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(cph, f)
        if scaler:
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)

        # Evaluate on validation
        partial_hazards = cph.predict_partial_hazard(val_df)
        
        durations = val_df['duration'].values
        events = val_df['event'].values

        c_index = concordance_index_censored(events.astype(bool), durations, partial_hazards)
        
        c_indices.append(c_index[0])

        if VERBOSE:
            print(f" Fold {fold + 1} C-index: {c_index[0]:.3f}")

    # Summarize results
    valid_c_indices = [x for x in c_indices if not pd.isna(x)]
    avg_cindex = np.mean(valid_c_indices)
    std_cindex = np.std(valid_c_indices)

    print(f"\n Average C-index (excluding failed folds): {avg_cindex:.3f} ± {std_cindex:.3f}")

    results_df = pd.DataFrame({
        'Fold': [f'Fold {i+1}' for i in range(len(c_indices))],
        'C-index': c_indices
    })
    results_df.loc[len(results_df)] = ['Average', avg_cindex]
    results_df.to_csv(os.path.join(COX_BASELINE_DIR, "cox_clinical_results.csv"), index=False)
    print(" Results saved to:", COX_BASELINE_DIR)

    return c_indices, avg_cindex

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

def inference_ensemble(df, fold_count=NUM_FOLDS):
    """
    Perform inference using ensemble of saved Cox models.
    Aligns input features per fold with what was used during training.
    Saves predictions against Case_IDs in a CSV file.
    """
    partial_hazards_list = []

    for fold in range(fold_count):
        model_path = os.path.join(COX_BASELINE_DIR, f"cox_model_fold{fold+1}.pkl")
        scaler_path = os.path.join(COX_BASELINE_DIR, f"scaler_fold{fold+1}.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f" Model for fold {fold+1} not found.")
        
        with open(model_path, 'rb') as f:
            cph = pickle.load(f)
        fold_features = cph.params_.index.tolist()  # Features used during model training

        if INFER_SINGLE:
            test_case = os.listdir(CLINICAL_JSON_DIR)[0]
            input_chimera_clinical_data_of_prostate_cancer_patients = f"{CLINICAL_JSON_DIR}/{test_case}/{test_case}_CD.json"
            with open(input_chimera_clinical_data_of_prostate_cancer_patients, "r") as f:
                jsdata = json.load(f)

            X_array = extract_clinical_vector(jsdata)
            X = pd.DataFrame(X_array, columns=CLINICAL_FEATURES)
        else:
            X = df[fold_features].copy()

        if SCALE_DATA and os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

            # Match the columns to the scaler’s expected input
            if hasattr(scaler, 'feature_names_in_'):
                expected_features = scaler.feature_names_in_.tolist()
            else:
                expected_features = fold_features  # fallback

            # Ensure all expected columns exist
            missing_cols = set(expected_features) - set(X.columns)
            if missing_cols:
                print(f" [Fold {fold+1}] Adding missing columns: {missing_cols}")
                for col in missing_cols:
                    X[col] = 0  # Fill with zero if it was dropped during training

            # Reorder columns to match scaler expectation
            X = X[expected_features]

            X_scaled = pd.DataFrame(
                scaler.transform(X),
                columns=expected_features,
                #index=df.index
            )
        else:
            X_scaled = X

        pred_partial_hazard = cph.predict_partial_hazard(X_scaled)
        partial_hazards_list.append(pred_partial_hazard)

    # Ensemble: average of all predicted partial hazards
    ensemble_partial_hazard = pd.concat(partial_hazards_list, axis=1).mean(axis=1)

    ## to test with a single json clinical file
    if INFER_SINGLE:
        print(f"Score for case {test_case}:  {ensemble_partial_hazard}")
        exit()

    os.makedirs(COX_BASELINE_DIR, exist_ok=True)

    # Save Case_ID with predictions
    output_df = pd.DataFrame({
        'Case_ID': df['Case_ID'].values,
        'Predicted_Risk': ensemble_partial_hazard.values
    })

    # Higher risk means worse prognosis → sort descending
    output_df = output_df.sort_values(by="Predicted_Risk", ascending=False)

    save_path = os.path.join(COX_BASELINE_DIR, "ensemble_predictions.csv")
    output_df.to_csv(save_path, index=False)

    print(f"Inference predictions saved to: {save_path}")

    # Compute C-index (requires survival data)
    if 'duration' in df.columns and 'event' in df.columns:
        durations = df['duration'].values
        events = df['event'].values
        c_index = concordance_index_censored(events.astype(bool), durations, ensemble_partial_hazard)
        print(f"\n Ensemble Inference C-index: {c_index[0]:.3f}")
        return c_index[0]
    else:
        print("\n No duration/event columns found. Skipping C-index calculation.")
        return None

def main(mode="train"):
    """
    Main entry point to either train or infer.
    mode: 'train' or 'inference'
    """
    if mode == "train":
        print("Starting 5-fold cross-validation training...")
        train_cv_cox_model(clinical_df, CLINICAL_FEATURES, fold_indices)

    elif mode == "inference":
        print("Starting inference using ensemble of 5 folds on entire dataset...")
        inference_ensemble(clinical_df, fold_count=len(fold_indices))

    else:
        raise ValueError("Invalid mode! Use 'train' or 'inference'.")

if __name__ == "__main__":
    mode = 'inference' # 'train' | 'inference'
    
    main(mode)