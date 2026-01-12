import os
import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
# from data_utils import load_clinical, load_folds, encode_clinical_features
from config_cox import *
import json
from pathlib import Path

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")
SURVIVAL_WEIGHTS_PATH = Path("/opt/app/resources/weights_scalers_folder")
SCALER_WEIGHTS_PATH = Path("/opt/app/resources/weights_scalers_folder")

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

    return df_encoded.values.astype(np.float32)


def inference_ensemble():
    """
    Perform inference using ensemble of saved Cox models.
    Aligns input features per fold with what was used during training.
    Saves predictions against Case_IDs in a CSV file.
    """
    partial_hazards_list = []

    for fold in range(NUM_FOLDS):
        model_path = os.path.join(SURVIVAL_WEIGHTS_PATH, f"cox_model_fold{fold+1}.pkl")
        scaler_path = os.path.join(SURVIVAL_WEIGHTS_PATH, f"scaler_fold{fold+1}.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f" Model for fold {fold+1} not found.")
        
        with open(model_path, 'rb') as f:
            cph = pickle.load(f)
        fold_features = cph.params_.index.tolist()  # Features used during model training

        if INFER_SINGLE:
            input_chimera_clinical_data_of_prostate_cancer_patients = INPUT_PATH / "chimera-clinical-data-of-bladder-cancer-recurrence-patients.json"
            with open(input_chimera_clinical_data_of_prostate_cancer_patients, "r") as f:
                jsdata = json.load(f)

            X_array = extract_clinical_vector(jsdata)
            X = pd.DataFrame(X_array, columns=CLINICAL_FEATURES)


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
    
    result = ensemble_partial_hazard.values[0]

    ## to test with a single json clinical file
    if INFER_SINGLE:
        print(f"Score for case: {result}")
        return result
    





def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))



def generic_handler():      

    output_likelihood_of_bladder_cancer_recurrence = inference_ensemble()

    print(f"Predicted score: {output_likelihood_of_bladder_cancer_recurrence}")

    write_json_file(
        location=OUTPUT_PATH / "likelihood-of-bladder-cancer-recurrence.json",
        content=output_likelihood_of_bladder_cancer_recurrence,
    )

    return 0

if __name__ == "__main__":
    generic_handler()