import pandas as pd
import numpy as np
import os
import glob
import optuna
import joblib

from sklearn.preprocessing import StandardScaler
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index as cindex
from scipy.stats import combine_pvalues
from tqdm import tqdm
from sksurv.metrics import concordance_index_censored
import re
from TransductiveSR import TransductiveSR as TSRR
from config import *

os.makedirs(RESULT_DIR, exist_ok=True)

# === Load Precomputed Folds ===
split_csv_path = f"{FOLDS_DIR}/task{TASK}_folds.csv"
fold_df = pd.read_csv(split_csv_path)
fold_df["Case_ID"] = fold_df["Case_ID"].astype(str)
print(f"Loaded folds from: {split_csv_path}, with {len(fold_df)} rows")

# === MRI Feature Loader ===
def load_mri_features(case_ids):
    feats = []
    missing = []

    for case_id in case_ids:
        pattern = os.path.join(MRI_FEATURE_DIR, f"ROI_{APPLY_ROI}", f"{case_id}_*.npy")
        matched_files = sorted(glob.glob(pattern))

        if matched_files:
            try:
                feat = np.load(matched_files[0])
                feats.append(feat)
            except Exception as e:
                print(f"Error loading {matched_files[0]}: {e}")
                feats.append(np.zeros((M_FEATURE_DIM,), dtype=np.float32))
                missing.append(case_id)
        else:
            feats.append(np.zeros((M_FEATURE_DIM,), dtype=np.float32))
            missing.append(case_id)

    if missing and VERBOSE:
        print(f"Missing MRI features for {len(missing)} case(s): {missing}")

    return np.stack(feats)

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

# === Data Loader ===
def load_chimera_dataset():
    clinical_data = pd.read_csv(CLINICAL_PATH)
    clinical_data["Case_ID"] = clinical_data["Case_ID"].astype(str)

    cols_to_use = ["Case_ID", TIME_COL, EVENT_COL]
    if USE_CLINICAL:
        cols_to_use += CLINICAL_FEATURES
    
    print('cols: ', clinical_data.columns)

    clinical_data = clinical_data[cols_to_use]

    # Apply mixed column conversion
    for col in MIXED_COLS:
        if col in clinical_data.columns:
            clinical_data[col] = convert_mixed_column_to_numeric(clinical_data[col])

    # Apply numeric conversion to remaining clinical features
    clinical_feature_cols = [col for col in CLINICAL_FEATURES if col in clinical_data.columns]
    clinical_data[clinical_feature_cols] = clinical_data[clinical_feature_cols].apply(pd.to_numeric, errors='coerce')
    clinical_data = clinical_data.dropna(subset=clinical_feature_cols)

    print(f"Loaded clinical data with shape {clinical_data.shape}")

    if USE_WSI:
        slide_embeddings = pd.read_csv(WSI_FEATURE_PATH)
        slide_embeddings[SLIDE_ID_COL] = slide_embeddings[SLIDE_ID_COL].astype(str)
        if TASK == 3:
            slide_embeddings[SLIDE_ID_COL] = slide_embeddings[SLIDE_ID_COL].str.replace('_HE$', '', regex=True)

        slide_embeddings["Case_ID"] = slide_embeddings[SLIDE_ID_COL].str.split('_').str[0]
        embedding_cols = [col for col in slide_embeddings.columns if col.startswith("dim")]
        case_wsi_embeddings = slide_embeddings.groupby("Case_ID")[embedding_cols].mean().reset_index()
        print(f"Aggregated WSI features with shape {case_wsi_embeddings.shape}")
    else:
        case_wsi_embeddings = None

    # --- Merge modalities ---
    if USE_CLINICAL and USE_WSI:
        dataset = pd.merge(clinical_data, case_wsi_embeddings, on="Case_ID", how="inner")
    elif USE_CLINICAL:
        dataset = clinical_data
    elif USE_WSI:
        dataset = case_wsi_embeddings
        dataset = pd.merge(dataset, clinical_data[["Case_ID", TIME_COL, EVENT_COL]], on="Case_ID", how="inner")
    elif USE_MRI:
        dataset = clinical_data[["Case_ID", TIME_COL, EVENT_COL]].copy()
    else:
        raise ValueError("You must enable at least one modality (clinical, WSI, or MRI).")

    if USE_MRI:
        mri_case_ids = dataset["Case_ID"].tolist()
        mri_array = load_mri_features(mri_case_ids)
        mri_df = pd.DataFrame(mri_array, columns=[f"mri_feat_{i}" for i in range(mri_array.shape[1])])
        dataset = pd.concat([dataset.reset_index(drop=True), mri_df], axis=1)
        print(f"Appended MRI features, final dataset shape: {dataset.shape}")

    print(f"\nFinal dataset preview:\n{dataset.head()}")
    print(f"Final dataset shape: {dataset.shape}")
    return dataset

# === Prepare Split Data ===
def prepare_data_split(case_ids, full_df, fit=True, scalers=None, return_scalers=False):
    split_df = full_df[full_df["Case_ID"].isin(case_ids)].copy()
    T = np.array(split_df[TIME_COL])
    E = np.array(split_df[EVENT_COL])

    # Censoring logic
    E[T > CENSORING] = 0
    T[T > CENSORING] = CENSORING

    # Drop non-feature columns
    exclude_set = set(EXCLUDE_COLS).intersection(split_df.columns)
    X_df = split_df.drop(columns=exclude_set).copy()

    # Identify modalities
    clinical_cols = [col for col in X_df.columns if col in CLINICAL_FEATURES]
    wsi_cols = [col for col in X_df.columns if col.startswith('dim')]
    mri_cols = [col for col in X_df.columns if col.startswith('mri_feat_')]

    if scalers is None:
        scalers = {}

    for modality, cols in zip(['clinical', 'wsi', 'mri'], [clinical_cols, wsi_cols, mri_cols]):
        if not cols:
            continue

        if fit:
            scaler = StandardScaler()
            X_df[cols] = scaler.fit_transform(X_df[cols])
            scalers[modality] = scaler
        else:
            X_df[cols] = scalers[modality].transform(X_df[cols])

    if return_scalers:
        return X_df.values, T, E, scalers
    else:
        return X_df.values, T, E

# === Optuna Objective ===
def objective(trial):
    lambda_w = trial.suggest_loguniform('lambda_w', 1e-2, 1.0)
    lambda_u = trial.suggest_loguniform('lambda_u', 1e-2, 1.0)
    LR = trial.suggest_loguniform('lr', 1e-4, 1e-1)
    latent_dim = trial.suggest_int('latent_dim', 16, 128)
    dropout = trial.suggest_uniform('dropout', 0.0, 0.5)
    p = 2
    cv_scores = []

    for fold in sorted(fold_df["fold"].unique()):
        test_ids = fold_df[fold_df["fold"] == fold]["Case_ID"].tolist()
        train_ids = fold_df[fold_df["fold"] != fold]["Case_ID"].tolist()

        # Unified data preparation with scalers
        X_train, T_train, E_train, scalers = prepare_data_split(
            train_ids, dataset, fit=True, return_scalers=True
        )
        X_val, T_val, E_val = prepare_data_split(
            test_ids, dataset, fit=False, scalers=scalers
        )

        tsr_model = TSRR(
            lambda_w=lambda_w,
            lambda_u=lambda_u,
            p=p,
            Tmax=2000,
            lr=LR,
            dropout=dropout,
            latent_dim=latent_dim,
            structure=TSR_STRUCTURE
        )

        tsr_model.fit(X_train, T_train, E_train, X_val, plot_loss=False)
        Z_val = tsr_model.decision_function(X_val)

        cindex_val, _, _, _, _ = concordance_index_censored(
            E_val.astype(bool), T_val, -Z_val
        )
        cv_scores.append(cindex_val)

    return np.mean(cv_scores)

# === Load dataset and run optimization ===
dataset = load_chimera_dataset()
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=TUNING_TRIALS)
best_params = study.best_params
print("\nBest hyperparameters:", best_params)

# === Final Evaluation ===
runs_cindex = []
runs_std = []

runs_cindex = []
runs_std = []

for run_numb in range(3):  #######@@@@@@@@ RUNS @@@@@###########
    Bootstrap_cindex = []
    Bootstrap_p_Values = []
    threshold = 0
    p = 2

    for fold in tqdm(sorted(fold_df["fold"].unique())):
        test_ids = fold_df[fold_df["fold"] == fold]["Case_ID"].tolist()
        train_ids = fold_df[fold_df["fold"] != fold]["Case_ID"].tolist()

        # Prepare and scale training data
        X_train, T_train, E_train, scalers = prepare_data_split(
            train_ids, dataset, fit=True, return_scalers=True
        )

        # Prepare test data using same scalers
        X_test, T_test, E_test = prepare_data_split(
            test_ids, dataset, fit=False, scalers=scalers
        )

        # === Save scalers per fold & run ===
        scaler_path = os.path.join(
            RESULT_DIR,
            f"scalers_TASK{TASK}_RUN{run_numb}_FOLD{fold}.pkl"
        )
        joblib.dump(scalers, scaler_path)

        # === Train TSRR model ===
        tsr_model = TSRR(
            lambda_w=best_params['lambda_w'],
            lambda_u=best_params['lambda_u'],
            p=p,
            Tmax=2000,
            lr=best_params['lr'],
            dropout=best_params['dropout'],
            latent_dim=best_params['latent_dim'],
            structure=TSR_STRUCTURE
        )

        tsr_model.fit(X_train, T_train, E_train, X_test, plot_loss=False)

        # === Save model per fold & run ===
        model_path = os.path.join(
            RESULT_DIR,
            f"model_TASK{TASK}_RUN{run_numb}_FOLD{fold}.pkl"
        )
        joblib.dump(tsr_model, model_path)

        # === Evaluation ===
        Z_test = tsr_model.decision_function(X_test)
        cindex_val, _, _, _, _ = concordance_index_censored(E_test.astype(bool), T_test, -Z_test)
        Bootstrap_cindex.append(cindex_val)

        results_df = pd.DataFrame({'Prediction': Z_test, 'Time': T_test, 'Event': E_test})
        low_group = results_df[results_df['Prediction'] <= threshold]
        high_group = results_df[results_df['Prediction'] > threshold]

        result = logrank_test(
            low_group['Time'], high_group['Time'],
            event_observed_A=low_group['Event'],
            event_observed_B=high_group['Event']
        )
        Bootstrap_p_Values.append(result.p_value)

    # === Log results for this run ===
    mean_score = round(np.mean(Bootstrap_cindex), 3)
    std_score = round(np.std(Bootstrap_cindex), 2)
    _, combined_p = combine_pvalues(Bootstrap_p_Values, method='fisher')

    print(f"\nFinal Evaluation Run {run_numb + 1}")
    print(f"Mean C-Index: {mean_score:.3f}")
    print(f"Std C-Index: {std_score:.2f}")
    print(f"Combined P-Value (Fisher): {combined_p:.3e}")

    runs_cindex.append(mean_score)
    runs_std.append(std_score)

    # Save all run metrics and params
    results_path = os.path.join(
        RESULT_DIR,
        f"PARMS_RESULTS_{TUNING_TRIALS}_RUN_{run_numb}.csv"
    )

    results_df = {
        "lambda_w": best_params["lambda_w"],
        "lambda_u": best_params["lambda_u"],
        "lr": best_params["lr"],
        "latent_dim": best_params["latent_dim"],
        "dropout": best_params["dropout"],
        "mean_cindex": mean_score,
        "std_cindex": std_score,
        "combined_p": combined_p,
        "cindices": str(Bootstrap_cindex),
        "p_values": str(Bootstrap_p_Values)
    }
    pd.DataFrame([results_df]).to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")

print(f"\nRuns Modalities: Clinical({USE_CLINICAL}), WSI({USE_WSI}), MRI({USE_MRI})")
print(f"Runs: Mean C-Index: {round(np.mean(runs_cindex), 3)}, C-Indices: {runs_cindex}")
print(f"Runs: Stds for C-Indices: {runs_std}")
