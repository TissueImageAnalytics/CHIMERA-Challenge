import pandas as pd
import numpy as np
import os
import glob
import json
import random
import optuna

from sklearn.preprocessing import StandardScaler
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index as cindex
from scipy.stats import combine_pvalues
from tqdm import tqdm
from sksurv.metrics import concordance_index_censored

from TransductiveSR import TransductiveSR as TSRR

# === Configuration ===
TASK = 1  # or 3
USE_CLINICAL = False
USE_MRI = False
USE_WSI = True
CENSORING = 120  # months (10 years)
TUNING_TRIALS = 1000

CLINICAL_FEATURES = [
    "age_at_prostatectomy",
    "primary_gleason",
    "secondary_gleason",
    "ISUP",
    "pre_operative_PSA",
    "positive_surgical_margins"
]

# === Paths ===
if TASK == 1:
    WSI_FEATURE_PATH = "/home/u1970167/chimera/task1/pathology/features/titan/Task1_TITAN_1024_embeddings.csv"
    CLINICAL_PATH = "/home/u1970167/chimera/task1/clinical_data.csv"
    TIME_COL = 'time_to_follow-up/BCR'
    EVENT_COL = 'BCR'
    SLIDE_ID_COL = 'Slide_ID'
    MRI_FEATURE_DIR = "/home/u1970167/chimera/task1/radiology/features/"
else:
    WSI_FEATURE_PATH = "Features/Task3_TITAN_embeddings.csv"
    CLINICAL_PATH = "Features/task3_clinical.csv"
    TIME_COL = 'Time_to_prog_or_FUend'
    EVENT_COL = 'progression'
    SLIDE_ID_COL = 'slide_id'
    MRI_FEATURE_DIR = "Features/task3_mri_features/"

FOLDS_DIR = "/home/u1970167/chimera/task1/experiments/folds/"
RESULT_DIR = "/home/u1970167/chimera/task1/experiments/TSR_results/"
EXCLUDE_COLS = ["Case_ID", TIME_COL, EVENT_COL, SLIDE_ID_COL]  # Make sure these columns are not used as features

M_FEATURE_DIM = 2048
APPLY_ROI = True
VERBOSE = True

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

# === Data Loader ===
def load_chimera_dataset():
    clinical_data = pd.read_csv(CLINICAL_PATH)
    clinical_data["Case_ID"] = clinical_data["Case_ID"].astype(str)

    cols_to_use = ["Case_ID", TIME_COL, EVENT_COL]
    if USE_CLINICAL:
        cols_to_use += CLINICAL_FEATURES

    clinical_data = clinical_data[cols_to_use]
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
def get_split_data(case_ids, full_df):
    split_df = full_df[full_df["Case_ID"].isin(case_ids)].copy()
    T = np.array(split_df[TIME_COL])
    E = np.array(split_df[EVENT_COL])

    # Censoring logic
    E[T > CENSORING] = 0
    T[T > CENSORING] = CENSORING

    # Identify columns to exclude
    exclude_set = set(EXCLUDE_COLS).intersection(split_df.columns)
    X = split_df.drop(columns=exclude_set).values

    return X, T, E

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

        X_train, T_train, E_train = get_split_data(train_ids, dataset)
        X_val, T_val, E_val = get_split_data(test_ids, dataset)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        tsr_model = TSRR(lambda_w=lambda_w, lambda_u=lambda_u, p=p, Tmax=2000,
                         lr=LR, dropout=dropout, latent_dim=latent_dim)
        tsr_model.fit(X_train, T_train, E_train, X_val, plot_loss=False)
        Z_val = tsr_model.decision_function(X_val)
        cindex_val, _, _, _, _ = concordance_index_censored(E_val.astype(bool), T_val, -Z_val)
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
for run_numb in range(3):
    Bootstrap_cindex = []
    Bootstrap_p_Values = []
    threshold = 0
    p = 2

    for fold in tqdm(sorted(fold_df["fold"].unique())):
        test_ids = fold_df[fold_df["fold"] == fold]["Case_ID"].tolist()
        train_ids = fold_df[fold_df["fold"] != fold]["Case_ID"].tolist()

        X_train, T_train, E_train = get_split_data(train_ids, dataset)
        X_test, T_test, E_test = get_split_data(test_ids, dataset)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        tsr_model = TSRR(lambda_w=best_params['lambda_w'],
                         lambda_u=best_params['lambda_u'],
                         p=p, Tmax=2000,
                         lr=best_params['lr'],
                         dropout=best_params['dropout'],
                         latent_dim=best_params['latent_dim'])

        tsr_model.fit(X_train, T_train, E_train, X_test, plot_loss=False)
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

    mean_score = np.mean(Bootstrap_cindex)
    std_score = np.std(Bootstrap_cindex)
    _, combined_p = combine_pvalues(Bootstrap_p_Values, method='fisher')

    print(f"\nFinal Evaluation Run {run_numb+1}")
    print(f"Mean C-Index: {mean_score:.3f}")
    print(f"Std C-Index: {std_score:.3f}")
    print(f"Combined P-Value (Fisher): {combined_p:.3e}")

    runs_cindex.append(mean_score)
    runs_std.append(std_score)

    results_path = f"{RESULT_DIR}/Task_{TASK}_Clinical_{USE_CLINICAL}_MRI_{USE_MRI}_WSI_{USE_WSI}_TUNING_{TUNING_TRIALS}_RUN_{run_numb}.csv"

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

    print(f"\n Saved experiment csv to {results_path}")

print(f"Runs Modalities: Clinical({USE_CLINICAL}), WSI({USE_WSI}), MRI({USE_MRI})")
print(f"Runs: C-Indices: {runs_cindex}, Mean C-Index: {np.mean(runs_cindex):.3f}")
print(f"Runs: Stds for C-Indices: {runs_std}")