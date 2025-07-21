import pandas as pd
import numpy as np
import random
import os
import json
from sklearn.preprocessing import StandardScaler
from lifelines.statistics import logrank_test
from scipy.stats import combine_pvalues
import optuna
from tqdm import tqdm
from lifelines.utils import concordance_index as cindex
from TransductiveSR import TransductiveSR as TSRR
from sksurv.metrics import concordance_index_censored

# ---------------- Config ----------------
task = 1  # or 3
modality = 'titan'
censoring = 120 # censoring after 10 years 
tuning_trials=1000


# ---------------- Load Data ----------------
if task == 1:
    time = 'time_to_follow-up/BCR'
    event = 'BCR'
    slide_embeddings = pd.read_csv("Features/Task1_TITAN_1024_embeddings.csv")
    slide_embeddings['Case_ID'] = slide_embeddings['slide_id'].astype(str).str.split('_').str[0]
    clinical_data = pd.read_csv("Features/task1_clinical.csv")
    clinical_data['Case_ID'] = clinical_data['Case_ID'].astype(str)
    clinical_data = clinical_data[['Case_ID', time, event]]
    dataset = slide_embeddings.merge(clinical_data, on='Case_ID', how='inner')
    dataset.set_index('slide_id', inplace=True)
else:
    time = 'Time_to_prog_or_FUend'
    event = 'progression'
    slide_embeddings = pd.read_csv("Features/Task3_TITAN_embeddings.csv")
    slide_embeddings["slide_id"]= slide_embeddings["slide_id"].str.replace('_HE$', '', regex=True)
    slide_embeddings.set_index('slide_id', inplace=True)
    slide_embeddings["Case_ID"] = slide_embeddings.index
    clinical_data = pd.read_csv("Features/task3_clinical.csv")
    clinical_data = clinical_data[['Case_ID', time, event]]
    clinical_data.set_index('Case_ID', inplace=True)
    dataset = slide_embeddings.merge(clinical_data, left_on="Case_ID", right_index=True)

dataset.drop(columns=["Case_ID"], inplace=True)

# ---------------- Load Precomputed Splits ----------------

split_csv_path = f"splits/task{task}_folds.csv"
fold_df = pd.read_csv(split_csv_path)
fold_df["Case_ID"] = fold_df["Case_ID"].astype(str)

# Add Case_ID back to dataset temporarily
dataset = dataset.copy()
dataset["Case_ID"] = dataset.index.astype(str).str.split("_").str[0]

# ---------------- Optuna Hyperparameter Tuning ----------------
def objective(trial):
    lambda_w = trial.suggest_loguniform('lambda_w', 1e-2, 1.0)
    lambda_u = trial.suggest_loguniform('lambda_u', 1e-2, 1.0)
    LR = trial.suggest_loguniform('lr', 1e-4, 1e-1)
    latent_dim = trial.suggest_int('latent_dim', 16, 128)
    dropout = trial.suggest_uniform('dropout', 0.0, 0.5)
    p = 2
    cv_scores = []

    unique_folds = sorted(fold_df["fold"].unique())
    for fold in unique_folds:
        test_cases = fold_df[fold_df["fold"] == fold]["Case_ID"].tolist()
        train_cases = fold_df[fold_df["fold"] != fold]["Case_ID"].tolist()

        val_data = dataset[dataset["Case_ID"].isin(test_cases)]
        train_data = dataset[dataset["Case_ID"].isin(train_cases)]


        T_train, E_train = np.array(train_data[time]), np.array(train_data[event])
        T_val, E_val = np.array(val_data[time]), np.array(val_data[event])
        X_train = train_data.drop([time, event], axis=1).values
        X_val = val_data.drop([time, event], axis=1).values

        E_train[T_train > censoring] = 0
        T_train[T_train > censoring] = censoring
        E_val[T_val > censoring] = 0
        T_val[T_val > censoring] = censoring

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        tsr_model = TSRR(lambda_w=lambda_w, lambda_u=lambda_u, p=p, Tmax=2000,
                         lr=LR, dropout=dropout, latent_dim=latent_dim)

        tsr_model.fit(X_train, T_train, E_train, X_val, plot_loss=False)
        Z_val = tsr_model.decision_function(X_val)
        event_indicator = E_val.astype(bool)
        cindex_val,_,_,_,_ = concordance_index_censored(event_indicator, T_val, -Z_val)

        cv_scores.append(cindex_val)

    return np.mean(cv_scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=tuning_trials)

best_params = study.best_params
print("Best hyperparameters found by Optuna:")
print(best_params)

# ---------------- Final Evaluation ----------------
Bootstrap_cindex = []
Bootstrap_p_Values = []
threshold = 0
p = 2

for fold in tqdm(sorted(fold_df["fold"].unique())):
    test_cases = fold_df[fold_df["fold"] == fold]["Case_ID"].tolist()
    train_cases = fold_df[fold_df["fold"] != fold]["Case_ID"].tolist()

    test_data = dataset[dataset["Case_ID"].isin(test_cases)]
    train_data = dataset[dataset["Case_ID"].isin(train_cases)]


    T_train, E_train = np.array(train_data[time]), np.array(train_data[event])
    T_test, E_test = np.array(test_data[time]), np.array(test_data[event])
    X_train = train_data.drop([time, event], axis=1).values
    X_test = test_data.drop([time, event], axis=1).values

    E_train[T_train > censoring] = 0
    T_train[T_train > censoring] = censoring
    E_test[T_test > censoring] = 0
    T_test[T_test > censoring] = censoring

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
    event_indicator = E_test.astype(bool)
    cindex_val,_,_,_,_ = concordance_index_censored(event_indicator, T_test, -Z_test)

    Bootstrap_cindex.append(cindex_val)

    Results_df = pd.DataFrame({'Prediction': Z_test, 'Time': T_test, 'Event': E_test})
    low_group = Results_df[Results_df['Prediction'] <= threshold]
    high_group = Results_df[Results_df['Prediction'] > threshold]

    result = logrank_test(low_group['Time'], high_group['Time'],
                          event_observed_A=low_group['Event'],
                          event_observed_B=high_group['Event'])
    Bootstrap_p_Values.append(result.p_value)

# ---------------- Save Final Results ----------------
mean_score = np.mean(Bootstrap_cindex)
std_score = np.std(Bootstrap_cindex)
_, combined_p = combine_pvalues(Bootstrap_p_Values, method='fisher')

print("\nFinal Evaluation:")
print(f"Mean C-Index: {mean_score:.3f}")
print(f"Std C-Index: {std_score:.3f}")
print(f"Combined P-Value (Fisher): {combined_p:.3e}")

results_path = f"{task}_{modality}_experiment_results.csv"

if os.path.exists(results_path):
    results_df = pd.read_csv(results_path, index_col=0)
    exp_index = results_df.index.max() + 1
else:
    results_df = pd.DataFrame(columns=[
        "lambda_w", "lambda_u", "lr", "latent_dim", "dropout",
        "mean_cindex", "std_cindex", "combined_p", "cindices", "p_values"
    ])
    exp_index = 1

results_df.loc[exp_index] = {
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
results_df.to_csv(results_path)

print(f"\n Saved experiment {exp_index} to {results_path}")
