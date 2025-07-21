"""
Note we have dropped teriary gleason as a clinical feature only present in cases with pathological findings. Also dropped:
BCR_PSA as only present when BCR is 1.
Also, only using basic radiomic features for now (n=4), rather than all of them.
Radiomic features extracted without any normalisation. Need to ammend to do z-score normalisation within the mask etc.
"""

import os
import json
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Paths
radiomics_csv_path = "/media/u1973415/data/u1973415/Chimera/output/task_1/radiomics_features_all_cases.csv"
clinical_data_dir = "/media/u1973415/data/u1973415/Chimera/data/task_1/clinical_data/"
output_csv_path = "/media/u1973415/data/u1973415/Chimera/output/task_1/survival_model_results2.csv"

# Load radiomics features
radiomics_df = pd.read_csv(radiomics_csv_path)
radiomics_df = radiomics_df[['case_id', 'total_volume_mm3', 'avg_intensity_adc', 'avg_intensity_hbv','avg_intensity_t2w']]

# Extract case IDs from radiomics data (assuming first column is case ID or filename)
if 'case_id' in radiomics_df.columns:
    case_ids = radiomics_df['case_id'].astype(str)
else:
    case_ids = radiomics_df.index.astype(str)

# Load clinical data and merge
clinical_records = []
for case_id in case_ids:
    json_path = os.path.join(clinical_data_dir, f"{case_id}.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            clinical = json.load(f)
            clinical['case_id'] = case_id
            clinical_records.append(clinical)

clinical_df = pd.DataFrame(clinical_records)
clinical_df['case_id'] = clinical_df['case_id'].astype('int64')
clinical_df.drop(columns=["tertiary_gleason"], inplace=True) # Drop teriary gleason a sonly if present in pathological findings

# Prepare survival labels
clinical_df['event'] = clinical_df['BCR'].astype(float)
clinical_df['duration'] = clinical_df['time_to_follow-up/BCR'].astype(float)

# Create folds using clinical data only
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_clinical = clinical_df.drop(columns=['case_id', 'BCR', 'BCR_PSA', 'time_to_follow-up/BCR', 'event', 'duration']).select_dtypes(include=[np.number])
y_event = clinical_df['event']
folds = list(skf.split(X_clinical, y_event))

# Prepare radiomics features
X_radiomics = radiomics_df.drop(columns=['case_id']).select_dtypes(include=[np.number])

# Merge radiomics and clinical features
merged_df = pd.merge(radiomics_df, clinical_df, on='case_id')
y_duration = merged_df['duration']
y_event = merged_df['event']
X_combined = merged_df.drop(columns=['case_id', 'BCR', 'BCR_PSA', 'time_to_follow-up/BCR', 'event', 'duration']).select_dtypes(include=[np.number])

# Evaluate models
def evaluate_model(X, y_duration, y_event, folds, label, normalise=False):
    c_indices = []
    features_used = list(X.columns)
    for fold, (train_idx, test_idx) in enumerate(folds):
        
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()

        if normalise:
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

        train_df = X_train.copy()
        train_df['duration'] = y_duration.iloc[train_idx]
        train_df['event'] = y_event.iloc[train_idx]
        
        test_df = X_test.copy()
        test_df['duration'] = y_duration.iloc[test_idx]
        test_df['event'] = y_event.iloc[test_idx]
        
        cph = CoxPHFitter()
        cph.fit(train_df, duration_col='duration', event_col='event')
        
        partial_hazards = cph.predict_partial_hazard(test_df)
        # c_index = concordance_index(test_df['duration'], -partial_hazards, test_df['event'])
        # print(f"C-index from hazard: {c_index}")
        # c_index = concordance_index_censored(np.asarray(test_df['event'], dtype='bool'), test_df['duration'], partial_hazards)[0]
        # print(f"C-index from hazard, sk-surv: {c_index}")
        survival_functions = cph.predict_survival_function(test_df)
        # median_survival_times = survival_functions.apply(lambda s: s[s <= 0.5].index.min())
        expected_survival_times = survival_functions.apply(lambda s: s.sum() * (s.index[1] - s.index[0]))
        # c_index = concordance_index(test_df['duration'], expected_survival_times, test_df['event'])
        # print(f"C-index from survival time: {c_index}")
        c_index = concordance_index_censored(np.asarray(test_df['event'], dtype='bool'), test_df['duration'], -expected_survival_times)[0]
        # print(f"C-index from survival time, sk-surv: {c_index}")
        c_indices.append(c_index)
        print(f"{label} - Fold {fold + 1} C-index: {c_index:.4f}")
    print(f"\n{label} - Average C-index: {np.mean(c_indices):.4f}\n")
    avg_c_index = np.mean(c_indices)
    return c_indices, avg_c_index, features_used

# Run evaluations
results = []
c_indices_clinical, avg_clinical, features_clinical = evaluate_model(X_clinical, y_duration, y_event, folds, "Clinical Only", normalise=False)
c_indices_radiomics, avg_radiomics, features_radiomics = evaluate_model(X_radiomics, y_duration, y_event, folds, "Radiomics Only", normalise=False)
c_indices_combined, avg_combined, features_combined = evaluate_model(X_combined, y_duration, y_event, folds, "Combined Features", normalise=False)

# Save results to CSV
results_df = pd.DataFrame({
    'Fold': [f'Fold {i+1}' for i in range(5)] + ['Average'],
    'Clinical Only': c_indices_clinical + [avg_clinical],
    'Radiomics Only': c_indices_radiomics + [avg_radiomics],
    'Combined Features': c_indices_combined + [avg_combined]
})

results_df.to_csv(output_csv_path, index=False)

# Save features used
features_summary = {
    'Clinical Only': features_clinical,
    'Radiomics Only': features_radiomics,
    'Combined Features': features_combined
}
features_summary_path = output_csv_path.replace('.csv', '_features_used.json')
with open(features_summary_path, 'w') as f:
    json.dump(features_summary, f, indent=2)

print(f"Results saved to {output_csv_path}")
