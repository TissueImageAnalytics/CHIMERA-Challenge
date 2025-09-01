#!/usr/bin/env python3
import os, json, argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from pprint import pprint
from glob import glob
import h5py
from TransductiveSR import TransductiveSR

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")
WEIGHTS_PATH = Path("/opt/app/resources/weights_scalers_folder")

DROP_CLIN_COLS = ['earlier_therapy', 'BCR_PSA', 'tertiary_gleason']

def read_json(p): 
    with open(p, "r") as f: 
        return json.load(f)

def find_first(*paths):
    for p in paths:
        if p and os.path.exists(p): 
            return p
    return None

# def normalize_clinical_ids(df):
#     out = df.copy()
#     if 'slide_id' in out.columns and 'Case_ID' not in out.columns:
#         out = out.rename(columns={'slide_id': 'Case_ID'})
#     out = out.drop(columns=['slide_id'], errors='ignore')
#     out['Case_ID'] = out['Case_ID'].astype(str)
#     return out

# def add_case_id_from_slide_index(wsi_df, task):
#     out = wsi_df.copy()
#     if 'slide_id' in out.columns:
#         out = out.set_index('slide_id')
#     if task == 3:
#         out.index = out.index.str.replace('_HE$', '', regex=True)
#     out['Case_ID'] = out.index.astype(str).str.split('_').str[0]
#     return out


def load_clinical_feats():
    with open(INPUT_PATH / "chimera-clinical-data-of-prostate-cancer-patients.json", "r") as f:
        clinical_data = json.load(f)


    # Wrap into a DataFrame for consistent preprocessing
    df = pd.DataFrame([clinical_data])
    pprint("Clinical data found:")
    pprint(df)
    return df


def load_WSI_feats():

    wsi_dir = INPUT_PATH / "images/prostatectomy-wsi"

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
    with h5py.File(wsi_feature_path, 'r') as f:
        features = f['features'][()]

    if features.ndim == 1:
        features = features.reshape(1, -1)

    print("WSI features extracted.")

    return features


def prepare_blocks(task, time_col, event_col,
                   drop_cols, wsi_feature_cols,
                   clin_numeric_cols, clin_categorical_cols, clin_dummy_cols, clin_numeric_medians):
    # WSI
    # wsi = pd.read_csv(wsi_csv)
    Xw = load_WSI_feats()


    wsi = pd.DataFrame(Xw, columns=wsi_feature_cols)
    wsi.loc[0, "Case_ID"] = 0


    # wsi = add_case_id_from_slide_index(wsi, task)
    missing_wsi = [c for c in wsi_feature_cols if c not in wsi.columns]
    if missing_wsi:
        raise ValueError(f"WSI CSV missing expected columns: {missing_wsi[:8]} ...")
    Xw = wsi[wsi_feature_cols].copy()

    

    # Clinical
    # clin = pd.read_csv(clinical_csv)
    clin = load_clinical_feats()
    clin.loc[0, "Case_ID"] = 0
    # clin = normalize_clinical_ids(clin)
    clin = clin.drop(columns=[c for c in set(drop_cols + [time_col, event_col]) if c in clin.columns], errors='ignore')

    merged = wsi[['Case_ID']].merge(clin, on='Case_ID', how='left')
    merged.index = wsi.index

    # Numeric
    for c in clin_numeric_cols:
        if c not in merged.columns:
            merged[c] = np.nan
    Xc_num_df = merged[clin_numeric_cols].copy()
    for c in clin_numeric_cols:
        Xc_num_df[c] = Xc_num_df[c].fillna(clin_numeric_medians.get(c, 0.0))
    Xc_num = Xc_num_df.to_numpy() if clin_numeric_cols else np.empty((len(merged), 0))

    # Categorical -> dummies aligned
    if clin_categorical_cols:
        for c in clin_categorical_cols:
            if c not in merged.columns:
                merged[c] = np.nan
        cat = pd.get_dummies(merged[clin_categorical_cols], drop_first=True, dummy_na=True, dtype=float)
        for col in clin_dummy_cols:
            if col not in cat.columns:
                cat[col] = 0
        cat = cat.reindex(columns=clin_dummy_cols, fill_value=0)
        Xc_cat = cat.to_numpy()
    else:
        Xc_cat = np.empty((len(merged), 0))

    return Xw, Xc_num, Xc_cat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--task", type=int, choices=[1,3], required=True)
    ap.add_argument("--wsi_csv", required=True)
    ap.add_argument("--clinical_csv", required=True)
    ap.add_argument("--time_col", default=None)
    ap.add_argument("--event_col", default=None)
    args = ap.parse_args()

    if args.time_col is None or args.event_col is None:
        time_col, event_col = ('time_to_follow-up/BCR', 'BCR') if args.task == 1 else ('Time_to_prog_or_FUend', 'progression')
    else:
        time_col, event_col = args.time_col, args.event_col

    # ---- columns_info (per-fold preferred) ----
    colinfo_path = find_first(
        os.path.join(WEIGHTS_PATH, f"columns_info_RUN{args.run}_FOLD{args.fold}.json"),
        os.path.join(WEIGHTS_PATH, "columns_info.json")
    )
    if not colinfo_path:
        raise FileNotFoundError(
            "Couldn't find columns info. Expected either "
            f"'columns_info_RUN{args.run}_FOLD{args.fold}.json' or 'columns_info.json' in {WEIGHTS_PATH}."
        )
    colinfo = read_json(colinfo_path)

    wsi_feature_cols      = colinfo["wsi_feature_cols"]
    clin_numeric_cols     = colinfo["clin_numeric_cols"]
    clin_categorical_cols = colinfo["clin_categorical_cols"]
    clin_dummy_cols       = colinfo["clin_dummy_columns"]
    clin_numeric_medians  = colinfo["clin_numeric_medians"]

    # ---- load model & scalers ----
    model_path = os.path.join(WEIGHTS_PATH, f"model_RUN{args.run}_FOLD{args.fold}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model file: {model_path}")

    wsi_scaler_path = find_first(
        os.path.join(WEIGHTS_PATH, f"wsi_scaler_fold{args.fold}.pkl"),
        os.path.join(WEIGHTS_PATH, f"wsi_scaler_RUN{args.run}_FOLD{args.fold}.pkl")
    )
    clin_scaler_path = find_first(
        os.path.join(WEIGHTS_PATH, f"clinical_scaler_fold{args.fold}.pkl"),
        os.path.join(WEIGHTS_PATH, f"clinical_scaler_RUN{args.run}_FOLD{args.fold}.pkl")
    )
    fused_scaler_path = find_first(
        os.path.join(WEIGHTS_PATH, f"fused_scaler_RUN{args.run}_FOLD{args.fold}.pkl"),
        None
    )

    model = joblib.load(model_path)
    wsi_scaler = joblib.load(wsi_scaler_path) if wsi_scaler_path else None
    clin_scaler = joblib.load(clin_scaler_path) if clin_scaler_path else None
    fused_scaler = joblib.load(fused_scaler_path) if fused_scaler_path else None

    # ---- build features ----
    Xw_raw, Xc_num_raw, Xc_cat = prepare_blocks(
        args.task, time_col, event_col,
        DROP_CLIN_COLS, wsi_feature_cols,
        clin_numeric_cols, clin_categorical_cols, clin_dummy_cols, clin_numeric_medians
    )

    Xw_raw = Xw_raw.astype(np.float32)
    Xw_raw = Xw_raw.values
    Xc_num_raw = Xc_num_raw.astype(np.float32)
    Xc_cat = Xc_cat.astype(np.float32)

    pprint("Xw_raw")
    pprint(Xw_raw)
    pprint("Xc_num_raw")
    pprint(Xc_num_raw)
    pprint("Xc_cat")
    pprint(Xc_cat)


    # scale continuous parts
    Xw_s = wsi_scaler.transform(Xw_raw) if (wsi_scaler and Xw_raw.shape[1] > 0) else Xw_raw
    Xcnum_s = clin_scaler.transform(Xc_num_raw) if (clin_scaler and Xc_num_raw.shape[1] > 0) else Xc_num_raw
    cont = np.hstack([Xw_s, Xcnum_s])
    cont_s = fused_scaler.transform(cont) if (fused_scaler and cont.shape[1] > 0) else cont

    # append one-hots (unscaled)
    X = np.hstack([cont_s, Xc_cat]) if Xc_cat.shape[1] > 0 else cont_s

    # pprint(X)

    # predict
    Z = model.decision_function(X)

    return float(Z[0]) + 1


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))



if __name__ == "__main__":
    output_time_to_biochemical_recurrence_for_prostate_cancer = main()
    print(f"Predicted time: {output_time_to_biochemical_recurrence_for_prostate_cancer}")

    write_json_file(
        location=OUTPUT_PATH
        / "time-to-biochemical-recurrence-for-prostate-cancer-months.json",
        content=output_time_to_biochemical_recurrence_for_prostate_cancer,
    )