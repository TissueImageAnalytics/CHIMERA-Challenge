#!/usr/bin/env python3
# Inference for TASK 1 CoxPH (ensemble risk scores)

import os, json, argparse
import numpy as np
import pandas as pd
import joblib

# ---------- utils ----------
def read_json(p): 
    with open(p, "r") as f: 
        return json.load(f)

def normalize_clinical_ids(df):
    out = df.copy()
    if "slide_id" in out.columns and "Case_ID" not in out.columns:
        out = out.rename(columns={"slide_id":"Case_ID"})
    out = out.drop(columns=["slide_id"], errors="ignore")
    out["Case_ID"] = out["Case_ID"].astype(str)
    return out

def add_case_id_from_slide_index(wsi_df):
    out = wsi_df.copy()
    if "slide_id" in out.columns:
        out = out.set_index("slide_id")
    out["Case_ID"] = out.index.astype(str).split("_")[0]  # won't work vectorized
    # proper vectorized:
    out["Case_ID"] = out.index.astype(str).str.split("_").str[0]
    return out

def clinical_transform_apply_task1(df_raw: pd.DataFrame, meta: dict):
    num_cols = meta.get("num_cols", [])
    medians  = meta.get("medians", {})
    cat_cols = meta.get("cat_cols", [])
    cat_dummy_cols = meta.get("cat_dummy_cols", [])

    if num_cols:
        df_num = df_raw[num_cols].copy()
        for c in num_cols:
            df_num[c] = df_num[c].fillna(medians.get(c, 0.0))
        X_num = df_num.to_numpy()
    else:
        X_num = np.empty((len(df_raw), 0))

    if cat_cols:
        df_cat = pd.get_dummies(df_raw[cat_cols], drop_first=True, dummy_na=True)
        df_cat = df_cat.reindex(columns=cat_dummy_cols, fill_value=0)
        X_cat = df_cat.to_numpy()
    else:
        X_cat = np.empty((len(df_raw), 0))

    return X_num, X_cat

def transform_with_scalers(Xw, Xcnum, Xccat, scalers):
    Xw_s    = scalers["scaler_wsi"].transform(Xw)      if (scalers["scaler_wsi"]  and Xw.shape[1]    > 0) else np.empty((len(Xw), 0))
    Xcnum_s = scalers["scaler_cnum"].transform(Xcnum)  if (scalers["scaler_cnum"] and Xcnum.shape[1] > 0) else np.empty((len(Xcnum), 0))
    cont    = np.hstack([Xw_s, Xcnum_s])
    cont_s  = scalers["scaler_fused"].transform(cont)  if (scalers["scaler_fused"] and cont.shape[1] > 0) else cont
    return np.hstack([cont_s, Xccat]) if Xccat.shape[1] > 0 else cont_s

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--wsi_csv", required=True)
    ap.add_argument("--clinical_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    args = ap.parse_args()

    params = read_json(os.path.join(args.exp_dir, "params.json"))
    time_col, event_col = params["time_col"], params["event_col"]

    # Load WSI & clinical
    wsi = pd.read_csv(args.wsi_csv)
    wsi = wsi.set_index("slide_id")
    wsi["Case_ID"] = wsi.index.astype(str).str.split("_").str[0]
    clinical = pd.read_csv(args.clinical_csv)
    clinical = normalize_clinical_ids(clinical)
    clinical = clinical.drop(columns=[c for c in [time_col, event_col] if c in clinical.columns], errors="ignore")

    data = wsi.merge(clinical, on="Case_ID", how="left")
    slide_ids = data.index
    case_ids  = data["Case_ID"].values
    wsi_feature_cols = [c for c in wsi.columns if c != "Case_ID"]

    # Collect folds
    fold_ids = sorted([int(f.split("FOLD")[1].split(".")[0]) for f in os.listdir(args.exp_dir) if f.startswith("model_FOLD") and f.endswith(".pkl")])
    if not fold_ids:
        raise RuntimeError("No saved fold models found in exp_dir")

    risks = []
    for fid in fold_ids:
        model   = joblib.load(os.path.join(args.exp_dir, f"model_FOLD{fid}.pkl"))
        scalers = joblib.load(os.path.join(args.exp_dir, f"scalers_FOLD{fid}.pkl"))
        meta    = read_json(os.path.join(args.exp_dir, f"clinmeta_FOLD{fid}.json"))

        # blocks
        Xw = data[wsi_feature_cols].values
        if meta["num_cols"] or meta["cat_cols"]:
            df_clin = data[meta["num_cols"] + meta["cat_cols"]].copy()
            Xcnum, Xccat = clinical_transform_apply_task1(df_clin, meta)
        else:
            Xcnum = np.empty((len(data), 0)); Xccat = np.empty((len(data), 0))

        X = transform_with_scalers(Xw, Xcnum, Xccat, scalers)
        Z = model.predict(X)
        risks.append(Z)

    risk_ens = np.mean(np.vstack(risks), axis=0)

    out = pd.DataFrame({
        "slide_id": slide_ids,
        "Case_ID": case_ids,
        "Risk": risk_ens
    }).set_index("slide_id")
    out.to_csv(args.output_csv)
    print(f"Saved ensemble risk scores to {args.output_csv}")

if __name__ == "__main__":
    main()
