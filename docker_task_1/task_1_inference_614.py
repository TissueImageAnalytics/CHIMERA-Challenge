import json
import os
from pprint import pprint
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import pandas as pd
from glob import glob
from collections import OrderedDict, defaultdict
import SimpleITK as sitk
from sklearn.preprocessing import StandardScaler
from skimage.transform import resize
import joblib

from data_utils import convert_mixed_column_to_numeric
from config614 import *
from model614 import MultimodalSurvivalModel
import h5py

## ==========User defined functions=============== ##
## Changes to the inference.py, assuming this will be the entry point

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")
SURVIVAL_WEIGHTS_PATH = Path("/opt/app/resources/weights_scalers_folder")
SCALER_WEIGHTS_PATH = Path("/opt/app/resources/weights_scalers_folder")


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))

def get_or_fit_scaler(name, train_array, fit=True, fold=0):
    os.makedirs(SCALER_WEIGHTS_PATH, exist_ok=True)
    scaler_path = os.path.join(SCALER_WEIGHTS_PATH, f"{name}_scaler_run_9_{fold}.pkl")
    
    if fit:
        scaler = StandardScaler()
        scaler.fit(train_array)
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)

    return scaler


def maybe_scale(name, train_array, val_array, fit=True, fold=0):
    scaler = get_or_fit_scaler(name, train_array, fit=fit, fold=fold)
    train_scaled = scaler.transform(train_array) if train_array is not None else None
    val_scaled = scaler.transform(val_array) if val_array is not None else None
    return train_scaled, val_scaled


def strip_prefix_if_present(state_dict, prefix="module."):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k[len(prefix):] if k.startswith(prefix) else k
        new_state_dict[new_key] = v
    return new_state_dict


def resample_to_reference(img, reference):
    """Resample 'img' to match 'reference' image space."""
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(reference)
    resample.SetInterpolator(sitk.sitkLinear)
    return resample.Execute(img)


def apply_mask(img_array, mask_array):
    return img_array * (mask_array > 0)

# Preprocessing for each slice
pre_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 120)),
    transforms.Normalize([0.5], [0.5])
])


def extract_features_from_volume(volume, model, device):
    desired_slices = 19

    # Resize each slice to (128, 120)
    resized = np.zeros((volume.shape[0], 128, 120), dtype=np.float32)
    for i in range(volume.shape[0]):
        resized[i] = resize(volume[i], (128, 120), mode='reflect', anti_aliasing=True)

    # Crop/pad to 19 slices
    num_slices = resized.shape[0]
    if num_slices >= desired_slices:
        center = num_slices // 2
        start = max(center - desired_slices // 2, 0)
        end = start + desired_slices
        resized = resized[start:end]
    else:
        # Pad equally on top and bottom if possible
        pad_total = desired_slices - num_slices
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        resized = np.pad(resized, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='constant')

    # Now safe to apply transforms
    slices = [pre_transforms(resized[i]) for i in range(desired_slices)]
    volume_tensor = torch.stack(slices, dim=1).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(volume_tensor)
        features = F.adaptive_avg_pool3d(features, output_size=1).squeeze()

    return features.cpu().numpy()


def extract_clinical_feats():
    """Read clinical data from JSON file.
        Simply returns the clinical data as a dictionary.
    Args:
        json_file_path (str): Path to the JSON file containing clinical data.
    Returns:
        dict: Clinical data.
    """
    with open(INPUT_PATH / "chimera-clinical-data-of-prostate-cancer-patients.json", "r") as f:
        clinical_data = json.loads(f.read())

    pprint("Clinical data found:")
    pprint(clinical_data)

    # Extract raw feature values from JSON
    features = [clinical_data.get(feat, None) for feat in CLINICAL_FEATURES]
 
    # Wrap into a DataFrame for consistent preprocessing
    df = pd.DataFrame([features], columns=CLINICAL_FEATURES)
 
    # Apply mixed column conversion
    for col in MIXED_COLS:
        if col in df.columns:
            df[col] = convert_mixed_column_to_numeric(df[col])
 
    # Convert all clinical features to numeric, coerce errors to NaN
    df[CLINICAL_FEATURES] = df[CLINICAL_FEATURES].apply(pd.to_numeric, errors='coerce')
 
    if df[CLINICAL_FEATURES].isnull().any().any():
        # raise ValueError("Missing or invalid clinical feature(s) in JSON input")
        df[CLINICAL_FEATURES] = df[CLINICAL_FEATURES].fillna(0)

    print("Clinical features extracted.")

    clinical_feats = df.values.astype(np.float32)

    return clinical_feats



def extract_radiomic_feats():
    # state_dict = torch.load(model_dir / "a_tarball_subdirectory" / "MRI_model_wts.pt", map_location='cpu')
    # model.load_state_dict(state_dict)
    # ...
    # mri = model(volumne_tensor)
    # return mri
    t2_mask_dir = INPUT_PATH / "images/prostate-tissue-mask-for-axial-t2-prostate-mri"
    t2_mask_path_list = glob(str(t2_mask_dir / "*.mha"))

    pprint("MRI files found:")
    pprint(t2_mask_path_list)

    # select the first mask
    mask_path = t2_mask_path_list[0]
    mask_img = sitk.ReadImage(mask_path)

    # Calculate volume
    spacing = mask_img.GetSpacing()
    voxel_volume = np.prod(spacing)
    mask_array = sitk.GetArrayFromImage(mask_img)
    total_volume = float(np.sum(mask_array > 0) * voxel_volume)  

    print("Radiomic features extracted.")

    total_volume = np.array([total_volume], dtype=np.float32)  # Convert to numpy array
    # Convert to [1,1] shape for consistency
    if total_volume.ndim == 1:
        total_volume = total_volume.reshape(1, 1)

    return total_volume


def extract_WSI_feats():

    wsi_dir = INPUT_PATH / "images/prostatectomy-wsi"

    wsi_path_list = glob(str(wsi_dir / "*.tif")) + glob(str(wsi_dir / "*.tiff")) + glob(str(wsi_dir / "*.svs")) + glob(str(wsi_dir / "*.ndpi"))

    pprint("WSI files found:")
    pprint(wsi_path_list)

    # select the first WSI
    wsi_path = wsi_path_list[0]
    pprint(f"Selected WSI: {wsi_path}")

    # TRIDENT Features
    try:
        trident_dir = OUTPUT_PATH / "trident_processed"
        trident_slide_features_titan_dir = trident_dir / "10x_896px_0px_overlap" / "slide_features_prism"
        print(f"TRIDENT slide features directory: {trident_slide_features_titan_dir}")
        print(os.listdir(trident_slide_features_titan_dir))
        wsi_features_list = glob(str(trident_slide_features_titan_dir / "*.h5"))
        wsi_feature_path = wsi_features_list[0]
        with h5py.File(wsi_feature_path, 'r') as f:
            features = f['features'][()]
            features = torch.tensor(features, dtype=torch.float32)

        print("WSI features extracted.")

        if features.ndim == 1:
            features = features.reshape(1, -1)

        return features

    except Exception as e:
        print(f"Error occurred while reading TRIDENT features: {e}")

        return torch.zeros((1,1280), dtype=torch.float32)  # Default to zero vector if error occurs


## ==========Challenge functions=============== ##
## Changes to the inference.py/interf9_handler(), assuming this will be the entry point

## I assume we would need to implement all (i.e. from 0 to 9) of the below handlers but just put the last one

def predict_score(clinical_feats, radiomic_feats, wsi_feats):
    """Predict the score using the model.
    
    Args:
        clinical_feats (np.ndarray): Clinical features.
        radiomic_feats (np.ndarray): Radiomic features.
        wsi_feats (torch.Tensor): WSI features.
    
    Returns:
        float: Predicted score.
    """

    ### Combine radiomic and clinical features
    c_dim = 10
    m_dim = 0
    # w_dim = 768 # For TITAN!
    w_dim = 1280  # For PRISM!
    r_dim = 1

    if radiomic_feats is not None:
    #     clinical_feats = np.concatenate([clinical_feats, radiomic_feats], axis=1)
        c_dim += r_dim

    # Prepare model template
    model = MultimodalSurvivalModel(
        clin_dim=c_dim,
        mri_dim=m_dim,
        wsi_dim=w_dim,
        fusion_type=FUSION_TYPE,
        survival_model=SURVIVAL_MODEL,
        time_bins=TIME_BINS
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if USE_ENSEMBLE:
        pmf_all_folds = []

        for fold_idx in range(NUM_FOLDS):
            model_path = os.path.join(SURVIVAL_WEIGHTS_PATH, f"best_model_run9_fold{fold_idx}.pt")
            if not os.path.exists(model_path):
                print(f"[Warning] Model missing for fold {fold_idx}: {model_path}")
                continue

            fold_clin_array, _ = maybe_scale("clinical", clinical_feats, clinical_feats, fit=False, fold=fold_idx) if USE_CLINICAL_FEATURES else (None, None)
            fold_mri_array, _ = (None, None)
            fold_radi_array, _ = maybe_scale("radiomic", radiomic_feats, radiomic_feats, fit=False, fold=fold_idx) if USE_RADIOMIC_FEATURES else (None, None)
            fold_wsi_array, _ = maybe_scale("wsi", wsi_feats, wsi_feats, fit=False, fold=fold_idx) if USE_WSI_FEATURES else (None, None)

            fold_clin_array = np.concatenate([fold_clin_array, fold_radi_array], axis=1)
            
            if fold_clin_array is not None and fold_clin_array.ndim == 1:
                fold_clin_array = fold_clin_array.reshape(1, -1)
            if fold_wsi_array is not None and fold_wsi_array.ndim == 1:
                fold_wsi_array = fold_wsi_array.reshape(1, -1)


 

            clin_tensor = torch.tensor(fold_clin_array, dtype=torch.float32).to(device) if fold_clin_array is not None else torch.zeros((1, c_dim), device=device)
            mri_tensor = torch.zeros((1, m_dim), device=device)
            wsi_tensor = torch.tensor(fold_wsi_array, dtype=torch.float32).to(device) if fold_wsi_array is not None else torch.zeros((1, w_dim), device=device)

            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            with torch.no_grad():
                out = model(clinical_feat=clin_tensor, mri_feat=mri_tensor, wsi_feat=wsi_tensor)
                pmf_all_folds.append(out[0].cpu().numpy())

        if not pmf_all_folds:
            raise RuntimeError("No models loaded for ensemble inference.")


        avg_pmf = np.mean(pmf_all_folds, axis=0)

    ## to test with a single json clinical file
    # time_bins = np.arange(TIME_BINS)
    # score = float(np.sum(avg_pmf * time_bins))
    # print('score: ', score)
    # return score

    ##Expected time = sum_t p(t) * t
    time_bins = np.arange(TIME_BINS)
    expected_times = np.sum(avg_pmf * time_bins[None, :], axis=1)

    # return dict(zip(test_case_ids, expected_times))
    return expected_times[0]

def generic_handler():      
    clin_feats = extract_clinical_feats() ## user defined function
    radiomic_feats = extract_radiomic_feats() ## user defined function, returns a single vector for the whole case
    wsi_feats = extract_WSI_feats() ## user defined function, returns a single vector for the whole case

    output_time_to_biochemical_recurrence_for_prostate_cancer = predict_score(clin_feats, radiomic_feats, wsi_feats)

    print(f"Predicted time: {output_time_to_biochemical_recurrence_for_prostate_cancer}")

    write_json_file(
        location=OUTPUT_PATH
        / "time-to-biochemical-recurrence-for-prostate-cancer-months.json",
        content=output_time_to_biochemical_recurrence_for_prostate_cancer,
    )

    return 0


if __name__ == "__main__":
    generic_handler()