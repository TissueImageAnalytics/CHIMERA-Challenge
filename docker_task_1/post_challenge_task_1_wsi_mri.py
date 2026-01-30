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
from collections import OrderedDict
import SimpleITK as sitk
from sklearn.preprocessing import StandardScaler
from skimage.transform import resize
import joblib

from post_challenge_config import *
import resnet
from post_challenge_model import MultimodalSurvivalModel
import h5py

## ==========User defined functions=============== ##
## Changes to the inference.py, assuming this will be the entry point

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")
MRI_WEIGHTS_PATH = Path("/opt/app/resources/wsi_mri/resnet_50_23dataset.pth")
SURVIVAL_WEIGHTS_PATH = Path("/opt/app/resources/wsi_mri")
SCALER_WEIGHTS_PATH = Path("/opt/app/resources/wsi_mri")


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))
        

def get_or_fit_scaler(name, train_array, fit=True, fold=0):
    os.makedirs(SCALER_WEIGHTS_PATH, exist_ok=True)
    scaler_path = os.path.join(SCALER_WEIGHTS_PATH, f"{name}_scaler_run_2_{fold}.pkl")
    
    if fit:
        print("Fitting scaler for", name)
        scaler = StandardScaler()
        scaler.fit(train_array)
        joblib.dump(scaler, scaler_path)
    else:
        print("Loading scaler for", name)
        scaler = joblib.load(scaler_path)
    
    return scaler


def maybe_scale(name, train_array, val_array, fit=False, fold=0):
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


def load_medicalnet_resnet50(model_weights_path, device):
    model = resnet.resnet50(
        sample_input_D=19,
        sample_input_H=128,
        sample_input_W=120,
        num_seg_classes=23,
        shortcut_type='B',
        no_cuda=False
    )
    checkpoint = torch.load(model_weights_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    state_dict = strip_prefix_if_present(state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


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


def extract_MRI_feats():
    t2_dir = INPUT_PATH / "images/axial-t2-prostate-mri"

    t2_mask_dir = INPUT_PATH / "images/prostate-tissue-mask-for-axial-t2-prostate-mri"

    t2_path_list = glob(str(t2_dir / "*.mha"))
 
    t2_mask_path_list = glob(str(t2_mask_dir / "*.mha"))

    pprint("MRI files found:")
    pprint(t2_path_list)

    pprint(t2_mask_path_list)

    # Just use T2w scan for now
    # select the first mask
    mask_path = t2_mask_path_list[0]
    mask_img = sitk.ReadImage(mask_path)
    t2_path = t2_path_list[0]
    t2_img = sitk.ReadImage(t2_path)

    # Apply mask
    mask_resampled = resample_to_reference(mask_img, t2_img)
    mask_array = sitk.GetArrayFromImage(mask_resampled)
    img_array = apply_mask(sitk.GetArrayFromImage(t2_img), mask_array)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_medicalnet_resnet50(MRI_WEIGHTS_PATH, device)
    feat = extract_features_from_volume(img_array, model, device)


    if feat.ndim == 1:
        feat = feat[np.newaxis, :]

    pprint("MRI features extracted.")

    return feat


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
        trident_slide_features_titan_dir = trident_dir / "10x_1024px_0px_overlap" / "slide_features_titan"
        pprint(f"TRIDENT slide features directory: {trident_slide_features_titan_dir}")
        pprint(os.listdir(trident_slide_features_titan_dir))
        wsi_features_list = glob(str(trident_slide_features_titan_dir / "*.h5"))
        wsi_feature_path = wsi_features_list[0]
        with h5py.File(wsi_feature_path, 'r') as f:
            features = f['features'][()]
            features = torch.tensor(features, dtype=torch.float32)

        pprint("WSI features extracted.")

        if features.ndim == 1:
            features = features.reshape(1, -1)

        return features

    except Exception as e:
        pprint(f"Error occurred while reading TRIDENT features: {e}")


## ==========Challenge functions=============== ##
## Changes to the inference.py/interf9_handler(), assuming this will be the entry point

## I assume we would need to implement all (i.e. from 0 to 9) of the below handlers but just put the last one

def predict_score(mri_feats, wsi_feats):
    """Predict the score using the model.
    
    Args:
        mri_feats (torch.Tensor): MRI features.
        wsi_feats (torch.Tensor): WSI features.
    
    Returns:
        float: Predicted score.
    """

    ### Combine radiomic and clinical features
    m_dim = 2048
    w_dim = 768 # For TITAN!


    # Prepare model template
    model = MultimodalSurvivalModel(
        clin_dim=0,
        mri_dim=m_dim,
        wsi_dim=w_dim,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if USE_ENSEMBLE:
        pmf_all_folds = []

        for fold_idx in range(NUM_FOLDS):
            model_path = os.path.join(SURVIVAL_WEIGHTS_PATH, f"best_model_run2_fold{fold_idx}.pt")
            if not os.path.exists(model_path):
                pprint(f"[Warning] Model missing for fold {fold_idx}: {model_path}")
                continue
            

            fold_clin_array, _ = None, None
            fold_mri_array, _ = maybe_scale("mri", mri_feats, mri_feats, fit=False, fold=fold_idx) if USE_MRI_FEATURES else (None, None)
            fold_wsi_array, _ = maybe_scale("wsi", wsi_feats, wsi_feats, fit=False, fold=fold_idx) if USE_WSI_FEATURES else (None, None)

            if fold_clin_array is not None and fold_clin_array.ndim == 1:
                fold_clin_array = fold_clin_array.reshape(1, -1)
            if fold_mri_array is not None and fold_mri_array.ndim == 1:
                fold_mri_array = fold_mri_array.reshape(1, -1)
            if fold_wsi_array is not None and fold_wsi_array.ndim == 1:
                fold_wsi_array = fold_wsi_array.reshape(1, -1)

            clin_tensor = None
            mri_tensor = torch.tensor(fold_mri_array, dtype=torch.float32).to(device) if fold_mri_array is not None else torch.zeros((1, m_dim), device=device)
            wsi_tensor = torch.tensor(fold_wsi_array, dtype=torch.float32).to(device) if fold_wsi_array is not None else torch.zeros((1, w_dim), device=device)


            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            with torch.no_grad():
                out = model(clinical_feat=None, mri_feat=mri_tensor, wsi_feat=wsi_tensor)
                pmf_all_folds.append(out.cpu().numpy())

        if not pmf_all_folds:
            raise RuntimeError("No models loaded for ensemble inference.")


        avg_pmf = np.mean(pmf_all_folds, axis=0)



    time_bins = np.arange(TIME_BINS)
    expected_times = np.sum(avg_pmf * time_bins[None, :], axis=1)

    return expected_times[0]

def generic_handler():      
    mri_feats = extract_MRI_feats() ## user defined function, returns a single vector for the whole case
    wsi_feats = extract_WSI_feats() ## user defined function, returns a single vector for the whole case

    
    output_time_to_biochemical_recurrence_for_prostate_cancer = predict_score(mri_feats, wsi_feats)

    pprint(f"Predicted time: {output_time_to_biochemical_recurrence_for_prostate_cancer}")

    write_json_file(
        location=OUTPUT_PATH
        / "time-to-biochemical-recurrence-for-prostate-cancer-months.json",
        content=output_time_to_biochemical_recurrence_for_prostate_cancer,
    )

    return 0


if __name__ == "__main__":
    generic_handler()