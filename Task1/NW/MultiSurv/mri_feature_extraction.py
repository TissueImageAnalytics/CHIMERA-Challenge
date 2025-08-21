import os
import re
import torch
import numpy as np
import torch.nn as nn
import SimpleITK as sitk
from torchvision import transforms
from tqdm import tqdm
from collections import OrderedDict, defaultdict
from skimage.transform import resize
import torch.nn.functional as F
import resnet
from config import *

# Preprocessing for each slice
pre_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 120)),
    transforms.Normalize([0.5], [0.5])
])

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

def load_mha_volume(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)

def resample_to_reference(img, reference):
    """Resample 'img' to match 'reference' image space."""
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(reference)
    resample.SetInterpolator(sitk.sitkLinear)
    return resample.Execute(img)

def apply_mask(img_array, mask_array):
    return img_array * (mask_array > 0)

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

def extract_features_for_patient(patient_dir, model, device, scan_id):
    modality_features = []

    mask_path = os.path.join(patient_dir, f"{scan_id}_mask.mha")
    mask_img = sitk.ReadImage(mask_path) if os.path.exists(mask_path) else None

    for mod in MODALITIES:
        mod_path = os.path.join(patient_dir, f"{scan_id}_{mod}.mha")
        if not os.path.exists(mod_path):
            raise FileNotFoundError(f"{mod_path} not found")

        img = sitk.ReadImage(mod_path)

        if APPLY_ROI and mask_img:
            mask_resampled = resample_to_reference(mask_img, img)
            mask_array = sitk.GetArrayFromImage(mask_resampled)
            img_array = apply_mask(sitk.GetArrayFromImage(img), mask_array)
        else:
            img_array = sitk.GetArrayFromImage(resample_to_reference(img, mask_img)) if mask_img else sitk.GetArrayFromImage(img)

        feat = extract_features_from_volume(img_array, model, device)
        modality_features.append(feat)

    if COMBINE_MODALITIES:
        return np.mean(modality_features, axis=0)
    else:
        return dict(zip(MODALITIES, modality_features))

def extract_and_save_features(model, device):
    # Construct output folder path with ROI flag and selected modalities
    mod_str = "_".join(MODALITIES)
    feature_dir = os.path.join(MRI_FEATURE_DIR, f"ROI_{APPLY_ROI}_{mod_str}")
    os.makedirs(feature_dir, exist_ok=True)

    # Get all patient directories
    patient_dirs = [os.path.join(MRI_IMG_DIR, d) for d in os.listdir(MRI_IMG_DIR) if os.path.isdir(os.path.join(MRI_IMG_DIR, d))]

    for patient_dir in tqdm(patient_dirs):
        try:
            all_files = os.listdir(patient_dir)
            t2w_files = [f for f in all_files if f.lower().endswith("_t2w.mha")]
            if not t2w_files:
                print(f"No T2W files in {patient_dir}")
                continue

            # Group scans by patient ID
            scan_groups = defaultdict(list)
            for f in t2w_files:
                match = re.match(r"([a-zA-Z0-9]+)_\d+_t2w\.mha$", f.lower())
                if match:
                    patient_id = match.group(1)
                    scan_groups[patient_id].append(f)

            for patient_id, scans in scan_groups.items():
                for t2_file in scans:
                    scan_id = t2_file.replace("_t2w.mha", "")

                    # Check if feature already exists
                    if COMBINE_MODALITIES:
                        output_path = os.path.join(feature_dir, f"{scan_id}.npy")
                        if os.path.isfile(output_path):
                            print(f"Skipping {scan_id}, already exists.")
                            continue
                    else:
                        all_exist = all([
                            os.path.isfile(os.path.join(feature_dir, f"{scan_id}_{mod}.npy"))
                            for mod in MODALITIES
                        ])
                        if all_exist:
                            print(f"Skipping {scan_id}, all modality features exist.")
                            continue

                    # Extract and save features
                    try:
                        features = extract_features_for_patient(patient_dir, model, device, scan_id)
                        if COMBINE_MODALITIES:
                            np.save(os.path.join(feature_dir, f"{scan_id}.npy"), features)
                        else:
                            for mod, vec in features.items():
                                np.save(os.path.join(feature_dir, f"{scan_id}_{mod}.npy"), vec)
                    except Exception as e:
                        print(f"Failed to process {scan_id}: {e}")
        except Exception as e:
            print(f"Error with directory {patient_dir}: {e}")

def extract_MRI_features():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_medicalnet_resnet50(MRI_MODEL_WEIGHTS_PATH, device)

    extract_and_save_features(model, device)

# if __name__ == "__main__":
#     root_data_dir = "/home/u1970167/chimera/task1/radiology/images/"
#     feature_output_dir = "/home/u1970167/chimera/task1/radiology/features_ROI/"
#     model_weights_path = "/home/u1970167/chimera/task1/radiology/resnet_50_23dataset.pth"

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = load_medicalnet_resnet50(model_weights_path, device)

#     extract_and_save_features(root_data_dir, feature_output_dir, model, device)
