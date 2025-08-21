from config import *
import glob
import SimpleITK as sitk
import numpy as np
import pandas as pd
import os

def extract_radiomic_features():
    case_dirs = [d for d in os.listdir(MRI_IMG_DIR) if os.path.isdir(os.path.join(MRI_IMG_DIR, d))]
    all_data = []

    for case_id in case_dirs:
        case_path = os.path.join(MRI_IMG_DIR, case_id)
        mask_files = glob.glob(os.path.join(case_path, "*_mask.mha"))

        if not mask_files:
            print(f"No mask found for case: {case_id}")
            continue

        mask_path = mask_files[0]
        mask_img = sitk.ReadImage(mask_path)

        # Calculate volume
        spacing = mask_img.GetSpacing()
        voxel_volume = np.prod(spacing)
        mask_array = sitk.GetArrayFromImage(mask_img)
        total_volume = float(np.sum(mask_array > 0) * voxel_volume)

        all_data.append({
            "Case_ID": case_id,
            "Volume": total_volume
        })

    if not all_data:
        print("No data found. Exiting.")
        return

    # Save to CSV
    df = pd.DataFrame(all_data)
    df.to_csv(RADIOMIC_CSV, index=False)
    print(f"Saved features for {len(df)} cases to {RADIOMIC_CSV}")
