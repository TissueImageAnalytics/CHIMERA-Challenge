"""
Radiomic features extracted without any normalisation. Need to ammend to do z-score normalisation within the mask etc.
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from matplotlib import pyplot as plt
from tqdm import tqdm


def resample_mask_to_image(mask, reference_image):
    """Resample the mask to match the reference image geometry."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(mask)

def calculate_average_intensity(image, mask):
    """Calculate average intensity within the mask."""
    image_data = sitk.GetArrayFromImage(image)
    mask_data = sitk.GetArrayFromImage(mask)
    masked_voxels = image_data[mask_data > 0]
    return float(np.mean(masked_voxels)) if masked_voxels.size > 0 else np.nan


def visualize_mask_overlay(image, mask, output_path, modality_name):
    image_np = sitk.GetArrayFromImage(image)
    mask_np = sitk.GetArrayFromImage(mask)
    middle_index = image_np.shape[0] // 2
    image_slice = image_np[middle_index]
    mask_slice = mask_np[middle_index]

    plt.figure(figsize=(6, 6))
    plt.imshow(image_slice, cmap='gray')
    # plt.imshow(mask_slice, cmap='Reds', alpha=0.4)
    plt.imshow(np.ma.masked_where(mask_slice == 0, mask_slice), cmap='Reds', alpha=0.5)
    plt.title(f"{modality_name} with Mask Overlay")
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"{modality_name} mask shape: {mask_np.shape}")
    print(f"{modality_name} non-zero voxels in mask: {np.sum(mask_np > 0)}")

def save_mask_contour_overlay(image, mask, output_path, title=""):
    """
    Save a visualization of the middle slice of the image with the mask contour overlaid.
    """
    # Convert to numpy arrays
    image_np = sitk.GetArrayFromImage(image)
    mask_np = sitk.GetArrayFromImage(mask)

    # Select the middle slice
    z = image_np.shape[0] // 2
    img_slice = image_np[z]
    mask_slice = mask_np[z]

    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(img_slice, cmap='gray')
    plt.contour(mask_slice, colors='red', linewidths=1)
    plt.axis('off')
    plt.title(title)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def extract_case_features(case_dir, output_dir, extractor, visualize=False):
    """Extract radiomics and intensity features for a single case."""
    # os.makedirs(output_dir, exist_ok=True)
    case_name = os.path.basename(case_dir)
    
    # Define image paths
    adc_path = os.path.join(case_dir, f'{case_name}_0001_adc.mha')
    hbv_path = os.path.join(case_dir, f'{case_name}_0001_hbv.mha')
    t2w_path = os.path.join(case_dir, f'{case_name}_0001_t2w.mha')
    mask_path = os.path.join(case_dir, f'{case_name}_0001_mask.mha')
    
    # Load images
    adc_img = sitk.ReadImage(adc_path)
    hbv_img = sitk.ReadImage(hbv_path)
    t2w_img = sitk.ReadImage(t2w_path)
    mask_img = sitk.ReadImage(mask_path)

    # Calculate volume
    spacing = mask_img.GetSpacing()
    voxel_volume = np.prod(spacing)
    mask_array = sitk.GetArrayFromImage(mask_img)
    total_volume = float(np.sum(mask_array > 0) * voxel_volume)

    # Resample mask and calculate average intensities
    resampled_mask_adc = resample_mask_to_image(mask_img, adc_img)
    resampled_mask_hbv = resample_mask_to_image(mask_img, hbv_img)
    resampled_mask_t2w = resample_mask_to_image(mask_img, t2w_img)

    if visualize:
        os.makedirs(output_dir, exist_ok=True)  
        # visualize_mask_overlay(adc_img, resampled_mask_adc, os.path.join(output_dir, "adc_mask_overlay.png"), "ADC")
        # visualize_mask_overlay(hbv_img, resampled_mask_hbv, os.path.join(output_dir, "hbv_mask_overlay.png"), "HBV")
        # visualize_mask_overlay(t2w_img, resampled_mask_t2w, os.path.join(output_dir, "t2w_mask_overlay.png"), "T2W")
        # Save contour overlays
        save_mask_contour_overlay(adc_img, resampled_mask_adc, os.path.join(output_dir, "adc_mask_contour.png"), title="ADC + Mask Contour")
        save_mask_contour_overlay(hbv_img, resampled_mask_hbv, os.path.join(output_dir, "hbv_mask_contour.png"), title="HBV + Mask Contour")
        save_mask_contour_overlay(t2w_img, resampled_mask_t2w, os.path.join(output_dir, "t2w_mask_contour.png"), title="T2W + Mask Contour")


    avg_intensity_adc = calculate_average_intensity(adc_img, resampled_mask_adc)
    avg_intensity_hbv = calculate_average_intensity(hbv_img, resampled_mask_hbv)
    avg_intensity_t2w = calculate_average_intensity(t2w_img, resampled_mask_t2w)

    # Extract radiomic features
    features_adc = extractor.execute(adc_img, resampled_mask_adc)
    features_hbv = extractor.execute(hbv_img, resampled_mask_hbv)
    features_t2w = extractor.execute(t2w_img, resampled_mask_t2w)

    # Prefix features
    features_adc = {f"ADC_{k}": v for k, v in features_adc.items()}
    features_hbv = {f"HBV_{k}": v for k, v in features_hbv.items()}
    features_t2w = {f"T2W_{k}": v for k, v in features_t2w.items()}

    # Combine all features
    summary = {
        'case_id': os.path.basename(os.path.normpath(case_dir)),
        'total_volume_mm3': total_volume,
        'avg_intensity_adc': avg_intensity_adc,
        'avg_intensity_hbv': avg_intensity_hbv,
        'avg_intensity_t2w': avg_intensity_t2w
    }

    all_features = {**summary, **features_adc, **features_hbv, **features_t2w}
    return all_features

def process_all_cases(base_data_dir, output_csv_path, visualize=False):
    """Process all cases in the base directory and save features to CSV."""
    # Initialize extractor
    params = {
        'binWidth': 25,
        'resampledPixelSpacing': None,
        'interpolator': 'sitkBSpline',
        'geometryTolerance': 1e-4,
        'verbose': True
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**params)

    all_feature_rows = []
    for case_id in tqdm(sorted(os.listdir(base_data_dir))):
        case_dir = os.path.join(base_data_dir, case_id)
        if not os.path.isdir(case_dir):
            continue
        output_dir = os.path.join(os.path.dirname(output_csv_path), case_id)
        try:
            features = extract_case_features(case_dir, output_dir, extractor, visualize=visualize)
            all_feature_rows.append(features)
        except Exception as e:
            print(f"Failed to process case {case_id}: {e}")

    df = pd.DataFrame(all_feature_rows)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved all features to {output_csv_path}")

if __name__ == "__main__":
    base_data_dir = "/media/u1973415/data/u1973415/Chimera/data/task_1/radiology/images/"
    output_csv_path = "/media/u1973415/data/u1973415/Chimera/output/task_1/radiomics_features_all_cases_new.csv"
    process_all_cases(base_data_dir, output_csv_path, visualize=True)