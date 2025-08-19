from tiatoolbox.wsicore.wsireader import WSIReader
import os
import numpy as np
import json
import cv2
from glob import glob
from pprint import pprint
from pathlib import Path

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")

def mask_to_geojson(mask, scale_factor):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    geojson_features = []

    for idx, contour in enumerate(contours):
        contour = contour.squeeze()
        if contour.ndim == 1:
            contour = np.expand_dims(contour, axis=0)
        # Close the contour if it is not closed
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack([contour, contour[0]])
        # Scale the coordinates
        scaled_contour = (contour * scale_factor).astype(int).tolist()
        geojson_feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [scaled_contour]
            },
            "properties": {
                "tissue_id": idx,
            }
        }
        geojson_features.append(geojson_feature)

    geojson_data = {
        "type": "FeatureCollection",
        "name": "prostatectomy-wsi",
        "features": geojson_features
    }
    return geojson_data


def convert_mask_to_geojson(wsi_dir, tissue_mask_dir, save_dir):

    tissue_mask_path_list = glob(str(tissue_mask_dir / "*.tif"))

    pprint("Tissue mask files found:")
    pprint(tissue_mask_path_list)

    # select the first mask
    tissue_mask_path = tissue_mask_path_list[0]

    wsi_mask_path_list = glob(str(wsi_dir / "*.tif"))
    pprint("WSI files found:")
    pprint(wsi_mask_path_list)
    # select the first WSI
    wsi_path = wsi_mask_path_list[0]
    wsi_name_without_ext = os.path.splitext(os.path.basename(wsi_path))[0]



    mask_reader = WSIReader.open(tissue_mask_path)
    print("Mask Reader Information:")
    print(mask_reader.info.as_dict())

    # Load mask at 2x magnification
    mask_thumbnail = mask_reader.slide_thumbnail(resolution=1.25, units='power')
    binary_mask = mask_thumbnail[:,:,0]
    binary_mask = binary_mask.astype(np.uint8)
    # remove small objects and holes
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones((33, 33), np.uint8))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, np.ones((33, 33), np.uint8))

    binary_mask = binary_mask.astype(np.uint8)
    print(binary_mask.shape)

    wsi_reader = WSIReader.open(wsi_path)
    print("WSI Reader Information:")
    print(wsi_reader.info.as_dict())
    wsi_base_power = wsi_reader.info.objective_power
    print(f"WSI base power: {wsi_base_power}")

    # Conver to geojson format
    print("Converting mask to GeoJSON format...")
    scale_factor = wsi_base_power / 1.25


    geojson_data = mask_to_geojson(binary_mask, scale_factor)
    print(f"Number of contours found: {len(geojson_data['features'])}")
    # Save to file
    contour_save_path = os.path.join(save_dir, f"{wsi_name_without_ext}.geojson")
    with open(contour_save_path, 'w') as f:
        json.dump(geojson_data, f, indent=4)
    print(f"GeoJSON data saved to {contour_save_path}")


if __name__ == "__main__":
    wsi_dir = INPUT_PATH / "images/prostatectomy-wsi"
    tissue_mask_dir = INPUT_PATH / "images/prostatectomy-tissue-mask"
    save_dir = OUTPUT_PATH / "trident_processed" / "contours_geojson"

    # Ensure output directory exists
    save_dir.mkdir(parents=True, exist_ok=True)

    convert_mask_to_geojson(wsi_dir, tissue_mask_dir, save_dir)