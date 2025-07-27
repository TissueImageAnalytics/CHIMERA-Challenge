import json
from pprint import pprint
from pathlib import Path
from tiatoolbox.wsicore.wsireader import WSIReader
import torch
from glob import glob
## ==========User defined functions=============== ##
## Changes to the inference.py, assuming this will be the entry point

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")
MODEL_DIR = Path("/opt/ml/model")


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))

    

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
    return clinical_data


def extract_MRI_feats():
    # state_dict = torch.load(model_dir / "a_tarball_subdirectory" / "MRI_model_wts.pt", map_location='cpu')
    # model.load_state_dict(state_dict)
    # ...
    # mri = model(volumne_tensor)
    # return mri
    t2_dir = INPUT_PATH / "images/axial-t2-prostate-mri"
    adc_dir = INPUT_PATH / "images/axial-adc-prostate-mri"
    hbv_dir = INPUT_PATH / "images/transverse-hbv-prostate-mri"
    t2_mask_dir = INPUT_PATH / "images/prostate-tissue-mask-for-axial-t2-prostate-mri"

    t2_path_list = glob(str(t2_dir / "*.mha"))
    adc_path_list = glob(str(adc_dir / "*.mha"))
    hbv_path_list = glob(str(hbv_dir / "*.mha"))    
    t2_mask_path_list = glob(str(t2_mask_dir / "*.mha"))

    pprint("MRI files found:")
    pprint(t2_path_list)
    pprint(adc_path_list)
    pprint(hbv_path_list)
    pprint(t2_mask_path_list)

    return None


def extract_WSI_feats():

    wsi_dir = INPUT_PATH / "images/prostatectomy-wsi"

    wsi_path_list = glob(str(wsi_dir / "*.tif")) + glob(str(wsi_dir / "*.tiff")) + glob(str(wsi_dir / "*.svs")) + glob(str(wsi_dir / "*.ndpi"))

    pprint("WSI files found:")
    pprint(wsi_path_list)

    # select the first WSI
    wsi_path = wsi_path_list[0]
    pprint(f"Selected WSI: {wsi_path}")

    reader = WSIReader.open(wsi_path)

    pprint(reader.info.as_dict())
    # command = [
    #     "python", "run_batch_of_slides.py",
    #     "--task", "all",
    #     "--wsi_dir", wsi_dir,
    #     "--job_dir", temp_output_dir,
    #     "--slide_encoder", "titan",
    #     "--mag", "20",
    #     "--patch_size", "1024",
    #     "--max_workers", "32"
    # ]
    # subprocess.run(command, check=True)
    # ...
    # wsi = aggregate(temp_output_dir)
    return None


## ==========Challenge functions=============== ##
## Changes to the inference.py/interf9_handler(), assuming this will be the entry point

## I assume we would need to implement all (i.e. from 0 to 9) of the below handlers but just put the last one

def predict_score(clinical_feats, mri_feats, wsi_feats):
    """Predict the score using the model.
    
    Args:
        clinical_feats (dict): Clinical features.
        mri_feats (torch.Tensor): MRI features.
        wsi_feats (torch.Tensor): WSI features.
    
    Returns:
        float: Predicted score.
    """
    # model = Test_Model(
    #     clin_dim=6,
    #     mri_dim=2048,
    #     wsi_dim=768
    # )
    # state_dict = torch.load(MODEL_DIR / "model_wts.pt", map_location='cpu')
    # model.load_state_dict(state_dict)

    # score = model(clin_feats, mri_feats, wsi_feats)

    score = 1.0
    return score

def generic_handler():      
    clin_feats = extract_clinical_feats() ## user defined function
    mri_feats = extract_MRI_feats() ## user defined function, returns a single vector for the whole case
    wsi_feats = extract_WSI_feats() ## user defined function, returns a single vector for the whole case

    output_time_to_biochemical_recurrence_for_prostate_cancer = predict_score(clin_feats, mri_feats, wsi_feats)

    write_json_file(
        location=OUTPUT_PATH
        / "time-to-biochemical-recurrence-for-prostate-cancer-months.json",
        content=output_time_to_biochemical_recurrence_for_prostate_cancer,
    )

    return 0


if __name__ == "__main__":
    generic_handler()