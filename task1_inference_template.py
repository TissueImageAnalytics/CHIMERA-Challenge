
## ==========User defined functions=============== ##
## Changes to the inference.py, assuming this will be the entry point

def extract_clinical_feat(json_file, clin_feat_list):
    ## clin =get the clin_feat_list from the json_file
    return clin

def extract_MRI_feats(...):
    state_dict = torch.load(model_dir / "a_tarball_subdirectory" / "MRI_model_wts.pt", map_location='cpu')
    model.load_state_dict(state_dict)
    ...
    mri = model(volumne_tensor)
    return mri

def extract_WSI_feats(...):
    load_trident()
    command = [
        "python", "run_batch_of_slides.py",
        "--task", "all",
        "--wsi_dir", wsi_dir,
        "--job_dir", temp_output_dir,
        "--slide_encoder", "titan",
        "--mag", "20",
        "--patch_size", "1024",
        "--max_workers", "32"
    ]
    subprocess.run(command, check=True)
    ...
    wsi = aggregate(temp_output_dir)
    return wsi


## ==========Challenge functions=============== ##
## Changes to the inference.py/interf9_handler(), assuming this will be the entry point

## I assume we would need to implement all (i.e. from 0 to 9) of the below handlers but just put the last one

def interf9_handler():      
    clin_feat_list = torch.load(model_dir / "a_tarball_subdirectory" / "clin_feat_list.txt", map_location='cpu')
    clin_feats = extract_clinical_feat(input_chimera_clinical_data_of_prostate_cancer_patients, clin_feat_list) ## user defined function
    mri_feats = extract_MRI_feats(input_axial_t2_prostate_mri, input_axial_adc_prostate_mri, input_transverse_hbv_prostate_mri, input_prostate_tissue_mask_for_axial_t2_prostate_mri) ## user defined function, returns a single vector for the whole case
    wsi_feats = extract_WSI_feats(input_prostatectomy_tissue_whole_slide_image, input_prostatectomy_tissue_whole_slide_image_1_1, ... input_prostatectomy_tissue_whole_slide_image_1_10) ## user defined function, returns a single vector for the whole case

    ## user define model
    model = Test_Model(
        clin_dim=6,
        mri_dim=2048,
        wsi_dim=768
    )

    state_dict = torch.load(model_dir / "a_tarball_subdirectory" / "model_wts.pt", map_location='cpu')
    model.load_state_dict(state_dict)

    output_time_to_biochemical_recurrence_for_prostate_cancer = model(clin_feats, mri_feats, wsi_feats)