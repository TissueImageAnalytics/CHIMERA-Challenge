# config.py

# === common =====
SCALE_DATA = True
TASK = 1 ## 1: prostate
RUNS = 10 ## number of times to run the 5-folds cross-validation

# === Base line Cox PH model === ## for cox_baseline.py only
COX_BASELINE_DIR = "/home/u1970167/chimera/task1/experiments/results/Cox_baseline/"

# === MRI deep features extraction ===
MRI_MODEL_WEIGHTS_PATH = "/home/u1970167/chimera/task1/radiology/resnet_50_23dataset.pth" ## patch to weight from a resnet_50 pretrained on CT, MRI images. :https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet50/tree/main
#resnet_50_23dataset.pth
MRI_IMG_DIR = "/home/u1970167/chimera/task1/radiology/images/"
MRI_FEATURE_DIR = "/home/u1970167/chimera/task1/radiology/features/" # Directory to save the MRI features (as .npy files) to
COMBINE_MODALITIES = True  # True = average features; False = keep separate per modality (t2w, adc, hbv)
APPLY_ROI = True           # True = extract features only from ROI; False = use whole image
MODALITIES = ['t2w'] ## to use t2w only: ['t2w'] , to use all: ['t2w', 'adc', 'hbv'] 

# === Data paths ===
CLINICAL_CSV = "/home/u1970167/chimera/task1/clinical_data.csv"        # Clinical features CSV with Case_ID column
RADIOMIC_CSV = "/home/u1970167/chimera/task1/radiology/features/radiology_data.csv" # Radiomic handcrafted features CSV with Case_ID column
CLINICAL_JSON_DIR = '/home/u1970167/chimera/task1/clinical_data_v2/' ## path to the clincal json files for each case. this is needed when doing inference from clinical features from json files instead of a single csv file
EMBEDDER = 'prism' ## 'titan' or 'prism'
MAG = 10
PATCH_SIZE = 1024

if EMBEDDER == 'titan':
    PATCH_SIZE = 512
    WSI_FEATURES_CSV = f"/home/u1970167/chimera/task1/pathology/features/titan/Task1_titan_{MAG}x_{PATCH_SIZE}_embeddings.csv"  # csv file with WSI-level feature vector (e.g. From CONCH+TITAN embeddings at 20x of 1024 patch size)
if EMBEDDER == 'prism':
    PATCH_SIZE = 896
    WSI_FEATURES_CSV = f"/home/u1970167/chimera/task1/pathology/features/prism/Task1_prism_{MAG}x_{PATCH_SIZE}_embeddings.csv"

FOLDS_CSV = f"/home/u1970167/chimera/task1/experiments/folds/task{TASK}_folds.csv"          # Directory for saving/loading fold CSVs
OUTPUT_DIR = "/home/u1970167/chimera/task1/experiments/results/"       # Directory for saving results/models

# === Event & Time Columns in Clinical Data ===
EVENT_COLUMN = "BCR"                # Column name for event indicator (1=event, 0=censored)
TIME_COLUMN = "time_to_follow-up/BCR"  # Column name for time-to-event or censoring

# === Clinical Features to Use ===
CLINICAL_FEATURES = [
    "age_at_prostatectomy",
    "primary_gleason",
    "secondary_gleason",
    "ISUP",
    "pre_operative_PSA",
    "capsular_penetration",
    "positive_surgical_margins",
    "invasion_seminal_vesicles",
    "lymphovascular_invasion",
    "pT_stage"
]

MIXED_COLS = ["pT_stage"] ## pT_stage has values such 2, 2a, 2b, 3 etc. these needs to be changed to values like 2.0, 2.1, 2.2, 3.0 etc repectively

# === Experiment settings ===
USE_CLINICAL_FEATURES = True
USE_RADIOMIC_FEATURES = True
USE_MRI_FEATURES = False
USE_WSI_FEATURES = True

SURVIVAL_MODEL = 'deephit'  # Options: 'cox' or 'deephit'
# Options: 'modality' (softmax weights per modality) or 'linear' (linear layer after concat) or 'simple' (concat with no learnable params) or 
# 'gated_cross' (sample-specific gating + cross-modal attention using Clinical as query), 'cross_att' (AS cross attention fusion)
FUSION_TYPE = 'cross_att'  
DEEPHIT_LOSS = 'uncensored' # 'censored' or 'uncensored'. 'censored' has a extra term for accounting for censored data whereas 'uncensored' only considers uncensored cases
TIME_BINS = 30  # Only for deephit
NUM_FOLDS = 5
SEED = 42
HIDDEN_DIM = 128  # or 128, tune as needed


# === Feature dimensions ===
if EMBEDDER == 'prism':
    W_FEATURE_DIM = 1280  # deep features from WSIs. Titan: 768, Prism: 1280
elif EMBEDDER == 'titan':
    W_FEATURE_DIM = 768  # deep features from WSIs. Titan: 768, Prism: 1280

AGGREG_CASE_WSI = False ## False: just select the first WSI in alphabetical order for reproducibility in cross-validations. True: mean aggregates the multiple WSIs per case.

M_FEATURE_DIM = 2048    # 2048-dim MRI features
R_FEATURE_DIM = 1 ## handcrafted MRI features
CLINICAL_DIM = len(CLINICAL_FEATURES)

# === Training settings ===
EPOCHS = 100
LR = 1e-3
WD = 1e-4
BATCH_SIZE = 16
PATIENCE = 20 ## early-stop on no improvement to c-index
# === Debug ===
VERBOSE = True

# === Inference ===
USE_ENSEMBLE = True  # True means ensemble the results of the best models from the 5 folds across the N runs. False means use the best of the 5 folds across the N runs. Currently, only ENSEMBLE is implemented so do not set to False
INFER_LOCAL = True ## True means: use the features in the form of csv files rather than Challenge expected json (for clinical), wsi_path (for WSIs) and mri_path (for MRI files);
                   ## False means: Challenge expected json (for clinical), wsi_path (for WSIs) and mri_path (for MRI files); This is currenly only implemented for clinical features.
                   ## To verify the local code works the same in the docker container, compare scores for 1 or 2 cases. the scores are saved to the results folder to a csv file _train_predictions.csv
INFER_SINGLE = False ## True means: print score for a single file using the Challenge expected interface i.e. using paths to the files for a single case to generate score for a single case.
                    ## Note: the INFER_SINGLE will not produce c-index but only print the score of the first case from the csv file CLINICAL_CSV
#GLOBAL_DIR = f"{OUTPUT_DIR}CLINICAL_{USE_CLINICAL_FEATURES}_MRI_{USE_MRI_FEATURES}_RADIO_{USE_RADIOMIC_FEATURES}_WSI_{USE_WSI_FEATURES}_MAGNIF_{MAG}_PATCH_SZ_{PATCH_SIZE}_EMBEDDER_{EMBEDDER}_MODEL_{SURVIVAL_MODEL}_FUSION_{FUSION_TYPE}_SCALE_{SCALE_DATA}_EPOCHS_{EPOCHS}/" ## main path for results
GLOBAL_DIR = f"{OUTPUT_DIR}task1_submit2/"