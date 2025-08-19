# === Configuration ===
# === MRI deep features extraction ===
MRI_MODEL_WEIGHTS_PATH = "/home/u1970167/chimera/task1/radiology/resnet_50_23dataset.pth" ## patch to weight from a resnet_50 pretrained on CT, MRI images. :https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet50/tree/main
#resnet_50_23dataset.pth
MRI_IMG_DIR = "/home/u1970167/chimera/task1/radiology/images/"
MRI_FEATURE_DIR = "/home/u1970167/chimera/task1/radiology/features/" # Directory to save the MRI features (as .npy files) to
COMBINE_MODALITIES = True  # True = average features; False = keep separate per modality (t2w, adc, hbv)
APPLY_ROI = True           # True = extract features only from ROI; False = use whole image
MODALITIES = ['t2w'] ## to use t2w only: ['t2w'] , to use all: ['t2w', 'adc', 'hbv'] 

RUNS = 1 ## number of times to do the k-folds cross-validation
TASK = 1
USE_CLINICAL_FEATURES = True
USE_MRI_FEATURES = False
USE_WSI_FEATURES = False

CENSORING = 120  # months (10 years)
TUNING_TRIALS = 1 #00
NUM_FOLDS = 5
EMBEDDER = 'titan' ## 'titan' or 'prism'
MASKER = 'grandqc'  ## Tissue masking: 'challengeMasks' (for mask provided by the organisers) | 'grandqc' | '' (hest masking)
PATCHES = '500' ## max number of patches per case
MAG = 10

TSR_STRUCTURE= 'SurvivalNet' ## TSR structure. options: 'RankModel', 'DeepSurv', 'SurvivalNet', 'SimpleSurv'

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

CLINICAL_JSON_DIR = '/home/u1970167/chimera/task1/clinical_data_v2/' ## path to the clincal json files for each case. this is needed when doing inference from clinical features from json files instead of a single csv file

MIXED_COLS = ["pT_stage"] ## pT_stage has values such 2, 2a, 2b, 3 etc. these needs to be changed to values like 2.0, 2.1, 2.2, 3.0 etc repectively

# === Paths ===

if EMBEDDER == 'titan':
    PATCH_SIZE = 1024
    WSI_FEATURES_CSV = f"/home/u1970167/chimera/task1/pathology/features/titan/Task1_titan_{MAG}x_{PATCH_SIZE}_{MASKER}_{PATCHES}_embeddings.csv"  # csv file with WSI-level feature vector (e.g. From CONCH+TITAN embeddings at 20x of 1024 patch size)
if EMBEDDER == 'prism':
    PATCH_SIZE = 896
    WSI_FEATURES_CSV = f"/home/u1970167/chimera/task1/pathology/features/prism/Task1_prism_{MAG}x_{PATCH_SIZE}_{MASKER}_{PATCHES}_embeddings.csv"

CLINICAL_CSV = "/home/u1970167/chimera/task1/clinical_data.csv"
TIME_COLUMN = 'time_to_follow-up/BCR'
EVENT_COLUMN = 'BCR'
SLIDE_ID_COLUMN = 'Slide_ID'
MRI_FEATURE_DIR = "/home/u1970167/chimera/task1/radiology/features/"

FOLDS_CSV = f"/home/u1970167/chimera/task{TASK}/experiments/folds/task{TASK}_folds.csv" 
EXCLUDE_COLS = ["Case_ID", TIME_COLUMN, EVENT_COLUMN, SLIDE_ID_COLUMN]  # Make sure these columns are not used as features

HIDDEN_DIM = 128  # projection dims
VERBOSE = True
SCALE_DATA = True

FUSION_TYPE = 'linear'  # Options: 'modality' (softmax weights per modality) or 'linear' (linear layer after concat) or 'simple' (concat with no learnable params),  'cross_att' (AS cross attention fusion)

# === Feature dimensions ===
if EMBEDDER == 'prism':
    WSI_FEATURE_DIM = 1280  # deep features from WSIs. Titan: 768, Prism: 1280
elif EMBEDDER == 'titan':
    WSI_FEATURE_DIM = 768  # deep features from WSIs. Titan: 768, Prism: 1280

MRI_FEATURE_DIM = 2048    # 2048-dim MRI features

AGGREG_CASE_WSI = False ## False: just select the first WSI in alphabetical order for reproducibility in cross-validations. True: mean aggregates the multiple WSIs per case.

# === inference ===
USE_ENSEMBLE = True  # True means ensemble the results of the best models from the 5 folds across the N runs. False means use the best of the 5 folds across the N runs. Currently, only ENSEMBLE is implemented so do not set to False

INFER_LOCAL = False ## True means: use the features in the form of csv files rather than Challenge expected json (for clinical), wsi_path (for WSIs) and mri_path (for MRI files);
                   ## False means: Challenge expected json (for clinical), wsi_path (for WSIs) and mri_path (for MRI files); This is currenly only implemented for clinical features.
                   ## To verify the local code works the same in the docker container, compare scores for 1 or 2 cases. the scores are saved to the results folder to a csv file _train_predictions.csv
INFER_SINGLE = True ## True means: print score for a single file using the Challenge expected interface i.e. using paths to the files for a single case to generate score for a single case.
                    ## Note: the INFER_SINGLE will not produce c-index but only print the score of the first case from the csv file CLINICAL_CSV


# === global path ====
GLOBAL_DIR = f"/home/u1970167/chimera/task{TASK}/experiments/TSR_results/Task_{TASK}_Clin_{USE_CLINICAL_FEATURES}_MRI_{USE_MRI_FEATURES}_WSI_{USE_WSI_FEATURES}_MAG_{MAG}_PATCH_{PATCH_SIZE}_PATCHES_{PATCHES}_MASK_{MASKER}_EMBED_{EMBEDDER}_TSR_{TSR_STRUCTURE}_TUN_{TUNING_TRIALS}/"