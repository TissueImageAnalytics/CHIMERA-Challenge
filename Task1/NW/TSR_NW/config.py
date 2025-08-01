# === Configuration ===
TASK = 1  # or 3
USE_CLINICAL = True
USE_MRI = True
USE_WSI = True
CENSORING = 120  # months (10 years)
TUNING_TRIALS = 300#00
EMBEDDER = 'prism' ## 'titan' or 'prism'
MAG = 10
PATCH_SIZE = 1024
TSR_STRUCTURE= 'SurvivalNet' ## TSR structure. options: 'RankModel', 'DeepSurv', 'SurvivalNet', 'SimpleSurv'

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
MIXED_COLS = ["pT_stage"] ## pT_stage has values such 2, 2a, 2b, 3 etc. these needs to be changed to values like 2.0, 2.1, 2.2, 3.0 etc respectively

# === Paths ===
if TASK == 1:
    if EMBEDDER == "titan":
        PATCH_SIZE = 512
        WSI_FEATURE_PATH = f"/home/u1970167/chimera/task1/pathology/features/titan/Task1_titan_{MAG}x_{PATCH_SIZE}_embeddings.csv"
    elif EMBEDDER == 'prism':
        PATCH_SIZE = 224
        WSI_FEATURE_PATH = f"/home/u1970167/chimera/task1/pathology/features/prism/Task1_prism_{MAG}x_{PATCH_SIZE}_embeddings.csv"

    CLINICAL_PATH = "/home/u1970167/chimera/task1/clinical_data.csv"
    TIME_COL = 'time_to_follow-up/BCR'
    EVENT_COL = 'BCR'
    SLIDE_ID_COL = 'Slide_ID'
    MRI_FEATURE_DIR = "/home/u1970167/chimera/task1/radiology/features/"
elif TASK == 3:
    if EMBEDDER == 'titan':
        WSI_FEATURE_PATH = "/home/u1970167/chimera/task3/pathology/features/titan/Task3_TITAN_512_embeddings.csv"
    elif EMBEDDER == 'prism':
        WSI_FEATURE_PATH = "/home/u1970167/chimera/task3/pathology/features/prism/Task3_prism_224_embeddings.csv"

    CLINICAL_PATH = "/home/u1970167/chimera/task3/clinical/features/task3_clinical.csv"
    TIME_COL = 'Time_to_prog_or_FUend'
    EVENT_COL = 'progression'
    SLIDE_ID_COL = 'slide_id'
    MRI_FEATURE_DIR = "Features/task3_mri_features/"

FOLDS_DIR = "/home/u1970167/chimera/task1/experiments/folds/"
EXCLUDE_COLS = ["Case_ID", TIME_COL, EVENT_COL, SLIDE_ID_COL]  # Make sure these columns are not used as features

M_FEATURE_DIM = 2048 ## MRI features dimension
APPLY_ROI = True
VERBOSE = True

# === inference ===
USE_ENSEMBLE = True  # <-- Set to False to use best single model
ENSEMBLE_FOLDS = [0, 1, 2, 3, 4]
RUN = 0  # Only used when USE_ENSEMBLE = False

# === global path ====
RESULT_DIR = f"/home/u1970167/chimera/task1/experiments/TSR_results/TASK_{TASK}_CLINICAL_{USE_CLINICAL}_MRI_{USE_MRI}_WSI_{USE_WSI}_MAGNIF_{MAG}_PATCH_SZ_{PATCH_SIZE}_EMBEDDER_{EMBEDDER}_TSR_{TSR_STRUCTURE}_TUNING_{TUNING_TRIALS}/"