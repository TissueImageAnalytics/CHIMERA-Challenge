# config.py

# === common =====
SCALE_DATA = True
TASK = 3 ## 1: prostate, 2: response to treatment (binary), 3: Bladder survival
RUNS = 10 ## number of times to run the 5-folds cross-validation

# === Base line Cox PH model === ## for cox_baseline.py only
COX_BASELINE_DIR = f"/home/u1970167/chimera/task{TASK}/experiments/results/Cox_baseline/"
PENALIZER=0.3
L1_RATIO=0.5

# === Data paths ===
CLINICAL_CSV = f"/home/u1970167/chimera/task{TASK}/clinical/features/task{TASK}_clinical.csv"        # Clinical features CSV with Case_ID column
RNA_CSV = f"/home/u1970167/chimera/task3/RNA_seq/features/task3_rna.csv"
EMBEDDER = 'titan' ## 'titan' or 'prism'

MAG = 10
PATCH_SIZE = 1024

if EMBEDDER == 'titan':  ## Task3_TITAN_20x_512_embeddings.csv
    PATCH_SIZE = 1024
    WSI_FEATURES_CSV = f"/home/u1970167/chimera/task{TASK}/pathology/features/titan/Task{TASK}_titan_{MAG}x_{PATCH_SIZE}_embeddings.csv"  # csv file with WSI-level feature vector (e.g. From CONCH+TITAN embeddings at 20x of 1024 patch size)
if EMBEDDER == 'prism':
    PATCH_SIZE = 896
    WSI_FEATURES_CSV = f"/home/u1970167/chimera/task{TASK}/pathology/features/prism/Task{TASK}_prism_{MAG}x_{PATCH_SIZE}_embeddings.csv"

FOLDS_CSV = f"/home/u1970167/chimera/task{TASK}/experiments/folds/task{TASK}_folds.csv"          # Directory for saving/loading fold CSVs
OUTPUT_DIR = f"/home/u1970167/chimera/task{TASK}/experiments/results/"       # Directory for saving results/models

CLINICAL_JSON_DIR = '/home/u1970167/chimera/task3/clinical/clinical_data/' ## path to the clincal json files for each case. this is needed when doing inference from clinical features from json files instead of a single csv file

# === Event & Time Columns in Clinical Data ===
if TASK == 1:
    EVENT_COLUMN = "BCR"                # Column name for event indicator (1=event, 0=censored)
    TIME_COLUMN = "time_to_follow-up/BCR"  # Column name for time-to-event or censoring
elif TASK == 3:
    EVENT_COLUMN = 'progression'
    TIME_COLUMN = 'Time_to_prog_or_FUend'

# === Clinical Features to Use ===
#CLINICAL_FEATURES = ['age', 'sex', 'tumor', 'stage', 'grade', 'reTUR', 'variant', 'EORTC', 'no_instillations', 'BRS']
CLINICAL_FEATURES = ['age', 'sex', 'stage', 'grade', 'reTUR', 'variant', 'EORTC', 'BRS']

# === Experiment settings ===
USE_CLINICAL_FEATURES = True
USE_RNA_FEATURES = False
USE_WSI_FEATURES = True

# modality drop out during training
DROP_MODALITY = False ## set to false if do not want modality dropout

CLINICAL_DROPOUT=0.2
RNA_DROPOUT = 0.2
WSI_DROPOUT= 0.7

## model drop out
PROJECTION_DROPOUT = 0.1 ## drop out for each projection head for each modality
FUSION_DROPOUT = 0.1 ## drop out for each projection head for each modality

SURVIVAL_MODEL = 'deephit'  # Options: 'cox' or 'deephit' or 'deepsurv'
FUSION_TYPE = 'cross_att'  # Options: 'modality' (softmax weights per modality) or 'linear' (linear layer after concat) or 'simple' (concat with no learnable params),  'cross_att' (AS cross attention fusion)
DEEPHIT_LOSS = 'uncensored' # 'censored' or 'uncensored'. 'censored' has a extra term for accounting for censored data whereas 'uncensored' only considers uncensored cases

TIME_BINS = 30  # Only for deephit

BIN_EDGES = True ## set to true if changing discrete bins to bins with edges so that the times are not pushed to the few last bins
NUM_FOLDS = 5
SEED = 42
HIDDEN_DIM = 128  # or 128, tune as needed

# === Feature dimensions ===
if EMBEDDER == 'prism':
    WSI_FEATURE_DIM = 1280  # deep features from WSIs. Titan: 768, Prism: 1280
elif EMBEDDER == 'titan':
    WSI_FEATURE_DIM = 768  # deep features from WSIs. Titan: 768, Prism: 1280

RNA_FEATURE_DIM = 19359 ## 19359
RNA_DIM_REDUCE_TO = 128 ## number for reduction of the rna features. If not reducing then set to the same as RNA_FEATURE_DIM
CLINICAL_DIM = len(CLINICAL_FEATURES)

AGGREG_CASE_WSI = False ## False: just select the first WSI in alphabetical order for reproducibility in cross-validations. True: mean aggregates the multiple WSIs per case.

# === Training settings ===
EPOCHS = 500 #112
LR = 1e-4
WD = 1e-3
BATCH_SIZE = 16
PATIENCE = 50 ## early-stop on no improvement to c-index

FULL_BATCH = True   # False to use mini-batch of BATCH_SIZE

# === Debug ===
VERBOSE = True

# === Inference ===
USE_ENSEMBLE = True  # True means ensemble the results of the best models from the 5 folds across the N runs. False means use the best of the 5 folds across the N runs. Currently, only ENSEMBLE is implemented so do not set to False
# INFER_LOCAL = True ## True means: use the features in the form of csv files rather than Challenge expected json (for clinical), wsi_path (for WSIs) and mri_path (for MRI files);
#                    ## False means: Challenge expected json (for clinical), wsi_path (for WSIs) and mri_path (for MRI files); This is currenly only implemented for clinical features.
#                    ## To verify the local code works the same in the docker container, compare scores for 1 or 2 cases. the scores are saved to the results folder to a csv file _train_predictions.csv
INFER_SINGLE = False ## True means: print score for a single file using the Challenge expected interface i.e. using paths to the files for a single case to generate score for a single case.
                    ## Note: the INFER_SINGLE will not produce c-index but only print the score of the first case from the csv file CLINICAL_CSV

if SURVIVAL_MODEL == 'cox':
    GLOBAL_DIR = f"{OUTPUT_DIR}clin_{USE_CLINICAL_FEATURES}_rna_{USE_RNA_FEATURES}_wsi_{USE_WSI_FEATURES}_model_CoxPH/"
else:
    GLOBAL_DIR = f"{OUTPUT_DIR}clin_{USE_CLINICAL_FEATURES}_rna_{USE_RNA_FEATURES}_wsi_{USE_WSI_FEATURES}_embed_{EMBEDDER}_model_{SURVIVAL_MODEL}_fuse_{FUSION_TYPE}_loss_{DEEPHIT_LOSS}_scale_{SCALE_DATA}_ep_{EPOCHS}_ModDrop_{DROP_MODALITY}_binEdg_{BIN_EDGES}/" ## main path for results
#GLOBAL_DIR = f"{OUTPUT_DIR}task3_submission1_clin_wsi_xatt_modDrop"
GLOBAL_DIR = f"{OUTPUT_DIR}task3_submission3_clin_wsi"
