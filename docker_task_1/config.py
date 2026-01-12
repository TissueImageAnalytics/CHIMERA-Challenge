# config.py

# === common =====
SCALE_DATA = True
TASK = 1 ## 1: prostate

# === Data paths ===
CLINICAL_CSV = "/home/u1970167/chimera/task1/clinical_data.csv"        # Clinical features CSV with Case_ID column

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

# 'age_at_prostatectomy', 
# 'primary_gleason', 
# 'secondary_gleason', 
# 'tertiary_gleason', 
# 'ISUP', 
# 'pre_operative_PSA', 
# 'pT_stage', 
# 'positive_lymph_nodes', 
# 'capsular_penetration', 
# 'positive_surgical_margins', 
# 'invasion_seminal_vesicles', 
# 'lymphovascular_invasion', 
# 'earlier_therapy'

MIXED_COLS = ["pT_stage"] ## pT_stage has values such 2, 2a, 2b, 3 etc. these needs to be changed to values like 2.0, 2.1, 2.2, 3.0 etc repectively

# === Experiment settings ===
USE_CLINICAL_FEATURES = True
USE_MRI_FEATURES = False
USE_WSI_FEATURES = False

SURVIVAL_MODEL = 'deephit'  # Options: 'cox' or 'deephit'
FUSION_TYPE = 'linear'  # Options: 'modality' (softmax weights per modality) or 'linear' (linear layer after concat) or 'simple' (concat with no learnable params)
DEEPHIT_LOSS = 'uncensored' # 'censored' or 'uncensored'. 'censored' has a extra term for accounting for censored data whereas 'uncensored' only considers uncensored cases
TIME_BINS = 30  # Only for deephit
NUM_FOLDS = 5
SEED = 42

CLINICAL_DIM = len(CLINICAL_FEATURES)

# === Inference ===
USE_ENSEMBLE = True  # True means ensemble the results of the best models from the 5 folds. False means use the best of the 5 folds

GLOBAL_DIR = f"/home/u1970167/chimera/task1/experiments/results/task1_submission_clinical/" ## main path for results