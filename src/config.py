import os

# ==========================================
# PIPELINE CONTROL
# ==========================================
# Change this variable to run the analysis for a different fault.
FAULTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
TARGET_FAULT = 7

# ==========================================
# DIRECTORY CONFIGURATION
# ==========================================
# Base directory of this script (the src/ folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Root directory of the project (one level up from src/)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Main directories
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")

# Ensure output directories exist
for directory in [PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ==========================================
# FILE PATH CONFIGURATION
# ==========================================
# Input Files (Raw Data)
TRAIN_NORMAL_PATH = os.path.join(RAW_DATA_DIR, 'TEP_FaultFree_Training.RData')
TEST_NORMAL_PATH = os.path.join(RAW_DATA_DIR, 'TEP_FaultFree_Testing.RData')
TRAIN_FAULTY_PATH = os.path.join(RAW_DATA_DIR, 'TEP_Faulty_Training.RData')
TEST_FAULTY_PATH = os.path.join(RAW_DATA_DIR, 'TEP_Faulty_Testing.RData')

# Output Files (Artifacts & Reports - Dynamically named by Fault ID)
# Output Files (Processed Data - Dynamically named by Fault ID)
PROCESSED_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, f'processed_training_data_fault_{TARGET_FAULT}.csv')
PROCESSED_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, f'processed_testing_data_fault_{TARGET_FAULT}.csv')

MODEL_SAVE_PATH = os.path.join(MODELS_DIR, f'rf_model_fault_{TARGET_FAULT}.pkl')
SCALER_SAVE_PATH = os.path.join(MODELS_DIR, f'scaler_fault_{TARGET_FAULT}.pkl')

METRICS_SAVE_PATH = os.path.join(REPORTS_DIR, f'metrics_fault_{TARGET_FAULT}.json')
CM_PLOT_SAVE_PATH = os.path.join(REPORTS_DIR, f'confusion_matrix_fault_{TARGET_FAULT}.png')

# ==========================================
# PREPROCESSING CONFIGURATION
# ==========================================
# Outlier capping thresholds
LOWER_QUANTILE = 0.025
UPPER_QUANTILE = 0.975

# Features to drop 
COLUMNS_TO_DROP = [
    'simulationRun', 
    'faultNumber',
    'sample'
]

# Sensors
FEATURES = [
    'xmeas_1', 'xmeas_2', 'xmeas_3', 'xmeas_4', 'xmeas_5', 'xmeas_6', 'xmeas_7', 'xmeas_8', 'xmeas_9', 'xmeas_10',
    'xmeas_11', 'xmeas_12', 'xmeas_13', 'xmeas_14', 'xmeas_15', 'xmeas_16', 'xmeas_17', 'xmeas_18', 'xmeas_19', 'xmeas_20',
    'xmeas_21', 'xmeas_22', 'xmeas_23', 'xmeas_24', 'xmeas_25', 'xmeas_26', 'xmeas_27', 'xmeas_28', 'xmeas_29', 'xmeas_30',
    'xmeas_31', 'xmeas_32', 'xmeas_33', 'xmeas_34', 'xmeas_35', 'xmeas_36', 'xmeas_37', 'xmeas_38', 'xmeas_39', 'xmeas_40', 
    'xmeas_41', 'xmv_1', 'xmv_2', 'xmv_3', 'xmv_4', 'xmv_5', 'xmv_6', 'xmv_7', 'xmv_8', 'xmv_9', 'xmv_10'
]

# ==========================================
# MODEL CONFIGURATION (Random Forest)
# ==========================================
RF_PARAMS = {
    'random_state': 42,
    'class_weight': 'balanced',
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'n_jobs': -1
}