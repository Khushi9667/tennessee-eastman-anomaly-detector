import os

# ==========================================
# PIPELINE CONTROL
# ==========================================
# Change this variable to run the analysis for a different fault.
faults = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
TARGET_FAULT = 20

# ==========================================
# DIRECTORY CONFIGURATION
# ==========================================
# Base directory of the project 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Main directories
RAW_DATA_DIR = os.path.join(BASE_DIR, "../data/raw/")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "../data/processed/")
MODELS_DIR = os.path.join(BASE_DIR, "../models/")
REPORTS_DIR = os.path.join(BASE_DIR, "../reports/")

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
    'faultNumber'
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