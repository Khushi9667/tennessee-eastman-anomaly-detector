import os

# ==========================================
# PIPELINE CONTROL
# ==========================================
# Change this variable to run the analysis for a different fault.
faults = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
TARGET_FAULT = 20

# ==========================================
# PATH CONFIGURATION
# ==========================================
# Base directory of the project (assuming config.py is in the root or a src/ folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
RAW_DATA_PATH = os.path.join(BASE_DIR, "../data/raw/")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "../data/processed/")
MODELS_DIR = os.path.join(BASE_DIR, "../models/")

# Create directories if they don't exist
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ==========================================
# PREPROCESSING CONFIGURATION
# ==========================================
# Outlier capping thresholds
LOWER_QUANTILE = 0.025
UPPER_QUANTILE = 0.975

# Features to drop (update this list based on specific fault analysis)
COLUMNS_TO_DROP = [
    'simulationRun', 
    'faultNumber'
]

# ==========================================
# MODEL CONFIGURATION
# ==========================================
# Best hyperparameters found via GridSearchCV

RF_PARAMS = {
    'random_state': 42,
    'class_weight': 'balanced',
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'n_jobs': -1
}
