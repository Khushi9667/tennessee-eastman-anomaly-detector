import pandas as pd
import numpy as np
import gc
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Import dynamic preprocessing settings from config
from config import LOWER_QUANTILE, UPPER_QUANTILE, COLUMNS_TO_DROP, PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH, SCALER_SAVE_PATH

def cap_outliers_percentile(df, column_list):
    """
    Caps extreme values using quantiles defined in config.py.
    """
    capped_df = df.copy()
    
    for col in column_list:
        if col.startswith(('xmeas', 'xmv')):
            lower_bound = df[col].quantile(LOWER_QUANTILE)
            upper_bound = df[col].quantile(UPPER_QUANTILE)
            capped_df[col] = capped_df[col].clip(lower=lower_bound, upper=upper_bound)
            
    return capped_df

def scale_features(train_normal, train_fault, test_normal, test_fault):
    """
    Fits the StandardScaler strictly on the normal training data, 
    then transforms all datasets to simulate real-world baselines.
    """
    features = [col for col in train_normal.columns if col.startswith(('xmeas', 'xmv'))]
    scaler = StandardScaler()

    # FIT and TRANSFORM on Normal Train Data
    train_normal_scaled = train_normal.copy()
    train_normal_scaled[features] = scaler.fit_transform(train_normal[features])

    # TRANSFORM the Faulty Train Data
    train_fault_scaled = train_fault.copy()
    train_fault_scaled[features] = scaler.transform(train_fault[features])
    
    # TRANSFORM the Normal Test Data
    test_normal_scaled = test_normal.copy()
    test_normal_scaled[features] = scaler.transform(test_normal[features])
    
    # TRANSFORM the Faulty Test Data
    test_fault_scaled = test_fault.copy()
    test_fault_scaled[features] = scaler.transform(test_fault[features])

    # Combine into final dataframes
    final_train_df = pd.concat([train_normal_scaled, train_fault_scaled], axis=0).reset_index(drop=True)
    final_test_df = pd.concat([test_normal_scaled, test_fault_scaled], axis=0).reset_index(drop=True)
    
    # Clean up memory
    del train_normal_scaled, train_fault_scaled, test_normal_scaled, test_fault_scaled
    gc.collect()
    
    return final_train_df, final_test_df, scaler

def prepare_initial_data(final_train_df, final_test_df, fault_id):
    """
    Separates X and y to prepare for the feature selection process.
    """
    features = [col for col in final_train_df.columns if col.startswith(('xmeas', 'xmv'))]
    features = [f for f in features if f not in COLUMNS_TO_DROP]

    X_train_full = final_train_df[features].astype('float32') 
    y_train = final_train_df['faultNumber'].apply(lambda x: 1 if x == fault_id else 0).astype('int8')

    X_test_full = final_test_df[features].astype('float32') 
    y_test = final_test_df['faultNumber'].apply(lambda x: 1 if x == fault_id else 0).astype('int8')
    
    return X_train_full, X_test_full, y_train, y_test

def get_top_k_features(X_train, y_train, k=10):
    """
    Trains a baseline model to extract the top K most important features.
    """
    # A lightweight RF just to gauge feature importance quickly
    rf_baseline = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf_baseline.fit(X_train, y_train)
    
    importances = rf_baseline.feature_importances_
    
    # Sort indices in descending order and grab the top k
    top_k_indices = np.argsort(importances)[::-1][:k]
    top_k_features = X_train.columns[top_k_indices].tolist()
    
    return top_k_features

def remove_redundant_features(X_train_subset, threshold=0.90):
    """
    Calculates a correlation matrix on the selected features. 
    Drops features that have a correlation higher than the threshold.
    """
    # Calculate absolute correlation matrix
    corr_matrix = X_train_subset.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    # Keep the non-redundant features
    final_features = [col for col in X_train_subset.columns if col not in to_drop]
    
    return final_features, to_drop

def run_preprocessing(train_normal, test_normal, train_fault, test_fault, fault_id):
    """
    Main execution function orchestrating capping, scaling, and dynamic feature selection.
    """
    print(f"--- Starting Preprocessing for Fault {fault_id} ---")
    
    cols_to_cap = [col for col in train_normal.columns if col not in ['faultNumber', 'simulationRun']]
    
    # Cap Outliers
    train_normal_clean = cap_outliers_percentile(train_normal, cols_to_cap)
    test_normal_clean = cap_outliers_percentile(test_normal, cols_to_cap)
    train_fault_clean = cap_outliers_percentile(train_fault, cols_to_cap)
    test_fault_clean = cap_outliers_percentile(test_fault, cols_to_cap)
    print("Outlier capping complete.")
    
    # Scale Features
    final_train_df, final_test_df, scaler = scale_features(
        train_normal_clean, train_fault_clean, test_normal_clean, test_fault_clean
    )
    print("Feature scaling complete.")
    
    # Initial X, y split
    X_train_full, X_test_full, y_train, y_test = prepare_initial_data(final_train_df, final_test_df, fault_id)
    
    # Find Top 10 Features
    print("Calculating feature importance to find top 10 features...")
    top_10_features = get_top_k_features(X_train_full, y_train, k=10)
    print(f"Top 10 features identified: {top_10_features}")
    
    # Redundancy Check
    print("Checking for redundant features among the top 10...")
    final_features, dropped_redundant = remove_redundant_features(X_train_full[top_10_features], threshold=0.90)
    if dropped_redundant:
        print(f"Dropped redundant features (correlation > 0.90): {dropped_redundant}")
    else:
        print("No redundant features found.")
        
    print(f"Final selected features for modeling: {final_features}")
    
    # Filter final datasets
    X_train_final = X_train_full[final_features]
    X_test_final = X_test_full[final_features]
    
    print(f"Final Data prepared! X_train: {X_train_final.shape}, X_test: {X_test_final.shape}")
    
    return X_train_final, X_test_final, y_train, y_test, scaler

def save_preprocessed_data(X_train, y_train, X_test, y_test):
    """
    Combines features and target, then saves them to CSV files.
    """
    print("--- Saving Preprocessed Data ---")
    
    train_df = X_train.copy()
    train_df['Target'] = y_train
    train_df.to_csv(PROCESSED_TRAIN_PATH, index=False)
    print(f"Saved training data to {PROCESSED_TRAIN_PATH}")
    
    test_df = X_test.copy()
    test_df['Target'] = y_test
    test_df.to_csv(PROCESSED_TEST_PATH, index=False)
    print(f"Saved testing data to {PROCESSED_TEST_PATH}")

# Add this at the bottom of preprocess.py
def save_scaler(scaler):
    """
    Saves the fitted StandardScaler to the directory specified in config.py.
    """
    print(f"--- Saving Scaler ---")
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"Scaler successfully saved to {SCALER_SAVE_PATH}")