import pandas as pd
import gc
from sklearn.preprocessing import StandardScaler

def cap_outliers_percentile(df, column_list, lower_q=0.025, upper_q=0.975):
    """
    Caps extreme values at the lower and upper quantiles.
    """
    capped_df = df.copy()
    
    for col in column_list:
        if col.startswith(('xmeas', 'xmv')):
            lower_bound = df[col].quantile(lower_q)
            upper_bound = df[col].quantile(upper_q)
            capped_df[col] = capped_df[col].clip(lower=lower_bound, upper=upper_bound)
            
    return capped_df

def scale_features(train_normal, train_fault, test_normal, test_fault):
    """
    Fits the StandardScaler strictly on the normal training data, 
    then transforms the faulty and testing datasets to simulate real-world baselines.
    """
    # Identify sensor columns
    features = [col for col in train_normal.columns if col.startswith(('xmeas', 'xmv'))]
    scaler = StandardScaler()

    # 1. FIT and TRANSFORM on Cleaned Normal Train Data
    train_normal_scaled = train_normal.copy()
    train_normal_scaled[features] = scaler.fit_transform(train_normal[features])

    # 2. TRANSFORM the Train Faulty Data
    train_fault_scaled = train_fault.copy()
    train_fault_scaled[features] = scaler.transform(train_fault[features])
    
    # 3. TRANSFORM the Test Normal Data (Correction: do not use fit_transform here)
    test_normal_scaled = test_normal.copy()
    test_normal_scaled[features] = scaler.transform(test_normal[features])
    
    # 4. TRANSFORM the Test Faulty Data
    test_fault_scaled = test_fault.copy()
    test_fault_scaled[features] = scaler.transform(test_fault[features])

    # Combine into final dataframes
    final_train_df = pd.concat([train_normal_scaled, train_fault_scaled], axis=0).reset_index(drop=True)
    final_test_df = pd.concat([test_normal_scaled, test_fault_scaled], axis=0).reset_index(drop=True)
    
    # Clean up memory
    del train_normal_scaled, train_fault_scaled, test_normal_scaled, test_fault_scaled
    gc.collect()
    
    return final_train_df, final_test_df, scaler

def prepare_for_modeling(final_train_df, final_test_df, fault_id, drop_columns=None):
    """
    Separates the features (X) and target labels (y), mapping the target 
    to a binary format: 0 for Normal, 1 for the specific fault.
    """
    features = [col for col in final_train_df.columns if col.startswith(('xmeas', 'xmv'))]
    
    if drop_columns:
        features = [f for f in features if f not in drop_columns]

    X_train = final_train_df[features].astype('float32') 
    y_train = final_train_df['faultNumber'].apply(lambda x: 1 if x == fault_id else 0).astype('int8')

    X_test = final_test_df[features].astype('float32') 
    y_test = final_test_df['faultNumber'].apply(lambda x: 1 if x == fault_id else 0).astype('int8')
    
    return X_train, X_test, y_train, y_test

def run_preprocessing(train_normal, test_normal, train_fault, test_fault, fault_id, features_to_drop=None):
    """
    Main execution function to orchestrate the preprocessing steps.
    """
    print(f"Starting Preprocessing for Fault {fault_id}...")
    
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
    
    # Prepare X and y
    X_train, X_test, y_train, y_test = prepare_for_modeling(
        final_train_df, final_test_df, fault_id, drop_columns=features_to_drop
    )
    print(f"Data prepared! X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler