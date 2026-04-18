import pandas as pd
import pyreadr
import os

def load_rdata(file_path):
    """
    Helper function to load an .RData file and extract the main DataFrame.
    """
    print(f"Loading {file_path}...")
    try:
        result = pyreadr.read_r(file_path)
        # pyreadr returns a dictionary where keys are the R variable names. 
        # Usually, there's only one key for TEP data, so we grab the first one's values.
        key = list(result.keys())[0] 
        df = result[key]
        return df
    except Exception as e:
        raise FileNotFoundError(f"Failed to load {file_path}. Ensure the file exists and pyreadr is installed. Error: {e}")

def downcast_dtypes(df):
    """
    Downcasts float64 columns to float32 to save memory.
    """
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    return df

def load_and_filter_data(fault_id, raw_data_path="../data/raw/"):
    """
    Loads the normal datasets and the faulty datasets, filtering the 
    faulty data dynamically based on the requested fault_id.
    """
    print(f"--- Fetching Data for Fault {fault_id} ---")
    
    # Define file paths (these can be moved to config.py later)
    train_normal_path = os.path.join(raw_data_path, 'TEP_FaultFree_Training.RData')
    test_normal_path = os.path.join(raw_data_path, 'TEP_FaultFree_Testing.RData')
    train_faulty_path = os.path.join(raw_data_path, 'TEP_Faulty_Training.RData')
    test_faulty_path = os.path.join(raw_data_path, 'TEP_Faulty_Testing.RData')

    # 1. Load Normal Data
    train_normal = load_rdata(train_normal_path)
    test_normal = load_rdata(test_normal_path)

    # 2. Load Faulty Data (The entire datasets)
    train_faulty_full = load_rdata(train_faulty_path)
    test_faulty_full = load_rdata(test_faulty_path)

    # 3. Filter Faulty Data for the specific fault_id
    print(f"Filtering faulty data specifically for Fault Number: {fault_id}...")
    train_faulty = train_faulty_full[train_faulty_full['faultNumber'] == fault_id].copy()
    test_faulty = test_faulty_full[test_faulty_full['faultNumber'] == fault_id].copy()

    # 4. Memory Optimization
    print("Downcasting float64 to float32 to optimize memory usage...")
    train_normal = downcast_dtypes(train_normal)
    test_normal = downcast_dtypes(test_normal)
    train_faulty = downcast_dtypes(train_faulty)
    test_faulty = downcast_dtypes(test_faulty)

    print(f"Data loading complete!")
    print(f"Normal Train: {train_normal.shape} | Normal Test: {test_normal.shape}")
    print(f"Faulty Train: {train_faulty.shape} | Faulty Test: {test_faulty.shape}")

    return train_normal, test_normal, train_faulty, test_faulty