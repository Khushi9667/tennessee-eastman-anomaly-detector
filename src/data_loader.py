import pandas as pd
import pyreadr
from config import (
    TRAIN_NORMAL_PATH, 
    TEST_NORMAL_PATH, 
    TRAIN_FAULTY_PATH, 
    TEST_FAULTY_PATH
)

def load_rdata(file_path):
    """
    Helper function to load an .RData file and extract the main DataFrame.
    """
    print(f"Loading {file_path}...")
    try:
        result = pyreadr.read_r(file_path)
        key = list(result.keys())[0] 
        df = result[key]
        return df
    except Exception as e:
        raise FileNotFoundError(f"Failed to load {file_path}. Error: {e}")

def downcast_dtypes(df):
    """
    Downcasts float64 columns to float32 to save memory.
    """
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    return df

def load_and_filter_data(fault_id):
    """
    Loads normal datasets and faulty datasets, filtering the 
    faulty data dynamically based on the requested fault_id.
    """
    print(f"--- Fetching Data for Fault {fault_id} ---")

    # Load data using paths directly from config.py
    train_normal = load_rdata(TRAIN_NORMAL_PATH)
    test_normal = load_rdata(TEST_NORMAL_PATH)
    train_faulty_full = load_rdata(TRAIN_FAULTY_PATH)
    test_faulty_full = load_rdata(TEST_FAULTY_PATH)

    # Filter Faulty Data
    print(f"Filtering faulty data specifically for Fault Number: {fault_id}...")
    train_faulty = train_faulty_full[train_faulty_full['faultNumber'] == fault_id].copy()
    test_faulty = test_faulty_full[test_faulty_full['faultNumber'] == fault_id].copy()

    # Memory Optimization
    print("Downcasting float64 to float32 to optimize memory usage...")
    train_normal = downcast_dtypes(train_normal)
    test_normal = downcast_dtypes(test_normal)
    train_faulty = downcast_dtypes(train_faulty)
    test_faulty = downcast_dtypes(test_faulty)

    print(f"Data loading complete!")
    return train_normal, test_normal, train_faulty, test_faulty