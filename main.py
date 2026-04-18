from config import TARGET_FAULT
from data_loader import load_and_filter_data
from preprocess import run_preprocessing, save_preprocessed_data
from model import train_model, save_trained_model

def run_pipeline():
    print(f"==================================================")
    print(f"  STARTING DYNAMIC TEP PIPELINE FOR FAULT {TARGET_FAULT}")
    print(f"==================================================")
    
    # 1. Load Data
    train_norm, test_norm, train_fault, test_fault = load_and_filter_data(TARGET_FAULT)
    
    # 2. Preprocess
    X_train, X_test, y_train, y_test, scaler = run_preprocessing(
        train_norm, test_norm, train_fault, test_fault, TARGET_FAULT
    )
    
    # 3. Save Preprocessed Data to CSV
    save_preprocessed_data(X_train, y_train, X_test, y_test)
    
    # 4. Train Model
    model = train_model(X_train, y_train)
    
    # 5. Save Model
    save_trained_model(model)
    
    print("==================================================")
    print(f"  PIPELINE COMPLETE FOR FAULT {TARGET_FAULT}")
    print(f"==================================================")

if __name__ == "__main__":
    run_pipeline()