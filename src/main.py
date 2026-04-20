from config import TARGET_FAULT
from data_loader import load_and_filter_data
from preprocess import run_preprocessing, save_preprocessed_data, save_scaler
from model import train_model, save_trained_model
from evaluate import evaluate_model

def run_pipeline():
    print(f"  STARTING DYNAMIC TEP PIPELINE FOR FAULT {TARGET_FAULT}")
    
    # Load Data
    train_norm, test_norm, train_fault, test_fault = load_and_filter_data(TARGET_FAULT)
    
    # Preprocess & Select Features
    X_train, X_test, y_train, y_test, scaler = run_preprocessing(
        train_norm, test_norm, train_fault, test_fault, TARGET_FAULT
    )
    
    # Save Artifacts & Preprocessed Data
    save_scaler(scaler)
    save_preprocessed_data(X_train, y_train, X_test, y_test)
    
    # Train Model
    model = train_model(X_train, y_train)
    
    # Save Model
    save_trained_model(model)
    
    # Evaluate
    evaluate_model(model, X_test, y_test)
    
    print(f"  PIPELINE COMPLETE FOR FAULT {TARGET_FAULT}")

if __name__ == "__main__":
    run_pipeline()