import argparse
from data_loader import load_and_filter_data
from preprocess import cap_outliers_percentile, scale_features, prepare_for_modeling
from model import train_model
from evaluate import evaluate_model

def run_pipeline(fault_id, model_type):
    print(f"--- Starting Pipeline for Fault {fault_id} ---")
    
    # 1. Load Data
    train_norm, test_norm, train_fault, test_fault = load_and_filter_data(fault_id)
    
    # 2. Preprocess (Capping & Scaling)
    # ... call your preprocess functions here ...
    
    # 3. Train
    model = train_model(X_train, y_train, model_type)
    
    # 4. Evaluate
    evaluate_model(model, X_test, y_test)
    
    print("--- Pipeline Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TEP Dynamic Anomaly Detection Pipeline")
    parser.add_argument("--fault", type=int, required=True, help="Fault number to analyze (e.g., 20)")
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "xgb"], help="Model type to train")
    
    args = parser.parse_args()
    run_pipeline(args.fault, args.model)